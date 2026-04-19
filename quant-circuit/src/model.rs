use anyhow::{bail, Context, Result};
use candle::{DType, Device, Module, Tensor, D};
use candle_nn::{embedding, linear_b, Embedding, LayerNorm, Linear, VarBuilder};
use hf_hub::api::sync::Api;
use serde::Deserialize;
use tokenizers::Tokenizer;

pub const NUM_LAYERS: usize = 6;
pub const NUM_HEADS: usize = 8;
pub const HEAD_DIM: usize = 64;
pub const HIDDEN_DIM: usize = NUM_HEADS * HEAD_DIM;

#[derive(Clone)]
pub struct ActivationCache {
    pub attn_out: Vec<Vec<Tensor>>,
    pub final_logits: Tensor,
}

#[derive(Clone)]
pub struct Patch {
    pub layer: usize,
    pub head: usize,
    pub value: PatchValue,
}

#[derive(Clone)]
pub enum PatchValue {
    Zero,
    Mean(Tensor),
    Replace(Tensor),
}

pub struct HookedPythia {
    model: GptNeoXForCausalLM,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl HookedPythia {
    pub fn load(model_id: &str, device: &Device, dtype: DType) -> Result<Self> {
        let api = Api::new().context("create Hugging Face Hub API")?;
        let repo = api.model(model_id.to_string());
        let weights = vec![repo
            .get("model.safetensors")
            .with_context(|| format!("download {model_id}/model.safetensors"))?];
        let tokenizer_file = repo
            .get("tokenizer.json")
            .with_context(|| format!("download {model_id}/tokenizer.json"))?;
        let config_file = repo
            .get("config.json")
            .with_context(|| format!("download {model_id}/config.json"))?;

        let config_reader = std::fs::File::open(&config_file)
            .with_context(|| format!("open config file {}", config_file.display()))?;
        let config: GptNeoXConfig =
            serde_json::from_reader(config_reader).context("parse Pythia config")?;
        config.validate()?;

        let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(anyhow::Error::msg)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, device)? };
        let model = GptNeoXForCausalLM::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            dtype,
            num_layers: config.num_hidden_layers,
            num_heads: config.num_attention_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tokenize(&self, text: &str) -> Result<Tensor> {
        let ids = self.encode_ids(text)?;
        if ids.is_empty() {
            bail!("tokenizer produced no tokens for {text:?}");
        }
        Ok(Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?)
    }

    pub fn encode_ids(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_ids_with_special_tokens(text, true)
    }

    pub fn encode_ids_with_special_tokens(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(anyhow::Error::msg)?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode_token(&self, token_id: u32) -> Result<String> {
        self.tokenizer
            .decode(&[token_id], true)
            .map_err(anyhow::Error::msg)
    }

    pub fn forward_with_cache(&self, input_ids: &Tensor) -> Result<ActivationCache> {
        Ok(self.model.forward(input_ids, &[], true)?)
    }

    pub fn forward_with_patches(&self, input_ids: &Tensor, patches: &[Patch]) -> Result<Tensor> {
        Ok(self.model.forward(input_ids, patches, false)?.final_logits)
    }

    pub fn logit_diff(&self, logits: &Tensor, correct_token_id: u32) -> Result<f32> {
        let values = tensor_to_vec_f32(logits)?;
        let correct_idx = correct_token_id as usize;
        if correct_idx >= values.len() {
            bail!(
                "target token id {correct_token_id} outside logits vector of length {}",
                values.len()
            );
        }
        let correct = values[correct_idx];
        let best_other = values
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| (idx != correct_idx).then_some(*value))
            .fold(f32::NEG_INFINITY, f32::max);
        Ok(correct - best_other)
    }

    pub fn argmax_token(&self, logits: &Tensor) -> Result<u32> {
        let values = tensor_to_vec_f32(logits)?;
        let (idx, _) = values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .context("empty logits tensor")?;
        Ok(idx as u32)
    }
}

pub fn tensor_to_vec_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

#[derive(Debug, Clone, Deserialize)]
struct GptNeoXConfig {
    hidden_size: usize,
    intermediate_size: usize,
    layer_norm_eps: f64,
    max_position_embeddings: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    rotary_emb_base: f32,
    rotary_pct: f32,
    use_parallel_residual: bool,
    vocab_size: usize,
}

impl GptNeoXConfig {
    fn validate(&self) -> Result<()> {
        if self.hidden_size != HIDDEN_DIM {
            bail!("expected hidden_size {HIDDEN_DIM}, got {}", self.hidden_size);
        }
        if self.num_attention_heads != NUM_HEADS {
            bail!(
                "expected {NUM_HEADS} attention heads, got {}",
                self.num_attention_heads
            );
        }
        if self.num_hidden_layers != NUM_LAYERS {
            bail!(
                "expected {NUM_LAYERS} layers, got {}",
                self.num_hidden_layers
            );
        }
        if !self.use_parallel_residual {
            bail!("Pythia-70M is expected to use parallel residual blocks");
        }
        if self.max_position_embeddings == 0 {
            bail!("max_position_embeddings must be positive");
        }
        if self.rotary_ndims() % 2 != 0 {
            bail!("rotary dimensions must be even, got {}", self.rotary_ndims());
        }
        Ok(())
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn rotary_ndims(&self) -> usize {
        (self.head_dim() as f32 * self.rotary_pct) as usize
    }
}

fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> candle::Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = vb.get(size, "bias")?;
    Ok(LayerNorm::new(weight, bias, eps))
}

#[derive(Clone)]
struct GptNeoXForCausalLM {
    embed_in: Embedding,
    layers: Vec<GptNeoXLayer>,
    final_layer_norm: LayerNorm,
    embed_out: Linear,
}

impl GptNeoXForCausalLM {
    fn load(vb: VarBuilder, config: &GptNeoXConfig) -> candle::Result<Self> {
        let embed_in = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("gpt_neox.embed_in"),
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(GptNeoXLayer::load(
                vb.pp(format!("gpt_neox.layers.{layer_idx}")),
                config,
                layer_idx,
            )?);
        }
        let final_layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("gpt_neox.final_layer_norm"),
        )?;
        let embed_out = linear_b(
            config.hidden_size,
            config.vocab_size,
            false,
            vb.pp("embed_out"),
        )?;
        Ok(Self {
            embed_in,
            layers,
            final_layer_norm,
            embed_out,
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        patches: &[Patch],
        cache_activations: bool,
    ) -> candle::Result<ActivationCache> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        if batch_size != 1 {
            candle::bail!("only batch size 1 is supported, got {batch_size}");
        }

        let mut hidden_states = self.embed_in.forward(input_ids)?;
        let causal_mask = if seq_len <= 1 {
            None
        } else {
            Some(causal_mask(batch_size, seq_len, input_ids.device())?)
        };

        let mut attn_out = if cache_activations {
            Vec::with_capacity(self.layers.len())
        } else {
            Vec::new()
        };

        for layer in &self.layers {
            let mut layer_cache = if cache_activations {
                Some(Vec::with_capacity(NUM_HEADS))
            } else {
                None
            };
            hidden_states =
                layer.forward(&hidden_states, causal_mask.as_ref(), patches, &mut layer_cache)?;
            if let Some(layer_cache) = layer_cache {
                attn_out.push(layer_cache);
            }
        }

        let hidden_states = self.final_layer_norm.forward(&hidden_states)?;
        let last_hidden = hidden_states.narrow(1, seq_len - 1, 1)?;
        let final_logits = self.embed_out.forward(&last_hidden)?.squeeze(1)?.squeeze(0)?;
        Ok(ActivationCache {
            attn_out,
            final_logits,
        })
    }
}

#[derive(Clone)]
struct GptNeoXLayer {
    layer_idx: usize,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    attention: GptNeoXAttention,
    mlp: GptNeoXMlp,
}

impl GptNeoXLayer {
    fn load(vb: VarBuilder, config: &GptNeoXConfig, layer_idx: usize) -> candle::Result<Self> {
        let input_layernorm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let attention = GptNeoXAttention::load(vb.pp("attention"), config)?;
        let mlp = GptNeoXMlp::load(vb.pp("mlp"), config)?;
        Ok(Self {
            layer_idx,
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        causal_mask: Option<&Tensor>,
        patches: &[Patch],
        layer_cache: &mut Option<Vec<Tensor>>,
    ) -> candle::Result<Tensor> {
        // GPT-NeoX/Pythia uses parallel residuals: attention and MLP both read
        // from the original residual stream, so attention patches do not change
        // the same-layer MLP input.
        let attn_input = self.input_layernorm.forward(hidden_states)?;
        let mlp_input = self.post_attention_layernorm.forward(hidden_states)?;
        let attn_output = self.attention.forward(&attn_input, causal_mask)?;
        let attn_output = handle_head_hooks(
            &attn_output,
            self.layer_idx,
            patches,
            layer_cache,
            NUM_HEADS,
            HEAD_DIM,
        )?;
        let mlp_output = self.mlp.forward(&mlp_input)?;
        let hidden_states = (hidden_states + &attn_output)?;
        &hidden_states + &mlp_output
    }
}

#[derive(Clone)]
struct GptNeoXAttention {
    query_key_value: Linear,
    dense: Linear,
    rotary: RotaryEmbedding,
    num_heads: usize,
    head_dim: usize,
    rotary_ndims: usize,
    scale: f64,
}

impl GptNeoXAttention {
    fn load(vb: VarBuilder, config: &GptNeoXConfig) -> candle::Result<Self> {
        let hidden_size = config.hidden_size;
        let query_key_value = linear_b(
            hidden_size,
            3 * hidden_size,
            true,
            vb.pp("query_key_value"),
        )?;
        let dense = linear_b(hidden_size, hidden_size, true, vb.pp("dense"))?;
        let head_dim = config.head_dim();
        let rotary_ndims = config.rotary_ndims();
        let rotary = RotaryEmbedding::new(rotary_ndims, config.rotary_emb_base, vb.device())?;
        Ok(Self {
            query_key_value,
            dense,
            rotary,
            num_heads: config.num_attention_heads,
            head_dim,
            rotary_ndims,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, causal_mask: Option<&Tensor>) -> candle::Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let qkv = self.query_key_value.forward(hidden_states)?;
        let qkv = qkv.reshape((batch_size, seq_len, self.num_heads, 3 * self.head_dim))?;
        let query = qkv.narrow(D::Minus1, 0, self.head_dim)?;
        let key = qkv.narrow(D::Minus1, self.head_dim, self.head_dim)?;
        let value = qkv.narrow(D::Minus1, 2 * self.head_dim, self.head_dim)?;

        let query = query.transpose(1, 2)?;
        let key = key.transpose(1, 2)?;
        let value = value.transpose(1, 2)?;

        let (query, key) = self.apply_rotary(&query, &key)?;
        let query_f32 = query.to_dtype(DType::F32)?;
        let key_f32 = key.to_dtype(DType::F32)?;
        let mut attn_scores =
            (query_f32.matmul(&key_f32.transpose(D::Minus1, D::Minus2)?)? * self.scale)?;

        if let Some(mask) = causal_mask {
            let mask = mask.broadcast_as(attn_scores.shape())?;
            attn_scores = masked_fill(&attn_scores, &mask, f32::NEG_INFINITY)?;
        }

        let attn_probs =
            candle_nn::ops::softmax(&attn_scores, D::Minus1)?.to_dtype(hidden_states.dtype())?;
        let attn_output = attn_probs
            .matmul(&value)?
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
        self.dense.forward(&attn_output)
    }

    fn apply_rotary(&self, query: &Tensor, key: &Tensor) -> candle::Result<(Tensor, Tensor)> {
        if self.rotary_ndims == 0 {
            return Ok((query.clone(), key.clone()));
        }

        let q_rot = query.narrow(D::Minus1, 0, self.rotary_ndims)?;
        let q_pass = query.narrow(
            D::Minus1,
            self.rotary_ndims,
            self.head_dim - self.rotary_ndims,
        )?;
        let k_rot = key.narrow(D::Minus1, 0, self.rotary_ndims)?;
        let k_pass = key.narrow(
            D::Minus1,
            self.rotary_ndims,
            self.head_dim - self.rotary_ndims,
        )?;
        let (q_rot, k_rot) = self.rotary.forward(&q_rot, &k_rot)?;
        let query = Tensor::cat(&[&q_rot, &q_pass], D::Minus1)?;
        let key = Tensor::cat(&[&k_rot, &k_pass], D::Minus1)?;
        Ok((query, key))
    }
}

#[derive(Clone)]
struct RotaryEmbedding {
    inv_freq: Tensor,
}

impl RotaryEmbedding {
    fn new(rotary_ndims: usize, base: f32, device: &Device) -> candle::Result<Self> {
        let inv_freq: Vec<f32> = (0..rotary_ndims)
            .step_by(2)
            .map(|idx| 1.0 / base.powf(idx as f32 / rotary_ndims as f32))
            .collect();
        Ok(Self {
            inv_freq: Tensor::new(inv_freq.as_slice(), device)?,
        })
    }

    fn forward(&self, query: &Tensor, key: &Tensor) -> candle::Result<(Tensor, Tensor)> {
        let (_, _, seq_len, rotary_ndims) = query.dims4()?;
        let dtype = query.dtype();
        let positions = Tensor::arange(0, seq_len as u32, query.device())?.to_dtype(dtype)?;
        let inv_freq = self.inv_freq.to_dtype(dtype)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos = emb.cos()?.reshape((1, 1, seq_len, rotary_ndims))?;
        let sin = emb.sin()?.reshape((1, 1, seq_len, rotary_ndims))?;
        let query = (query.broadcast_mul(&cos)? + rotate_half(query)?.broadcast_mul(&sin)?)?;
        let key = (key.broadcast_mul(&cos)? + rotate_half(key)?.broadcast_mul(&sin)?)?;
        Ok((query, key))
    }
}

fn rotate_half(xs: &Tensor) -> candle::Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let x1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let x2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

#[derive(Clone)]
struct GptNeoXMlp {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
}

impl GptNeoXMlp {
    fn load(vb: VarBuilder, config: &GptNeoXConfig) -> candle::Result<Self> {
        let dense_h_to_4h = linear_b(
            config.hidden_size,
            config.intermediate_size,
            true,
            vb.pp("dense_h_to_4h"),
        )?;
        let dense_4h_to_h = linear_b(
            config.intermediate_size,
            config.hidden_size,
            true,
            vb.pp("dense_4h_to_h"),
        )?;
        Ok(Self {
            dense_h_to_4h,
            dense_4h_to_h,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        let xs = self.dense_h_to_4h.forward(xs)?.gelu()?;
        self.dense_4h_to_h.forward(&xs)
    }
}

fn causal_mask(batch_size: usize, seq_len: usize, device: &Device) -> candle::Result<Tensor> {
    let mask: Vec<u8> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (seq_len, seq_len), device)?.broadcast_as((
        batch_size, 1, seq_len, seq_len,
    ))
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> candle::Result<Tensor> {
    let on_true = Tensor::new(on_true, on_false.device())?
        .to_dtype(on_false.dtype())?
        .broadcast_as(mask.shape().dims())?;
    mask.where_cond(&on_true, on_false)
}

fn handle_head_hooks(
    attn_output: &Tensor,
    layer_idx: usize,
    patches: &[Patch],
    layer_cache: &mut Option<Vec<Tensor>>,
    num_heads: usize,
    head_dim: usize,
) -> candle::Result<Tensor> {
    let (batch_size, seq_len, hidden_dim) = attn_output.dims3()?;
    if batch_size != 1 {
        candle::bail!("only batch size 1 is supported, got {batch_size}");
    }
    if hidden_dim != num_heads * head_dim {
        candle::bail!(
            "attention output hidden dim {hidden_dim} does not match {num_heads} heads x {head_dim}"
        );
    }
    let attn_heads = attn_output.reshape((batch_size, seq_len, num_heads, head_dim))?;

    if let Some(cache) = layer_cache {
        for head_idx in 0..num_heads {
            let head = attn_heads
                .narrow(2, head_idx, 1)?
                .squeeze(2)?
                .squeeze(0)?;
            cache.push(head);
        }
    }

    if !patches.iter().any(|patch| patch.layer == layer_idx) {
        return Ok(attn_output.clone());
    }

    let mut repaired_heads = Vec::with_capacity(num_heads);
    for head_idx in 0..num_heads {
        let original = attn_heads.narrow(2, head_idx, 1)?;
        if let Some(patch) = patches
            .iter()
            .rev()
            .find(|patch| patch.layer == layer_idx && patch.head == head_idx)
        {
            let replacement = patch_to_head_tensor(
                &patch.value,
                seq_len,
                head_dim,
                attn_output.dtype(),
                attn_output.device(),
            )?
            .unsqueeze(0)?
            .unsqueeze(2)?;
            repaired_heads.push(replacement);
        } else {
            repaired_heads.push(original);
        }
    }

    let refs: Vec<&Tensor> = repaired_heads.iter().collect();
    Tensor::cat(&refs, 2)?.reshape((batch_size, seq_len, num_heads * head_dim))
}

fn patch_to_head_tensor(
    value: &PatchValue,
    seq_len: usize,
    head_dim: usize,
    dtype: DType,
    device: &Device,
) -> candle::Result<Tensor> {
    match value {
        PatchValue::Zero => Tensor::zeros((seq_len, head_dim), dtype, device),
        PatchValue::Mean(tensor) | PatchValue::Replace(tensor) => {
            normalize_patch_tensor(tensor, seq_len, head_dim, dtype, device)
        }
    }
}

fn normalize_patch_tensor(
    tensor: &Tensor,
    seq_len: usize,
    head_dim: usize,
    dtype: DType,
    device: &Device,
) -> candle::Result<Tensor> {
    let tensor = tensor.to_device(device)?.to_dtype(dtype)?;
    let dims = tensor.dims().to_vec();
    match dims.as_slice() {
        [d] if *d == head_dim => tensor.reshape((1, head_dim))?.broadcast_as((seq_len, head_dim)),
        [s, d] if *s == seq_len && *d == head_dim => Ok(tensor),
        [1, d] if *d == head_dim => tensor.broadcast_as((seq_len, head_dim)),
        [1, s, d] if *s == seq_len && *d == head_dim => tensor.squeeze(0),
        dims => candle::bail!(
            "patch tensor has shape {dims:?}; expected [{head_dim}], [1, {head_dim}], or [{seq_len}, {head_dim}]"
        ),
    }
}
