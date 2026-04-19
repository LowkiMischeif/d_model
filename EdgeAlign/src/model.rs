//! Hooked Pythia/GPT-NeoX model implementation.
//!
//! Candle does not expose a hook API for Pythia attention heads, so this file
//! implements the small GPT-NeoX forward pass directly and inserts one hook:
//! after the attention output projection and before the residual addition.

use anyhow::{bail, Context, Result};
use candle::{DType, Device, Module, Tensor, D};
use candle_nn::{embedding, linear_b, Embedding, LayerNorm, Linear, VarBuilder};
use hf_hub::api::sync::Api;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

pub const NUM_LAYERS: usize = 6;
pub const NUM_HEADS: usize = 8;
pub const HEAD_DIM: usize = 64;
pub const HIDDEN_DIM: usize = NUM_HEADS * HEAD_DIM;

/// Model precision mode used by the experiment.
///
/// `Int8` is intentionally not reported as native Candle INT8 weight inference.
/// The current Pythia path loads safetensors through regular Candle tensors, so
/// `Int8` means f32 weights plus dynamic fake-quantized activations. See
/// `fake_quantize_int8_activation` for the exact approximation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum Precision {
    F32,
    F16,
    Int8,
}

impl Precision {
    /// Candle dtype used to load model weights for this precision mode.
    pub fn load_dtype(self) -> DType {
        match self {
            Self::F32 | Self::Int8 => DType::F32,
            Self::F16 => DType::F16,
        }
    }

    /// Stable lowercase label used in JSON and progress logs.
    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Int8 => "int8",
        }
    }

    fn uses_fake_int8(self) -> bool {
        matches!(self, Self::Int8)
    }
}

/// Activations captured from one model forward pass.
///
/// `attn_out[layer][head]` has shape `[seq_len, head_dim]`. `final_logits` is
/// the next-token logits vector at the final sequence position.
#[derive(Clone)]
pub struct ActivationCache {
    pub attn_out: Vec<Vec<Tensor>>,
    pub final_logits: Tensor,
}

/// One surgical intervention at a specific `(layer, head)` site.
#[derive(Clone)]
pub struct Patch {
    pub layer: usize,
    pub head: usize,
    pub value: PatchValue,
}

/// The value used to replace a head's attention output.
#[derive(Clone)]
pub enum PatchValue {
    /// Replace the head output with zeros.
    Zero,
    /// Replace with a precomputed reference mean activation.
    Mean(Tensor),
    /// Replace with another run's activation, used for f32-into-f16 repair.
    Replace(Tensor),
}

/// User-facing wrapper around the hooked Pythia model and tokenizer.
pub struct HookedPythia {
    model: GptNeoXForCausalLM,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    precision: Precision,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl HookedPythia {
    /// Downloads model files from Hugging Face and loads them at the requested dtype.
    ///
    /// This keeps the original API intact for f32/f16 callers. Use
    /// `load_precision` for the simulated INT8 path.
    pub fn load(model_id: &str, device: &Device, dtype: DType) -> Result<Self> {
        let precision = match dtype {
            DType::F16 => Precision::F16,
            DType::F32 => Precision::F32,
            other => bail!("unsupported Pythia load dtype {other:?}; use F32, F16, or load_precision(..., Precision::Int8)"),
        };
        Self::load_precision(model_id, device, precision)
    }

    /// Downloads model files and loads the requested precision mode.
    ///
    /// For `Precision::Int8`, this deliberately loads f32 weights and enables
    /// fake activation quantization. Candle's quantized loader is GGUF/QTensor
    /// oriented and does not directly load this safetensors GPT-NeoX checkpoint
    /// as native int8 weights.
    pub fn load_precision(model_id: &str, device: &Device, precision: Precision) -> Result<Self> {
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
        let dtype = precision.load_dtype();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, device)? };
        let model = GptNeoXForCausalLM::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            dtype,
            precision,
            num_layers: config.num_hidden_layers,
            num_heads: config.num_attention_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
        })
    }

    /// Returns the dtype the model weights were loaded with.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Stable lowercase precision label used in output records.
    pub fn precision_label(&self) -> &'static str {
        self.precision.label()
    }

    /// Human-readable note describing what this precision mode means.
    pub fn precision_description(&self) -> &'static str {
        match self.precision {
            Precision::F32 => "native f32 Candle tensor execution",
            Precision::F16 => "native f16 Candle tensor execution",
            Precision::Int8 => {
                "simulated dynamic activation int8: f32 weights, selected activations quantized to int8 levels and dequantized before continuing"
            }
        }
    }

    /// Returns the Candle device used for model execution.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Tokenizes text as a batch of size 1: shape `[1, seq_len]`.
    pub fn tokenize(&self, text: &str) -> Result<Tensor> {
        let ids = self.encode_ids(text)?;
        if ids.is_empty() {
            bail!("tokenizer produced no tokens for {text:?}");
        }
        Ok(Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?)
    }

    /// Encodes text using the tokenizer default of adding special tokens.
    pub fn encode_ids(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_ids_with_special_tokens(text, true)
    }

    /// Encodes text while explicitly controlling special-token insertion.
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

    /// Decodes a single token ID for readable validation logs.
    pub fn decode_token(&self, token_id: u32) -> Result<String> {
        self.tokenizer
            .decode(&[token_id], true)
            .map_err(anyhow::Error::msg)
    }

    /// Runs a normal forward pass and returns all cached per-head activations.
    pub fn forward_with_cache(&self, input_ids: &Tensor) -> Result<ActivationCache> {
        Ok(self.model.forward(input_ids, &[], true, self.precision)?)
    }

    /// Runs a forward pass with one or more attention-head replacements.
    pub fn forward_with_patches(&self, input_ids: &Tensor, patches: &[Patch]) -> Result<Tensor> {
        Ok(self
            .model
            .forward(input_ids, patches, false, self.precision)?
            .final_logits)
    }

    /// Computes `correct_logit - best_other_logit`.
    ///
    /// This is the scalar task metric used by ablation and repair.
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

    /// Returns the highest-logit token ID.
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

/// Converts any tensor to a CPU `Vec<f32>` for scalar metric calculations.
pub fn tensor_to_vec_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

/// Subset of the Hugging Face GPT-NeoX config needed by Pythia-70M.
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
    /// Ensures the downloaded model matches the architecture assumed by hooks.
    fn validate(&self) -> Result<()> {
        if self.hidden_size != HIDDEN_DIM {
            bail!(
                "expected hidden_size {HIDDEN_DIM}, got {}",
                self.hidden_size
            );
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
            bail!(
                "rotary dimensions must be even, got {}",
                self.rotary_ndims()
            );
        }
        Ok(())
    }

    /// Per-head hidden size.
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Number of dimensions in each head that receive rotary embeddings.
    fn rotary_ndims(&self) -> usize {
        (self.head_dim() as f32 * self.rotary_pct) as usize
    }
}

/// Loads a layer norm from Hugging Face-style `weight` and `bias` tensors.
fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> candle::Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = vb.get(size, "bias")?;
    Ok(LayerNorm::new(weight, bias, eps))
}

/// Minimal GPT-NeoX causal language model for Pythia-70M.
#[derive(Clone)]
struct GptNeoXForCausalLM {
    embed_in: Embedding,
    layers: Vec<GptNeoXLayer>,
    final_layer_norm: LayerNorm,
    embed_out: Linear,
}

impl GptNeoXForCausalLM {
    /// Loads embeddings, transformer blocks, final norm, and output head.
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

    /// Shared forward path used for clean, cached, and patched executions.
    ///
    /// `cache_activations` controls whether every attention head is saved.
    /// `patches` controls whether selected heads are replaced before the
    /// attention output is added to the residual stream.
    fn forward(
        &self,
        input_ids: &Tensor,
        patches: &[Patch],
        cache_activations: bool,
        precision: Precision,
    ) -> candle::Result<ActivationCache> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        if batch_size != 1 {
            candle::bail!("only batch size 1 is supported, got {batch_size}");
        }

        let mut hidden_states = self.embed_in.forward(input_ids)?;
        hidden_states = maybe_fake_quantize_int8_activation(&hidden_states, precision)?;
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

        // Each transformer block receives the same causal mask. No KV cache is
        // used because experiments run full prompts rather than generation.
        for layer in &self.layers {
            let mut layer_cache = if cache_activations {
                Some(Vec::with_capacity(NUM_HEADS))
            } else {
                None
            };
            hidden_states = layer.forward(
                &hidden_states,
                causal_mask.as_ref(),
                patches,
                &mut layer_cache,
                precision,
            )?;
            if let Some(layer_cache) = layer_cache {
                attn_out.push(layer_cache);
            }
        }

        let hidden_states = self.final_layer_norm.forward(&hidden_states)?;
        let hidden_states = maybe_fake_quantize_int8_activation(&hidden_states, precision)?;
        let last_hidden = hidden_states.narrow(1, seq_len - 1, 1)?;
        let final_logits = self
            .embed_out
            .forward(&last_hidden)?
            .squeeze(1)?
            .squeeze(0)?;
        Ok(ActivationCache {
            attn_out,
            final_logits,
        })
    }
}

/// One Pythia transformer block: layer norms, attention, and MLP.
#[derive(Clone)]
struct GptNeoXLayer {
    layer_idx: usize,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    attention: GptNeoXAttention,
    mlp: GptNeoXMlp,
}

impl GptNeoXLayer {
    /// Loads one transformer block from its `gpt_neox.layers.N` prefix.
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

    /// Runs a parallel-residual GPT-NeoX block.
    ///
    /// The hook is applied to `attn_output` before it is added to the residual.
    fn forward(
        &self,
        hidden_states: &Tensor,
        causal_mask: Option<&Tensor>,
        patches: &[Patch],
        layer_cache: &mut Option<Vec<Tensor>>,
        precision: Precision,
    ) -> candle::Result<Tensor> {
        // GPT-NeoX/Pythia uses parallel residuals: attention and MLP both read
        // from the original residual stream, so attention patches do not change
        // the same-layer MLP input.
        let attn_input = self.input_layernorm.forward(hidden_states)?;
        let mlp_input = self.post_attention_layernorm.forward(hidden_states)?;
        let attn_output = self
            .attention
            .forward(&attn_input, causal_mask, precision)?;
        let attn_output = handle_head_hooks(
            &attn_output,
            self.layer_idx,
            patches,
            layer_cache,
            NUM_HEADS,
            HEAD_DIM,
        )?;
        let mlp_output = self.mlp.forward(&mlp_input, precision)?;
        let hidden_states = (hidden_states + &attn_output)?;
        let hidden_states = (&hidden_states + &mlp_output)?;
        maybe_fake_quantize_int8_activation(&hidden_states, precision)
    }
}

/// Multi-head self-attention with GPT-NeoX QKV packing and rotary embeddings.
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
    /// Loads the fused QKV projection and output projection.
    fn load(vb: VarBuilder, config: &GptNeoXConfig) -> candle::Result<Self> {
        let hidden_size = config.hidden_size;
        let query_key_value =
            linear_b(hidden_size, 3 * hidden_size, true, vb.pp("query_key_value"))?;
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

    /// Computes attention output with shape `[batch, seq_len, hidden_dim]`.
    fn forward(
        &self,
        hidden_states: &Tensor,
        causal_mask: Option<&Tensor>,
        precision: Precision,
    ) -> candle::Result<Tensor> {
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
        let attn_output = attn_probs.matmul(&value)?.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;
        let attn_output = self.dense.forward(&attn_output)?;
        maybe_fake_quantize_int8_activation(&attn_output, precision)
    }

    /// Applies rotary embeddings to the rotary portion of query and key heads.
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

/// Rotary embedding state for GPT-NeoX attention.
#[derive(Clone)]
struct RotaryEmbedding {
    inv_freq: Tensor,
}

impl RotaryEmbedding {
    /// Builds inverse frequencies used to generate sin/cos position tables.
    fn new(rotary_ndims: usize, base: f32, device: &Device) -> candle::Result<Self> {
        let inv_freq: Vec<f32> = (0..rotary_ndims)
            .step_by(2)
            .map(|idx| 1.0 / base.powf(idx as f32 / rotary_ndims as f32))
            .collect();
        Ok(Self {
            inv_freq: Tensor::new(inv_freq.as_slice(), device)?,
        })
    }

    /// Rotates query and key tensors according to their sequence positions.
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

/// Implements the GPT-NeoX rotary helper: `[-x2, x1]`.
fn rotate_half(xs: &Tensor) -> candle::Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let x1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let x2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

/// Feed-forward network inside each GPT-NeoX block.
#[derive(Clone)]
struct GptNeoXMlp {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
}

impl GptNeoXMlp {
    /// Loads the up-projection and down-projection MLP weights.
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

    /// Runs GELU MLP: hidden -> intermediate -> hidden.
    fn forward(&self, xs: &Tensor, precision: Precision) -> candle::Result<Tensor> {
        let xs = self.dense_h_to_4h.forward(xs)?.gelu()?;
        let xs = self.dense_4h_to_h.forward(&xs)?;
        maybe_fake_quantize_int8_activation(&xs, precision)
    }
}

/// Applies the project's simulated INT8 activation quantization.
///
/// This is dynamic per-tensor symmetric fake quantization:
/// 1. compute `scale = max(abs(x)) / 127`
/// 2. round/clamp `x / scale` to the signed int8 range `[-127, 127]`
/// 3. multiply by `scale` to dequantize back to a normal Candle float tensor
///
/// No int8 matmul kernels or int8 weight storage are used here. The goal is a
/// reproducible fallback for circuit-level comparisons when native INT8
/// safetensors inference is not available for this GPT-NeoX path.
fn maybe_fake_quantize_int8_activation(
    tensor: &Tensor,
    precision: Precision,
) -> candle::Result<Tensor> {
    if !precision.uses_fake_int8() {
        return Ok(tensor.clone());
    }

    let original_dtype = tensor.dtype();
    let tensor_f32 = tensor.to_dtype(DType::F32)?;
    let max_abs = tensor_f32.abs()?.max_all()?.to_scalar::<f32>()?;
    if !max_abs.is_finite() || max_abs == 0.0 {
        return Ok(tensor.clone());
    }

    let scale = (max_abs / 127.0).max(f32::MIN_POSITIVE);
    ((&tensor_f32 / scale as f64)?
        .round()?
        .clamp(-127.0, 127.0)?
        * scale as f64)?
        .to_dtype(original_dtype)
}

/// Builds a boolean upper-triangular causal mask.
fn causal_mask(batch_size: usize, seq_len: usize, device: &Device) -> candle::Result<Tensor> {
    let mask: Vec<u8> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (seq_len, seq_len), device)?
        .broadcast_as((batch_size, 1, seq_len, seq_len))
}

/// Replaces entries selected by `mask` with `on_true`.
fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> candle::Result<Tensor> {
    let on_true = Tensor::new(on_true, on_false.device())?
        .to_dtype(on_false.dtype())?
        .broadcast_as(mask.shape().dims())?;
    mask.where_cond(&on_true, on_false)
}

/// Caches and/or patches per-head attention outputs at the hook site.
///
/// The attention output arrives as `[1, seq_len, hidden_dim]`. This function
/// reshapes it to `[1, seq_len, num_heads, head_dim]`, saves requested head
/// slices, applies any patches for this layer, and reshapes back.
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

    // Cache before patching so `forward_with_cache` records the model's natural
    // activations, not any intervened values.
    if let Some(cache) = layer_cache {
        for head_idx in 0..num_heads {
            let head = attn_heads.narrow(2, head_idx, 1)?.squeeze(2)?.squeeze(0)?;
            cache.push(head);
        }
    }

    if !patches.iter().any(|patch| patch.layer == layer_idx) {
        return Ok(attn_output.clone());
    }

    // Candle tensors are immutable, so replacement is implemented by building a
    // list of head tensors and concatenating them back together.
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

/// Converts a patch enum into a concrete `[seq_len, head_dim]` tensor.
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

/// Moves/casts a replacement tensor and broadcasts it to the current prompt.
///
/// Mean activations are stored as one vector, while repair activations are
/// stored as full sequences. This helper accepts both shapes.
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
