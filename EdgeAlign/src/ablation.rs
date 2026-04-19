//! Attention-head importance scoring.
//!
//! This module asks how much each head matters for a prompt by replacing one
//! head at a time and measuring the drop in the target-token logit difference.

use anyhow::{bail, Result};
use candle::Tensor;
use serde::Serialize;

use crate::model::{tensor_to_vec_f32, HookedPythia, Patch, PatchValue};

/// Importance scores for one prompt.
///
/// `zero_importance` and `mean_importance` both have shape
/// `[num_layers][num_heads]`. Positive values mean the head was helpful:
/// removing/replacing it lowered the target-token logit difference.
#[derive(Clone, Debug, Serialize)]
pub struct ImportanceResult {
    pub prompt: String,
    pub task_type: String,
    pub target_token: String,
    pub clean_logit_diff: f32,
    pub zero_importance: Vec<Vec<f32>>,
    pub mean_importance: Vec<Vec<f32>>,
}

/// Computes average reference activations used by mean-ablation.
///
/// Reference prompts have different lengths, so we average only each head's
/// final-position vector. When used as a patch, the model code broadcasts that
/// vector across the current prompt length.
pub fn compute_mean_activations(
    model: &HookedPythia,
    reference_prompts: &[String],
) -> Result<Vec<Vec<Tensor>>> {
    if reference_prompts.is_empty() {
        bail!("cannot compute mean activations from zero reference prompts");
    }

    let mut sums = vec![vec![vec![0f32; model.head_dim]; model.num_heads]; model.num_layers];
    let mut count = 0usize;

    for prompt in reference_prompts {
        // Cache every head activation for this neutral reference prompt.
        let input_ids = model.tokenize(prompt)?;
        let cache = model.forward_with_cache(&input_ids)?;
        for layer in 0..model.num_layers {
            for head in 0..model.num_heads {
                // Use only the final token position so all reference prompts
                // contribute same-sized vectors.
                let activation = &cache.attn_out[layer][head];
                let seq_len = activation.dim(0)?;
                let last = activation.narrow(0, seq_len - 1, 1)?.squeeze(0)?;
                let values = tensor_to_vec_f32(&last)?;
                for (idx, value) in values.iter().enumerate().take(model.head_dim) {
                    sums[layer][head][idx] += *value;
                }
            }
        }
        count += 1;
    }

    let mut means = Vec::with_capacity(model.num_layers);
    for layer_sums in sums {
        let mut layer_means = Vec::with_capacity(model.num_heads);
        for mut head_sum in layer_sums {
            for value in &mut head_sum {
                *value /= count as f32;
            }
            layer_means.push(Tensor::from_vec(
                head_sum,
                (model.head_dim,),
                model.device(),
            )?);
        }
        means.push(layer_means);
    }
    Ok(means)
}

/// Computes zero-ablation and mean-ablation importance for one prompt.
///
/// The clean logit difference is measured once. Then each head is patched in
/// isolation, and importance is `clean_logit_diff - patched_logit_diff`.
pub fn compute_importance(
    model: &HookedPythia,
    prompt: &str,
    target_token_id: u32,
    task_type: &str,
    mean_activations: &Vec<Vec<Tensor>>,
) -> Result<ImportanceResult> {
    let input_ids = model.tokenize(prompt)?;
    let clean_cache = model.forward_with_cache(&input_ids)?;
    let clean_logit_diff = model.logit_diff(&clean_cache.final_logits, target_token_id)?;

    let mut zero_importance = vec![vec![0f32; model.num_heads]; model.num_layers];
    let mut mean_importance = vec![vec![0f32; model.num_heads]; model.num_layers];

    // Zero-ablation: replace one head with zeros and measure the damage.
    for layer in 0..model.num_layers {
        for head in 0..model.num_heads {
            let patch = Patch {
                layer,
                head,
                value: PatchValue::Zero,
            };
            let patched_logits = model.forward_with_patches(&input_ids, &[patch])?;
            let patched_logit_diff = model.logit_diff(&patched_logits, target_token_id)?;
            zero_importance[layer][head] = clean_logit_diff - patched_logit_diff;
        }
    }

    // Mean-ablation: replace one head with its average reference activation.
    for layer in 0..model.num_layers {
        for head in 0..model.num_heads {
            let patch = Patch {
                layer,
                head,
                value: PatchValue::Mean(mean_activations[layer][head].clone()),
            };
            let patched_logits = model.forward_with_patches(&input_ids, &[patch])?;
            let patched_logit_diff = model.logit_diff(&patched_logits, target_token_id)?;
            mean_importance[layer][head] = clean_logit_diff - patched_logit_diff;
        }
    }

    let target_token = model
        .decode_token(target_token_id)
        .unwrap_or_else(|_| target_token_id.to_string());

    Ok(ImportanceResult {
        prompt: prompt.to_string(),
        task_type: task_type.to_string(),
        target_token,
        clean_logit_diff,
        zero_importance,
        mean_importance,
    })
}
