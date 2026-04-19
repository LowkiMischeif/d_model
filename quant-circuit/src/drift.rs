//! Quantization drift measurement.
//!
//! This module compares the cached attention-head activations from the f32 and
//! f16 model runs on the same prompt. The metrics are computed on the final
//! sequence position because that position feeds the next-token logits.

use anyhow::Result;
use candle::Tensor;
use serde::Serialize;

use crate::model::{tensor_to_vec_f32, HookedPythia};

/// Per-prompt drift metrics for every attention head.
///
/// Each matrix has shape `[num_layers][num_heads]`.
#[derive(Clone, Debug, Serialize)]
pub struct DriftResult {
    pub prompt: String,
    pub task_type: String,
    pub precision_pair: (String, String),
    pub drift: Vec<Vec<f32>>,
    pub relative_drift: Vec<Vec<f32>>,
    pub cosine_sim: Vec<Vec<f32>>,
}

/// Runs the f32 and f16 models on one prompt and compares their head outputs.
///
/// `drift` is raw L2 distance, `relative_drift` normalizes by the f32
/// activation norm, and `cosine_sim` measures directional agreement.
pub fn compute_drift(
    model_f32: &HookedPythia,
    model_f16: &HookedPythia,
    prompt: &str,
    task_type: &str,
) -> Result<DriftResult> {
    let input_ids = model_f32.tokenize(prompt)?;
    let cache_f32 = model_f32.forward_with_cache(&input_ids)?;
    let cache_f16 = model_f16.forward_with_cache(&input_ids)?;

    // Allocate one metric grid per comparison. The dimensions match Pythia-70M:
    // 6 layers x 8 attention heads.
    let mut drift = vec![vec![0f32; model_f32.num_heads]; model_f32.num_layers];
    let mut relative_drift = vec![vec![0f32; model_f32.num_heads]; model_f32.num_layers];
    let mut cosine_sim = vec![vec![0f32; model_f32.num_heads]; model_f32.num_layers];

    for layer in 0..model_f32.num_layers {
        for head in 0..model_f32.num_heads {
            // Compare only the last-position head vector, since that is the
            // position whose hidden state is converted into final logits.
            let f32_values = last_position_vec(&cache_f32.attn_out[layer][head])?;
            let f16_values = last_position_vec(&cache_f16.attn_out[layer][head])?;
            let mut sq_diff = 0f32;
            let mut sq_f32 = 0f32;
            let mut sq_f16 = 0f32;
            let mut dot = 0f32;

            for (a, b) in f32_values.iter().zip(f16_values.iter()) {
                let delta = a - b;
                sq_diff += delta * delta;
                sq_f32 += a * a;
                sq_f16 += b * b;
                dot += a * b;
            }

            let l2 = sq_diff.sqrt();
            let norm_f32 = sq_f32.sqrt();
            let norm_f16 = sq_f16.sqrt();
            drift[layer][head] = l2;
            relative_drift[layer][head] = if norm_f32 > 0.0 { l2 / norm_f32 } else { 0.0 };
            cosine_sim[layer][head] = if norm_f32 > 0.0 && norm_f16 > 0.0 {
                dot / (norm_f32 * norm_f16)
            } else {
                0.0
            };
        }
    }

    Ok(DriftResult {
        prompt: prompt.to_string(),
        task_type: task_type.to_string(),
        precision_pair: ("f32".to_string(), "f16".to_string()),
        drift,
        relative_drift,
        cosine_sim,
    })
}

/// Extracts the final sequence-position vector from a cached head tensor.
///
/// Cached head tensors have shape `[seq_len, head_dim]`; this returns the
/// final `[head_dim]` vector as host-side `f32` values.
fn last_position_vec(tensor: &Tensor) -> Result<Vec<f32>> {
    let seq_len = tensor.dim(0)?;
    let last = tensor.narrow(0, seq_len - 1, 1)?.squeeze(0)?;
    tensor_to_vec_f32(&last)
}
