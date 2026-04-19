//! Quantization drift measurement.
//!
//! This module compares cached attention-head activations from any two
//! `HookedPythia` precision modes on the same prompt. The metrics are computed
//! on the final sequence position because that position feeds the next-token
//! logits.

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

/// Runs two model precision modes on one prompt and compares their head outputs.
///
/// `drift` is raw L2 distance, `relative_drift` normalizes by the reference
/// model's activation norm, and `cosine_sim` measures directional agreement.
pub fn compute_drift(
    reference_model: &HookedPythia,
    comparison_model: &HookedPythia,
    prompt: &str,
    task_type: &str,
) -> Result<DriftResult> {
    let input_ids = reference_model.tokenize(prompt)?;
    let reference_cache = reference_model.forward_with_cache(&input_ids)?;
    let comparison_cache = comparison_model.forward_with_cache(&input_ids)?;

    // Allocate one metric grid per comparison. The dimensions match Pythia-70M:
    // 6 layers x 8 attention heads.
    let mut drift = vec![vec![0f32; reference_model.num_heads]; reference_model.num_layers];
    let mut relative_drift =
        vec![vec![0f32; reference_model.num_heads]; reference_model.num_layers];
    let mut cosine_sim = vec![vec![0f32; reference_model.num_heads]; reference_model.num_layers];

    for layer in 0..reference_model.num_layers {
        for head in 0..reference_model.num_heads {
            // Compare only the last-position head vector, since that is the
            // position whose hidden state is converted into final logits.
            let reference_values = last_position_vec(&reference_cache.attn_out[layer][head])?;
            let comparison_values = last_position_vec(&comparison_cache.attn_out[layer][head])?;
            let mut sq_diff = 0f32;
            let mut sq_reference = 0f32;
            let mut sq_comparison = 0f32;
            let mut dot = 0f32;

            for (a, b) in reference_values.iter().zip(comparison_values.iter()) {
                let delta = a - b;
                sq_diff += delta * delta;
                sq_reference += a * a;
                sq_comparison += b * b;
                dot += a * b;
            }

            let l2 = sq_diff.sqrt();
            let norm_reference = sq_reference.sqrt();
            let norm_comparison = sq_comparison.sqrt();
            drift[layer][head] = l2;
            relative_drift[layer][head] = if norm_reference > 0.0 {
                l2 / norm_reference
            } else {
                0.0
            };
            cosine_sim[layer][head] = if norm_reference > 0.0 && norm_comparison > 0.0 {
                dot / (norm_reference * norm_comparison)
            } else {
                0.0
            };
        }
    }

    Ok(DriftResult {
        prompt: prompt.to_string(),
        task_type: task_type.to_string(),
        precision_pair: (
            reference_model.precision_label().to_string(),
            comparison_model.precision_label().to_string(),
        ),
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
