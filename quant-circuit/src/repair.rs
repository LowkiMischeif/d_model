use anyhow::Result;
use serde::Serialize;

use crate::model::{HookedPythia, Patch, PatchValue};

#[derive(Clone, Debug, Serialize)]
pub struct RepairResult {
    pub prompt: String,
    pub task_type: String,
    pub baseline_logit_diff_f32: f32,
    pub baseline_logit_diff_quantized: f32,
    pub repair_curve: Vec<f32>,
    pub repaired_heads: Vec<(usize, usize)>,
}

pub fn compute_repair(
    model_f32: &HookedPythia,
    model_f16: &HookedPythia,
    prompt: &str,
    target_token_id: u32,
    task_type: &str,
    importance_ranking: &[(usize, usize)],
    max_repair_k: usize,
) -> Result<RepairResult> {
    let input_ids = model_f32.tokenize(prompt)?;
    let f32_cache = model_f32.forward_with_cache(&input_ids)?;
    let f16_cache = model_f16.forward_with_cache(&input_ids)?;

    let baseline_logit_diff_f32 = model_f32.logit_diff(&f32_cache.final_logits, target_token_id)?;
    let baseline_logit_diff_quantized =
        model_f16.logit_diff(&f16_cache.final_logits, target_token_id)?;

    let limit = max_repair_k.min(importance_ranking.len());
    let mut repair_curve = Vec::with_capacity(limit);
    for k in 1..=limit {
        let mut patches = Vec::with_capacity(k);
        for &(layer, head) in importance_ranking.iter().take(k) {
            patches.push(Patch {
                layer,
                head,
                value: PatchValue::Replace(f32_cache.attn_out[layer][head].clone()),
            });
        }
        let repaired_logits = model_f16.forward_with_patches(&input_ids, &patches)?;
        repair_curve.push(model_f16.logit_diff(&repaired_logits, target_token_id)?);
    }

    Ok(RepairResult {
        prompt: prompt.to_string(),
        task_type: task_type.to_string(),
        baseline_logit_diff_f32,
        baseline_logit_diff_quantized,
        repair_curve,
        repaired_heads: importance_ranking.iter().take(limit).copied().collect(),
    })
}
