use anyhow::{Context, Result};
use serde::Serialize;
use std::path::Path;

use crate::repair::RepairResult;

#[derive(Clone, Debug, Serialize)]
pub struct ExperimentOutput {
    pub model: String,
    pub num_layers: usize,
    pub num_heads: usize,
    pub prompts: Vec<PromptResult>,
    pub repair: Vec<RepairResult>,
    pub mean_activation_reference_count: usize,
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize)]
pub struct PromptResult {
    pub prompt: String,
    pub target_token: String,
    pub task_type: String,
    pub zero_importance: Vec<Vec<f32>>,
    pub mean_importance: Vec<Vec<f32>>,
    pub drift_f32_f16: Vec<Vec<f32>>,
    pub cosine_sim_f32_f16: Vec<Vec<f32>>,
    pub clean_logit_diff_f32: f32,
    pub clean_logit_diff_f16: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Metadata {
    pub timestamp: String,
    pub device: String,
    pub total_forward_passes: usize,
    pub total_runtime_seconds: f32,
}

pub fn save_results(output: &ExperimentOutput, path: &str) -> Result<()> {
    let path = Path::new(path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(output).context("serialize experiment output")?;
    std::fs::write(path, json).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}
