//! CLI orchestration for the full quantization-circuit experiment.
//!
//! The binary loads prompts and two Pythia-70M instances, validates prompts,
//! runs importance/drift/repair experiments, then writes one JSON file for the
//! Python plotting script.

mod ablation;
mod drift;
mod model;
mod output;
mod repair;

use anyhow::{bail, Context, Result};
use candle::{DType, Device};
use clap::Parser;
use serde::Deserialize;
use std::time::Instant;

use ablation::{compute_importance, compute_mean_activations, ImportanceResult};
use drift::{compute_drift, DriftResult};
use model::HookedPythia;
use output::{save_results, ExperimentOutput, Metadata, PromptResult};
use repair::{compute_repair, RepairResult};

const MODEL_ID: &str = "EleutherAI/pythia-70m";

/// Command-line options for the experiment runner.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(long, default_value = "prompts.json")]
    prompts: String,

    #[arg(long, default_value = "results/experiment.json")]
    output: String,

    #[arg(long, default_value = "auto")]
    device: String,

    #[arg(long, default_value_t = 10)]
    max_repair_k: usize,

    #[arg(long)]
    use_all_prompts: bool,
}

/// Shape of `prompts.json`.
#[derive(Debug, Deserialize)]
struct PromptData {
    factual: Vec<LabeledPrompt>,
    ioi: Vec<LabeledPrompt>,
    reference: Vec<String>,
}

/// A prompt with the expected next-token text.
#[derive(Clone, Debug, Deserialize)]
struct LabeledPrompt {
    prompt: String,
    target: String,
}

/// A prompt that has been converted into the IDs needed for experiments.
#[derive(Clone, Debug)]
struct ValidatedPrompt {
    prompt: String,
    target: String,
    target_token_id: u32,
    task_type: String,
}

/// Runs the complete pipeline: load, validate, score, repair, and serialize.
fn main() -> Result<()> {
    let args = Args::parse();
    let started = Instant::now();
    let mut total_forward_passes = 0usize;

    let prompt_data = load_prompts(&args.prompts)?;
    let device = select_device(&args.device)?;
    eprintln!("[load] using device: {}", device_label(&device));

    let model_f32 = HookedPythia::load(MODEL_ID, &device, DType::F32)
        .with_context(|| format!("load {MODEL_ID} as f32"))?;
    let model_f16 = HookedPythia::load(MODEL_ID, &device, DType::F16)
        .with_context(|| format!("load {MODEL_ID} as f16"))?;
    eprintln!(
        "[load] loaded {MODEL_ID} as {:?} and {:?}",
        model_f32.dtype(),
        model_f16.dtype()
    );

    // Validation filters out prompts where the model does not put the expected
    // token at top-1. Pythia-70M is weak, so the fallback prevents empty output.
    let mut factual = validate_prompts(
        &model_f32,
        &prompt_data.factual,
        "FactualRecall",
        &mut total_forward_passes,
    );
    let mut ioi = validate_prompts(
        &model_f32,
        &prompt_data.ioi,
        "IOI",
        &mut total_forward_passes,
    );

    if args.use_all_prompts {
        eprintln!(
            "[validate] --use-all-prompts set; keeping every prompt regardless of top-1 prediction"
        );
        factual = prompts_without_top1_filter(&model_f32, &prompt_data.factual, "FactualRecall");
        ioi = prompts_without_top1_filter(&model_f32, &prompt_data.ioi, "IOI");
    } else {
        if factual.is_empty() {
            eprintln!(
                "[validate] warning: no factual prompts passed; falling back to all factual prompts so results are not empty"
            );
            factual =
                prompts_without_top1_filter(&model_f32, &prompt_data.factual, "FactualRecall");
        }
        if ioi.is_empty() {
            eprintln!(
                "[validate] warning: no IOI prompts passed; falling back to all IOI prompts so results are not empty"
            );
            ioi = prompts_without_top1_filter(&model_f32, &prompt_data.ioi, "IOI");
        }
    }

    if factual.len() < 3 {
        eprintln!(
            "[validate] warning: only {} factual prompts passed top-1 validation",
            factual.len()
        );
    }
    if ioi.len() < 3 {
        eprintln!(
            "[validate] warning: only {} IOI prompts passed top-1 validation",
            ioi.len()
        );
    }
    let mut validated = Vec::with_capacity(factual.len() + ioi.len());
    validated.extend(factual);
    validated.extend(ioi);

    // Mean activations are computed once and reused for every mean-ablation
    // sweep. They are based on unrelated reference prompts.
    let mean_activations = compute_mean_activations(&model_f32, &prompt_data.reference)
        .context("compute mean activations")?;
    total_forward_passes += prompt_data.reference.len();
    eprintln!(
        "[mean] mean activations computed from {} reference prompts",
        prompt_data.reference.len()
    );

    // Importance is the expensive part: 1 clean pass + 48 zero patches + 48
    // mean patches for each prompt.
    let mut importance_results = Vec::<(ValidatedPrompt, ImportanceResult)>::new();
    for (idx, prompt) in validated.iter().enumerate() {
        let step_started = Instant::now();
        match compute_importance(
            &model_f32,
            &prompt.prompt,
            prompt.target_token_id,
            &prompt.task_type,
            &mean_activations,
        ) {
            Ok(result) => {
                total_forward_passes += 97;
                eprintln!(
                    "[importance] prompt {}/{}: {:?} done ({:.2}s)",
                    idx + 1,
                    validated.len(),
                    short_prompt(&prompt.prompt),
                    step_started.elapsed().as_secs_f32()
                );
                importance_results.push((prompt.clone(), result));
            }
            Err(err) => {
                eprintln!(
                    "[importance] warning: skipped {:?}: {err:#}",
                    short_prompt(&prompt.prompt)
                );
            }
        }
    }

    // Drift and repair are run after importance because repair needs each
    // prompt's zero-ablation ranking.
    let mut prompt_results = Vec::<PromptResult>::new();
    let mut repair_results = Vec::<RepairResult>::new();
    for (idx, (prompt, importance)) in importance_results.iter().enumerate() {
        let drift_started = Instant::now();
        let drift = match compute_drift(&model_f32, &model_f16, &prompt.prompt, &prompt.task_type) {
            Ok(result) => {
                total_forward_passes += 2;
                eprintln!(
                    "[drift] prompt {}/{}: {:?} done ({:.2}s)",
                    idx + 1,
                    importance_results.len(),
                    short_prompt(&prompt.prompt),
                    drift_started.elapsed().as_secs_f32()
                );
                result
            }
            Err(err) => {
                eprintln!(
                    "[drift] warning: skipped {:?}: {err:#}",
                    short_prompt(&prompt.prompt)
                );
                continue;
            }
        };

        let ranking = importance_ranking(&importance.zero_importance);
        let repair_started = Instant::now();
        let repair = match compute_repair(
            &model_f32,
            &model_f16,
            &prompt.prompt,
            prompt.target_token_id,
            &prompt.task_type,
            &ranking,
            args.max_repair_k,
        ) {
            Ok(result) => {
                total_forward_passes += 2 + args.max_repair_k.min(ranking.len());
                eprintln!(
                    "[repair] prompt {}/{}: {:?} done ({:.2}s)",
                    idx + 1,
                    importance_results.len(),
                    short_prompt(&prompt.prompt),
                    repair_started.elapsed().as_secs_f32()
                );
                result
            }
            Err(err) => {
                eprintln!(
                    "[repair] warning: skipped {:?}: {err:#}",
                    short_prompt(&prompt.prompt)
                );
                continue;
            }
        };

        prompt_results.push(to_prompt_result(prompt, importance, &drift, &repair));
        repair_results.push(repair);
    }

    let output = ExperimentOutput {
        model: MODEL_ID.to_string(),
        num_layers: model_f32.num_layers,
        num_heads: model_f32.num_heads,
        prompts: prompt_results,
        repair: repair_results,
        mean_activation_reference_count: prompt_data.reference.len(),
        metadata: Metadata {
            timestamp: chrono::Utc::now().to_rfc3339(),
            device: device_label(&device),
            total_forward_passes,
            total_runtime_seconds: started.elapsed().as_secs_f32(),
        },
    };
    save_results(&output, &args.output)?;
    eprintln!(
        "[done] wrote {} in {:.2}s with {} forward passes",
        args.output,
        started.elapsed().as_secs_f32(),
        total_forward_passes
    );
    Ok(())
}

/// Loads and deserializes the prompt file.
fn load_prompts(path: &str) -> Result<PromptData> {
    let file = std::fs::File::open(path).with_context(|| format!("open prompts file {path}"))?;
    serde_json::from_reader(file).with_context(|| format!("parse prompts file {path}"))
}

/// Converts the user's device choice into a Candle device.
///
/// `auto` prefers CUDA when Candle reports it is available, otherwise CPU.
fn select_device(requested: &str) -> Result<Device> {
    match requested.to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::new_cuda(0)?),
        "auto" => {
            if candle::utils::cuda_is_available() {
                Ok(Device::new_cuda(0)?)
            } else {
                Ok(Device::Cpu)
            }
        }
        other => bail!("unknown --device {other:?}; expected auto, cpu, or cuda"),
    }
}

/// Returns a compact device label for metadata and progress messages.
fn device_label(device: &Device) -> String {
    if matches!(device, Device::Cpu) {
        "cpu".to_string()
    } else {
        format!("{device:?}").to_ascii_lowercase()
    }
}

/// Keeps only prompts whose target token is the f32 model's top-1 prediction.
fn validate_prompts(
    model: &HookedPythia,
    prompts: &[LabeledPrompt],
    task_type: &str,
    total_forward_passes: &mut usize,
) -> Vec<ValidatedPrompt> {
    let mut passed = Vec::new();
    for (idx, item) in prompts.iter().enumerate() {
        let result = validate_one_prompt(model, item, task_type);
        match result {
            Ok((validated, predicted_token)) => {
                *total_forward_passes += 1;
                if predicted_token == validated.target_token_id {
                    eprintln!(
                        "[validate] {task_type} {}/{} pass: {:?} -> {:?}",
                        idx + 1,
                        prompts.len(),
                        short_prompt(&item.prompt),
                        item.target
                    );
                    passed.push(validated);
                } else {
                    let predicted = model
                        .decode_token(predicted_token)
                        .unwrap_or_else(|_| predicted_token.to_string());
                    eprintln!(
                        "[validate] {task_type} {}/{} fail: {:?}; target {:?}, predicted {:?}",
                        idx + 1,
                        prompts.len(),
                        short_prompt(&item.prompt),
                        item.target,
                        predicted
                    );
                }
            }
            Err(err) => {
                eprintln!(
                    "[validate] warning: skipped {:?}: {err:#}",
                    short_prompt(&item.prompt)
                );
            }
        }
    }
    passed
}

/// Converts prompts into experiment inputs without checking top-1 correctness.
///
/// This is useful when strict validation would leave no data, which is common
/// for small models on hand-written factual prompts.
fn prompts_without_top1_filter(
    model: &HookedPythia,
    prompts: &[LabeledPrompt],
    task_type: &str,
) -> Vec<ValidatedPrompt> {
    let mut kept = Vec::new();
    for item in prompts {
        match target_token_id(model, &item.target) {
            Ok(target_token_id) => kept.push(ValidatedPrompt {
                prompt: item.prompt.clone(),
                target: item.target.clone(),
                target_token_id,
                task_type: task_type.to_string(),
            }),
            Err(err) => {
                eprintln!(
                    "[validate] warning: could not keep {:?}: {err:#}",
                    short_prompt(&item.prompt)
                );
            }
        }
    }
    kept
}

/// Validates one prompt and returns both its experiment record and prediction.
fn validate_one_prompt(
    model: &HookedPythia,
    item: &LabeledPrompt,
    task_type: &str,
) -> Result<(ValidatedPrompt, u32)> {
    let target_token_id = target_token_id(model, &item.target)?;
    let input_ids = model.tokenize(&item.prompt)?;
    let cache = model.forward_with_cache(&input_ids)?;
    let predicted_token = model.argmax_token(&cache.final_logits)?;
    Ok((
        ValidatedPrompt {
            prompt: item.prompt.clone(),
            target: item.target.clone(),
            target_token_id,
            task_type: task_type.to_string(),
        },
        predicted_token,
    ))
}

/// Converts a target string like `" Paris"` into the token ID being scored.
///
/// If the tokenizer splits a target into multiple tokens, the experiment uses
/// the first token because the model predicts one next token at a time.
fn target_token_id(model: &HookedPythia, target: &str) -> Result<u32> {
    let ids = model.encode_ids_with_special_tokens(target, false)?;
    if ids.is_empty() {
        bail!("target {target:?} tokenized to zero tokens");
    }
    if ids.len() > 1 {
        eprintln!(
            "[validate] warning: target {:?} tokenized to {:?}; using first token",
            target, ids
        );
    }
    Ok(ids[0])
}

/// Sorts all heads by descending zero-ablation importance for one prompt.
fn importance_ranking(zero_importance: &[Vec<f32>]) -> Vec<(usize, usize)> {
    let mut scored = Vec::new();
    for (layer, heads) in zero_importance.iter().enumerate() {
        for (head, score) in heads.iter().enumerate() {
            scored.push((layer, head, *score));
        }
    }
    scored.sort_by(|a, b| b.2.total_cmp(&a.2));
    scored
        .into_iter()
        .map(|(layer, head, _)| (layer, head))
        .collect()
}

/// Combines the separate experiment outputs into the JSON prompt schema.
fn to_prompt_result(
    prompt: &ValidatedPrompt,
    importance: &ImportanceResult,
    drift: &DriftResult,
    repair: &RepairResult,
) -> PromptResult {
    PromptResult {
        prompt: prompt.prompt.clone(),
        target_token: prompt.target.clone(),
        task_type: prompt.task_type.clone(),
        zero_importance: importance.zero_importance.clone(),
        mean_importance: importance.mean_importance.clone(),
        drift_f32_f16: drift.relative_drift.clone(),
        cosine_sim_f32_f16: drift.cosine_sim.clone(),
        clean_logit_diff_f32: importance.clean_logit_diff,
        clean_logit_diff_f16: repair.baseline_logit_diff_quantized,
    }
}

/// Truncates long prompt text for readable stderr progress logs.
fn short_prompt(prompt: &str) -> String {
    const LIMIT: usize = 72;
    if prompt.chars().count() <= LIMIT {
        prompt.to_string()
    } else {
        let mut shortened: String = prompt.chars().take(LIMIT - 3).collect();
        shortened.push_str("...");
        shortened
    }
}
