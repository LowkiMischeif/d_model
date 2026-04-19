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
use model::{HookedPythia, Precision};
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

    /// Skip the simulated INT8 comparison path.
    #[arg(long)]
    disable_int8: bool,

    /// Also run f32-activation repair on the simulated INT8 model.
    #[arg(long)]
    repair_int8: bool,

    /// Benchmark clean forward-pass speed and exit without running experiments.
    #[arg(long)]
    benchmark_speed: bool,

    /// Timed iterations for --benchmark-speed.
    #[arg(long, default_value_t = 30)]
    benchmark_iters: usize,

    /// Untimed warmup iterations for --benchmark-speed.
    #[arg(long, default_value_t = 3)]
    benchmark_warmup: usize,

    /// Prompt used by --benchmark-speed.
    #[arg(long, default_value = "The Eiffel Tower is located in")]
    benchmark_prompt: String,
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
    let model_int8 = if args.disable_int8 {
        eprintln!("[load] simulated INT8 disabled by --disable-int8");
        None
    } else {
        eprintln!(
            "[load] native INT8 safetensors GPT-NeoX loading is not available in this Candle path; using simulated dynamic activation INT8"
        );
        match HookedPythia::load_precision(MODEL_ID, &device, Precision::Int8) {
            Ok(model) => Some(model),
            Err(err) => {
                eprintln!("[load] warning: could not initialize simulated INT8 model: {err:#}");
                None
            }
        }
    };
    eprintln!(
        "[load] loaded {MODEL_ID} as {:?} and {:?}",
        model_f32.dtype(),
        model_f16.dtype()
    );
    if let Some(model_int8) = &model_int8 {
        eprintln!("[load] INT8 mode: {}", model_int8.precision_description());
    }

    if args.benchmark_speed {
        benchmark_speed(
            &model_f32,
            &model_f16,
            model_int8.as_ref(),
            &args.benchmark_prompt,
            args.benchmark_warmup,
            args.benchmark_iters,
        )?;
        return Ok(());
    }

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
    let mut repair_int8_results = Vec::<RepairResult>::new();
    for (idx, (prompt, importance)) in importance_results.iter().enumerate() {
        let drift_started = Instant::now();
        let drift_f16 =
            match compute_drift(&model_f32, &model_f16, &prompt.prompt, &prompt.task_type) {
                Ok(result) => {
                    total_forward_passes += 2;
                    eprintln!(
                        "[drift:f16] prompt {}/{}: {:?} done ({:.2}s)",
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

        let drift_int8 = if let Some(model_int8) = &model_int8 {
            let drift_started = Instant::now();
            match compute_drift(&model_f32, model_int8, &prompt.prompt, &prompt.task_type) {
                Ok(result) => {
                    total_forward_passes += 2;
                    eprintln!(
                        "[drift:int8] prompt {}/{}: {:?} done ({:.2}s)",
                        idx + 1,
                        importance_results.len(),
                        short_prompt(&prompt.prompt),
                        drift_started.elapsed().as_secs_f32()
                    );
                    Some(result)
                }
                Err(err) => {
                    eprintln!(
                        "[drift:int8] warning: skipped {:?}: {err:#}",
                        short_prompt(&prompt.prompt)
                    );
                    None
                }
            }
        } else {
            None
        };

        let clean_logit_diff_int8 = if let Some(model_int8) = &model_int8 {
            match compute_clean_logit_diff(model_int8, &prompt.prompt, prompt.target_token_id) {
                Ok(value) => {
                    total_forward_passes += 1;
                    Some(value)
                }
                Err(err) => {
                    eprintln!(
                        "[clean:int8] warning: skipped {:?}: {err:#}",
                        short_prompt(&prompt.prompt)
                    );
                    None
                }
            }
        } else {
            None
        };

        let ranking = importance_ranking(&importance.zero_importance);
        let repair_started = Instant::now();
        let repair_f16 = match compute_repair(
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
                    "[repair:f16] prompt {}/{}: {:?} done ({:.2}s)",
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

        let repair_int8 = if args.repair_int8 {
            if let Some(model_int8) = &model_int8 {
                let repair_started = Instant::now();
                match compute_repair(
                    &model_f32,
                    model_int8,
                    &prompt.prompt,
                    prompt.target_token_id,
                    &prompt.task_type,
                    &ranking,
                    args.max_repair_k,
                ) {
                    Ok(result) => {
                        total_forward_passes += 2 + args.max_repair_k.min(ranking.len());
                        eprintln!(
                            "[repair:int8] prompt {}/{}: {:?} done ({:.2}s)",
                            idx + 1,
                            importance_results.len(),
                            short_prompt(&prompt.prompt),
                            repair_started.elapsed().as_secs_f32()
                        );
                        Some(result)
                    }
                    Err(err) => {
                        eprintln!(
                            "[repair:int8] warning: skipped {:?}: {err:#}",
                            short_prompt(&prompt.prompt)
                        );
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        prompt_results.push(to_prompt_result(
            prompt,
            importance,
            &drift_f16,
            drift_int8.as_ref(),
            &repair_f16,
            clean_logit_diff_int8,
        ));
        repair_results.push(repair_f16);
        if let Some(repair_int8) = repair_int8 {
            repair_int8_results.push(repair_int8);
        }
    }

    let output = ExperimentOutput {
        model: MODEL_ID.to_string(),
        num_layers: model_f32.num_layers,
        num_heads: model_f32.num_heads,
        prompts: prompt_results,
        repair: repair_results,
        repair_int8: repair_int8_results,
        mean_activation_reference_count: prompt_data.reference.len(),
        metadata: Metadata {
            timestamp: chrono::Utc::now().to_rfc3339(),
            device: device_label(&device),
            int8_mode: model_int8
                .as_ref()
                .map(|model| model.precision_description().to_string()),
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

/// Runs one clean pass and computes the target-token logit difference.
fn compute_clean_logit_diff(
    model: &HookedPythia,
    prompt: &str,
    target_token_id: u32,
) -> Result<f32> {
    let input_ids = model.tokenize(prompt)?;
    let cache = model.forward_with_cache(&input_ids)?;
    model.logit_diff(&cache.final_logits, target_token_id)
}

/// Measures clean logits-only forward speed for available precision modes.
///
/// This intentionally calls `forward_with_patches(..., &[])` instead of
/// `forward_with_cache` so the timing measures normal next-token inference and
/// does not include the extra cost of saving all attention-head activations.
fn benchmark_speed(
    model_f32: &HookedPythia,
    model_f16: &HookedPythia,
    model_int8: Option<&HookedPythia>,
    prompt: &str,
    warmup_iters: usize,
    timed_iters: usize,
) -> Result<()> {
    if timed_iters == 0 {
        bail!("--benchmark-iters must be greater than zero");
    }

    println!("Speed benchmark prompt: {prompt:?}");
    println!("Warmup iterations: {warmup_iters}");
    println!("Timed iterations: {timed_iters}");
    println!();
    println!(
        "{:<18} {:>12} {:>12} {:>12}",
        "precision", "total_s", "avg_ms", "tok/s"
    );

    benchmark_one_model("f32", model_f32, prompt, warmup_iters, timed_iters)?;
    benchmark_one_model("f16", model_f16, prompt, warmup_iters, timed_iters)?;
    if let Some(model_int8) = model_int8 {
        benchmark_one_model(
            "simulated int8",
            model_int8,
            prompt,
            warmup_iters,
            timed_iters,
        )?;
    } else {
        println!("{:<18} unavailable", "simulated int8");
    }
    Ok(())
}

/// Runs warmup and timed forward passes for one model.
fn benchmark_one_model(
    label: &str,
    model: &HookedPythia,
    prompt: &str,
    warmup_iters: usize,
    timed_iters: usize,
) -> Result<()> {
    let input_ids = model.tokenize(prompt)?;
    let token_count = input_ids.dim(1)?;

    for _ in 0..warmup_iters {
        let _ = model.forward_with_patches(&input_ids, &[])?;
    }

    let started = Instant::now();
    for _ in 0..timed_iters {
        let _ = model.forward_with_patches(&input_ids, &[])?;
    }
    let elapsed = started.elapsed().as_secs_f64();
    let avg_ms = elapsed * 1000.0 / timed_iters as f64;
    let tokens_per_second = token_count as f64 * timed_iters as f64 / elapsed;

    println!(
        "{:<18} {:>12.4} {:>12.3} {:>12.1}",
        label, elapsed, avg_ms, tokens_per_second
    );
    Ok(())
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
    drift_f16: &DriftResult,
    drift_int8: Option<&DriftResult>,
    repair_f16: &RepairResult,
    clean_logit_diff_int8: Option<f32>,
) -> PromptResult {
    PromptResult {
        prompt: prompt.prompt.clone(),
        target_token: prompt.target.clone(),
        task_type: prompt.task_type.clone(),
        zero_importance: importance.zero_importance.clone(),
        mean_importance: importance.mean_importance.clone(),
        drift_f32_f16: drift_f16.relative_drift.clone(),
        cosine_sim_f32_f16: drift_f16.cosine_sim.clone(),
        drift_f32_int8: drift_int8.map(|drift| drift.relative_drift.clone()),
        cosine_sim_f32_int8: drift_int8.map(|drift| drift.cosine_sim.clone()),
        clean_logit_diff_f32: importance.clean_logit_diff,
        clean_logit_diff_f16: repair_f16.baseline_logit_diff_quantized,
        clean_logit_diff_int8,
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
