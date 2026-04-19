from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


RESULTS = Path("results")
INPUT = RESULTS / "experiment.json"


def load_data(path: Path = INPUT) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pretty_task(task: str) -> str:
    return {
        "FactualRecall": "Factual Recall",
        "IOI": "Indirect Object ID",
    }.get(task, task)


def avg_matrix(prompts: list[dict], key: str) -> np.ndarray:
    if not prompts:
        raise ValueError(f"no prompts available for {key}")
    return np.asarray([prompt[key] for prompt in prompts], dtype=float).mean(axis=0)


def has_matrix(prompts: list[dict], key: str) -> bool:
    return bool(prompts) and all(prompt.get(key) is not None for prompt in prompts)


def has_values(prompts: list[dict], key: str) -> bool:
    return bool(prompts) and any(prompt.get(key) is not None for prompt in prompts)


def flatten_heads(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    num_layers, num_heads = matrix.shape
    layers = np.repeat(np.arange(num_layers), num_heads)
    labels = [f"L{layer}H{head}" for layer in range(num_layers) for head in range(num_heads)]
    return matrix.reshape(-1), layers, labels


def corr_text(x: np.ndarray, y: np.ndarray, include_pearson: bool = True) -> str:
    parts = []
    if include_pearson:
        try:
            r, p = pearsonr(x, y)
            parts.append(f"Pearson r = {r:.3f}, p = {p:.2g}")
        except Exception:
            parts.append("Pearson r = n/a")
    try:
        rho, p = spearmanr(x, y)
        parts.append(f"Spearman rho = {rho:.3f}, p = {p:.2g}")
    except Exception:
        parts.append("Spearman rho = n/a")
    return "\n".join(parts)


def annotate_heads(ax: plt.Axes, x: np.ndarray, y: np.ndarray, labels: list[str]) -> None:
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), xytext=(3, 3), textcoords="offset points", fontsize=6)


def scatter_by_layer(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    layers: np.ndarray,
    labels: list[str],
    title: str | None = None,
):
    scatter = ax.scatter(x, y, c=layers, cmap="viridis", s=42, edgecolor="black", linewidth=0.35)
    annotate_heads(ax, x, y, labels)
    if title:
        ax.set_title(title)
    return scatter


def fig1(prompts: list[dict]) -> None:
    importance = avg_matrix(prompts, "zero_importance")
    x, layers, labels = flatten_heads(importance)
    y_f16, _, _ = flatten_heads(avg_matrix(prompts, "drift_f32_f16"))
    has_int8 = has_matrix(prompts, "drift_f32_int8")
    y_int8 = flatten_heads(avg_matrix(prompts, "drift_f32_int8"))[0] if has_int8 else None

    fig, axes = plt.subplots(1, 2 if has_int8 else 1, figsize=(16 if has_int8 else 10, 7), squeeze=False)
    ax = axes[0][0]
    scatter = scatter_by_layer(ax, x, y_f16, layers, labels, "f32 -> f16")
    ax.text(
        0.03,
        0.97,
        corr_text(x, y_f16),
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    ax.set_xlabel("Causal Importance (logit diff drop, zero-ablation)")
    ax.set_ylabel("Quantization Drift (relative L2)")
    if has_int8 and y_int8 is not None:
        ax = axes[0][1]
        scatter = scatter_by_layer(ax, x, y_int8, layers, labels, "f32 -> simulated int8")
        ax.text(
            0.03,
            0.97,
            corr_text(x, y_int8),
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )
        ax.set_xlabel("Causal Importance (logit diff drop, zero-ablation)")
        y_min = min(float(np.min(y_f16)), float(np.min(y_int8)))
        y_max = max(float(np.max(y_f16)), float(np.max(y_int8)))
        for panel in axes[0]:
            panel.set_ylim(y_min, y_max)
    fig.suptitle("Do Important Circuits Break Under Quantization? (Pythia-70M)")
    fig.subplots_adjust(right=0.88, top=0.88, wspace=0.22)
    cbar_ax = fig.add_axes([0.91, 0.18, 0.018, 0.68])
    fig.colorbar(scatter, cax=cbar_ax, label="Layer")
    fig.savefig(RESULTS / "fig1_importance_vs_drift.png", dpi=200)
    plt.close(fig)


def fig2(prompts: list[dict]) -> None:
    zero = avg_matrix(prompts, "zero_importance")
    mean = avg_matrix(prompts, "mean_importance")
    x, layers, labels = flatten_heads(zero)
    y, _, _ = flatten_heads(mean)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = scatter_by_layer(ax, x, y, layers, labels)
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)
    ax.text(
        0.03,
        0.97,
        corr_text(x, y, include_pearson=False),
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    ax.set_xlabel("Zero-Ablation Importance")
    ax.set_ylabel("Mean-Ablation Importance")
    ax.set_title("Cross-Validation: Do Two Methods Agree on What Matters?")
    fig.colorbar(scatter, ax=ax, label="Layer")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig2_zero_vs_mean_ablation.png", dpi=200)
    plt.close(fig)


def task_xy(prompts: list[dict], drift_key: str = "drift_f32_f16") -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    importance = avg_matrix(prompts, "zero_importance")
    drift = avg_matrix(prompts, drift_key)
    x, layers, labels = flatten_heads(importance)
    y, _, _ = flatten_heads(drift)
    return x, y, layers, labels


def fig3(prompts: list[dict]) -> None:
    by_task = {
        "FactualRecall": [p for p in prompts if p["task_type"] == "FactualRecall"],
        "IOI": [p for p in prompts if p["task_type"] == "IOI"],
    }
    drift_specs = [("f32 -> f16", "drift_f32_f16")]
    if has_matrix(prompts, "drift_f32_int8"):
        drift_specs.append(("f32 -> simulated int8", "drift_f32_int8"))
    prepared = {}
    all_x = []
    all_y = []
    for row_idx, (_, drift_key) in enumerate(drift_specs):
        for col_idx, (task, task_prompts) in enumerate(by_task.items()):
            if task_prompts:
                x, y, layers, labels = task_xy(task_prompts, drift_key)
                prepared[(row_idx, col_idx)] = (x, y, layers, labels)
                all_x.append(x)
                all_y.append(y)

    fig, axes = plt.subplots(
        len(drift_specs),
        2,
        figsize=(14, 6 * len(drift_specs)),
        sharey="row",
        squeeze=False,
    )
    scatter = None
    if all_x and all_y:
        x_min, x_max = float(np.min(np.concatenate(all_x))), float(np.max(np.concatenate(all_x)))
    else:
        x_min = -1.0
        x_max = 1.0

    for row_idx, (precision_label, _) in enumerate(drift_specs):
        row_y = []
        for col_idx in range(2):
            if (row_idx, col_idx) in prepared:
                row_y.append(prepared[(row_idx, col_idx)][1])
        if row_y:
            y_min, y_max = float(np.min(np.concatenate(row_y))), float(np.max(np.concatenate(row_y)))
        else:
            y_min = -1.0
            y_max = 1.0
        for col_idx, task in enumerate(["FactualRecall", "IOI"]):
            ax = axes[row_idx][col_idx]
            if (row_idx, col_idx) in prepared:
                x, y, layers, labels = prepared[(row_idx, col_idx)]
                scatter = scatter_by_layer(ax, x, y, layers, labels, f"{pretty_task(task)} ({precision_label})")
                ax.text(
                    0.03,
                    0.97,
                    corr_text(x, y).split("\n")[0],
                    transform=ax.transAxes,
                    va="top",
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
                )
            else:
                ax.text(0.5, 0.5, "No validated prompts", ha="center", va="center")
                ax.set_title(f"{pretty_task(task)} ({precision_label})")
            ax.set_xlabel("Causal Importance (zero-ablation)")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        axes[row_idx][0].set_ylabel("Quantization Drift (relative L2)")

    fig.subplots_adjust(right=0.88, top=0.9, hspace=0.34, wspace=0.18)
    if scatter is not None:
        cbar_ax = fig.add_axes([0.91, 0.14, 0.018, 0.72])
        fig.colorbar(scatter, cax=cbar_ax, label="Layer")
    fig.suptitle("Task-Specific Circuit Fragility: Same Model, Different Circuits?")
    fig.savefig(RESULTS / "fig3_task_specificity.png", dpi=200)
    plt.close(fig)


def average_curve(repairs: list[dict]) -> tuple[np.ndarray, np.ndarray, float, float]:
    max_len = max((len(item["repair_curve"]) for item in repairs), default=0)
    curves = np.full((len(repairs), max_len), np.nan)
    for row, item in enumerate(repairs):
        curve = np.asarray(item["repair_curve"], dtype=float)
        curves[row, : len(curve)] = curve
    avg_curve = np.nanmean(curves, axis=0) if max_len else np.asarray([])
    f32 = float(np.mean([item["baseline_logit_diff_f32"] for item in repairs]))
    q = float(np.mean([item["baseline_logit_diff_quantized"] for item in repairs]))
    y = np.concatenate([[q], avg_curve])
    x = np.arange(len(y))
    return x, y, f32, q


def fig4(repairs: list[dict], repairs_int8: list[dict] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"FactualRecall": "tab:blue", "IOI": "coral"}
    repair_sets = [("f16", repairs, "o", "-")]
    if repairs_int8:
        repair_sets.append(("simulated int8", repairs_int8, "s", ":"))
    for task in ["FactualRecall", "IOI"]:
        ceiling_drawn = False
        for precision_label, repair_data, marker, linestyle in repair_sets:
            task_repairs = [item for item in repair_data if item["task_type"] == task]
            if not task_repairs:
                continue
            x, y, f32_baseline, _ = average_curve(task_repairs)
            ax.plot(
                x,
                y,
                marker=marker,
                linestyle=linestyle,
                color=colors[task],
                label=f"{pretty_task(task)} {precision_label}",
            )
            if not ceiling_drawn:
                ax.axhline(
                    f32_baseline,
                    linestyle="--",
                    color=colors[task],
                    linewidth=1,
                    label=f"{pretty_task(task)} f32 ceiling",
                )
                ceiling_drawn = True
    ax.set_xlabel("Number of Heads Repaired (top-k by importance)")
    ax.set_ylabel("Logit Difference (task performance)")
    ax.set_title("Circuit Repair: How Many Heads Must Be Restored to f32?")
    all_repairs = repairs + (repairs_int8 or [])
    max_x = max((len(item["repair_curve"]) for item in all_repairs), default=10)
    ax.set_xticks(np.arange(max_x + 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig4_circuit_repair.png", dpi=200)
    plt.close(fig)


def fig5(prompts: list[dict]) -> None:
    tasks = ["FactualRecall", "IOI"]
    precisions = ["f32", "f16"]
    keys = {"f32": "clean_logit_diff_f32", "f16": "clean_logit_diff_f16"}
    if has_values(prompts, "clean_logit_diff_int8"):
        precisions.append("simulated int8")
        keys["simulated int8"] = "clean_logit_diff_int8"
    colors = {"FactualRecall": "tab:blue", "IOI": "coral"}
    x = np.arange(len(precisions))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8, 6))
    for offset_idx, task in enumerate(tasks):
        task_prompts = [p for p in prompts if p["task_type"] == task]
        means = []
        stds = []
        for precision in precisions:
            values = np.asarray(
                [p[keys[precision]] for p in task_prompts if p.get(keys[precision]) is not None],
                dtype=float,
            )
            means.append(float(np.mean(values)) if values.size else np.nan)
            stds.append(float(np.std(values)) if values.size else 0.0)
        offset = (offset_idx - 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            color=colors[task],
            capsize=4,
            label=pretty_task(task),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(precisions)
    ax.set_xlabel("Precision Level")
    ax.set_ylabel("Mean Logit Difference (correct - incorrect)")
    ax.set_title("Task Performance Across Precision Levels (Pythia-70M)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig5_task_degradation.png", dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    data = load_data()
    prompts = data["prompts"]
    repairs = data["repair"]
    repairs_int8 = data.get("repair_int8", [])
    if not prompts:
        raise SystemExit(
            "No prompt results found in results/experiment.json. "
            "Rerun the Rust experiment with the updated binary; if you want to bypass "
            "top-1 prompt filtering explicitly, use `cargo run --release -- --use-all-prompts`."
        )
    fig1(prompts)
    fig2(prompts)
    fig3(prompts)
    fig4(repairs, repairs_int8)
    fig5(prompts)
    print("All 5 figures saved to results/")


if __name__ == "__main__":
    main()
