# Quant-Circuit

Quant-Circuit is a Rust CLI for probing how f32 versus f16 execution changes internal attention circuits in Pythia-70M. It asks which attention heads are most important for next-token task performance, which heads drift under reduced precision, and whether restoring selected full-precision activations repairs the quantized run.

## Requirements

- Rust toolchain with Cargo
- Python 3 with `numpy`, `matplotlib`, and `scipy`
- Enough disk space for the Pythia-70M Hugging Face files
- Optional CUDA build support for GPU execution

## Setup

```powershell
cargo build --release
```

For CUDA:

```powershell
cargo build --release --features cuda
```

## Run

```powershell
cargo run --release
```

With explicit inputs:

```powershell
cargo run --release -- --prompts prompts.json --output results/experiment.json --device auto
```

Use `--device cpu` or `--device cuda` to force a device. Use `--max-repair-k 10` to control the repair curve length.

## Figures

```powershell
python scripts/plot.py
```

The script reads `results/experiment.json` and writes:

- `fig1_importance_vs_drift.png`
- `fig2_zero_vs_mean_ablation.png`
- `fig3_task_specificity.png`
- `fig4_circuit_repair.png`
- `fig5_task_degradation.png`

## Experiments

The CLI validates factual-recall and indirect-object-identification prompts against the f32 Pythia-70M run, computes reference mean activations, measures head importance with zero-ablation and mean-ablation, measures f32/f16 head drift, and runs top-k circuit repair by injecting f32 attention-head activations into the f16 forward pass.

The hook is inserted after each attention output projection and before the residual addition. Pythia uses parallel residual blocks, so the MLP reads from the original layer-normalized residual stream rather than from the patched attention output in the same layer.

## Notes

The public `EleutherAI/pythia-70m` checkpoint is distributed as half-precision safetensors. The f32 run loads those weights into f32 tensors, so the comparison isolates dtype/activation arithmetic effects available from the public checkpoint rather than recovering unavailable original f32 training weights.

HackPrinceton writeup: TODO

## License

MIT
