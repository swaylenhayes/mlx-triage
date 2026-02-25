# mlx-triage

MLX Inference Quality Diagnostic Toolkit.

Run tiered diagnostic checks against MLX-served models on Apple Silicon to determine whether quality issues stem from the model weights or the inference infrastructure.

## Install

```bash
uv sync --extra dev
```

## Usage

```bash
mlx-triage check <model_path>
mlx-triage check <model_path> --format json
mlx-triage check <model_path> --tier 1
```
