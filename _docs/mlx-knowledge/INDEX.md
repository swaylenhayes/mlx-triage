# MLX Knowledge Corpus — Quick Reference Index

> Auto-synced from ml-explore/mlx and ml-explore/mlx-lm.
> Run `./scripts/sync-mlx-docs.sh --force` to update.
> Search: `grep -r "pattern" _docs/mlx-knowledge/`

---

## Key APIs for mlx-triage

### Model Loading (mlx-lm)
```python
from mlx_lm import load
model, tokenizer = load("path_or_hf_repo")
# Params: tokenizer_config, model_config, adapter_path, lazy, return_config, revision
```
**Source:** `mlx-lm/source/utils.py`

### Text Generation (mlx-lm)
```python
from mlx_lm import generate, stream_generate

# Simple (returns string)
text = generate(model, tokenizer, prompt, temp=0.0, seed=42, max_tokens=256)

# Streaming (yields GenerationResponse)
for response in stream_generate(model, tokenizer, prompt, max_tokens=256):
    response.text       # Next decoded segment
    response.token      # Token ID (int)
    response.logprobs   # Log probabilities (mx.array)
    response.prompt_tps # Prompt tokens/sec
    response.generation_tps  # Generation tokens/sec
    response.finish_reason   # "length", "stop", or None
```
**Source:** `mlx-lm/source/generate.py`

### Batch Generation (mlx-lm)
```python
from mlx_lm import batch_generate
result = batch_generate(model, tokenizer, prompts, max_tokens=128)
result.texts   # List[str]
result.stats   # BatchStats with TPS metrics
```
**Source:** `mlx-lm/source/generate.py`

### Sampling Control (mlx-lm)
```python
from mlx_lm.sample_utils import make_sampler
sampler = make_sampler(temp=0.0)  # temp=0 → argmax (deterministic)
# Params: temp, top_p, min_p, top_k, xtc_probability, xtc_threshold
```
**Source:** `mlx-lm/source/sample_utils.py`

### Perplexity Evaluation (mlx-lm)
```python
from mlx_lm.perplexity import load_data, eval_ppl
data = load_data(tokenizer, "allenai/tulu-3-sft-mixture", num_samples=100, sequence_length=512)
ppl, std_err = eval_ppl(model, data, batch_size=8)
```
**Source:** `mlx-lm/source/perplexity.py`

### Memory Monitoring (mlx.core)
```python
import mlx.core as mx
mx.metal.get_active_memory()   # Current active bytes
mx.metal.get_peak_memory()     # Peak bytes since last reset
mx.metal.set_memory_limit(n)   # Set limit, returns previous
mx.metal.reset_peak_memory()   # Reset peak counter
mx.metal.cache_size()          # Current cache size
mx.metal.set_cache_limit(n)    # Set cache limit
```
**Source:** `mlx-core/python/memory_management.rst`

### Random / Seed Control (mlx.core)
```python
import mlx.core as mx
mx.random.seed(42)             # Set global PRNG seed
key = mx.random.key(42)        # Create explicit PRNG key
```
**Source:** `mlx-core/python/random.rst`

---

## Corpus Structure

| Path | Contents | Format |
|------|----------|--------|
| `mlx-core/python/` | Python API reference (array, nn, metal, memory, random, etc.) | RST |
| `mlx-core/usage/` | Usage guides (lazy evaluation, unified memory, compilation, etc.) | RST |
| `mlx-lm/docs/` | mlx-lm documentation (README, benchmarks, server, LoRA, etc.) | Markdown |
| `mlx-lm/source/` | Key source files with docstrings (generate, utils, sample_utils, perplexity, evaluate) | Python |
