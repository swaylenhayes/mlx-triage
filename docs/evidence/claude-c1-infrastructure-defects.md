# When infrastructure breaks look like broken models

**Runtime and infrastructure defects routinely masquerade as model quality degradation — and the evidence is overwhelming.** Anthropic's September 2025 postmortem disclosed that three separate infrastructure bugs caused weeks of degraded Claude output, with users attributing the problems to intentional model throttling rather than routing misconfigurations and compiler bugs. This pattern repeats across the entire LLM serving ecosystem: from vLLM's batch-size-dependent output divergence to Ollama's silent context truncation to MLX's catastrophic floating-point error accumulation in Metal shaders. The practical implication is that any team serving LLMs must treat infrastructure correctness as a first-class quality concern, with systematic diagnostic methods to distinguish serving defects from model weight issues.

---

## Batching changes the math, and the math changes the output

The most fundamental and pervasive infrastructure-quality interaction is **batch non-invariance** — the phenomenon where processing the same input in different batch compositions produces different outputs due to floating-point non-associativity in GPU kernels.

vLLM's GitHub issue tracker documents this extensively. Issue #5898 showed that Llama3-8b on A100 with temperature=0 produced inconsistent responses when batch size exceeded 1; setting `max_num_seqs=1` restored consistency. Issue #11658 demonstrated that 98 out of 100 identical concurrent requests returned different outputs from each other, while sequential requests were consistent. Most dramatically, issue #17652 showed Qwen3-30B-A3B producing **degenerate output at batch_size=50** — repeating the single word "Supplier" dozens of times — while functioning correctly at smaller batch sizes.

TensorRT-LLM exhibits the same class of bugs. Issue #1823 documented quality degradation when in-flight batching activated with LoRA adapters on L4 GPUs. Issue #1753 revealed that T5 encoder-decoder models produced correct output with single inputs but **junk output with multiple inputs** batched together, caused by incorrect paged KV cache handling for encoder context. Issue #2495 showed Qwen2-VL returning correct results only for the first item in a batch, with all subsequent results empty.

The root cause is now well-characterized by the Thinking Machines Lab's September 2025 analysis. Three categories of non-batch-invariant GPU kernels exist in LLM forward passes: **RMSNorm** (normalization reduction ordering changes), **matrix multiplication** (different tiling/reduction strategies per batch size), and **attention** (split-K strategies in FlashAttention vary with batch size). Their `batch_invariant_ops` library provides drop-in replacements that produce identical per-example results regardless of batch size, at **~34% performance overhead**. SGLang integrated this as a first-class feature, and vLLM v0.9+ added batch-invariant `torch.compile` support. TGI's sliding window attention bug (PR #3112) represents a different but related class: the block-level KV cache management assumed all layers used sliding window attention, silently corrupting outputs for models like Gemma2 and Gemma3 that interleave local and global attention layers.

---

## Stop tokens, templates, and the silent corruption of prompts

EOS handling errors and chat template bugs are the **single most common class of infrastructure defect** mimicking model quality issues, affecting every major framework.

Ollama's silent context truncation (issues #14259, #4967, #7907) is perhaps the most impactful example. With a default context of only **2048 tokens**, Ollama silently drops older messages when conversations exceed the limit, with no user-visible indication. Over 400 GitHub issues relate to this behavior, with users invariably attributing the "forgetting" to poor model quality. The `truncate` parameter defaults to true, and truncation is logged only at debug level.

llama.cpp's chat template fallback creates a parallel problem. When the server cannot parse a model's native template, it falls back to ChatML format with only a warning log. For GLM-4-9b-chat, this produces wrong results; for Llama 3 8b, the model "starts making notes to itself and output garbage/training data." Issue #1360 documented that Llama 3's dual stop tokens (`<|end_of_text|>` and `<|eot_id|>`) were not both recognized, causing generation to continue past turn boundaries with raw role tags leaking into output. Grammar-constrained generation (issue #6277) prevents EOS token selection entirely, producing runaway generation filling output with hundreds of newlines.

Ollama's issue #1977 cataloged **widespread incorrect template definitions across official models**: codellama missing a space before responses that "severely hurts performance" with large code sections, deepseek-llm prepending system messages to every message causing increased Chinese Unicode responses, and multiple models (yi, mpt, qwen) having incorrect ChatML formatting. An entire community repository (`ktsaou/ollama-templates`) was created to address "critical issues with community-generated Ollama models that often have incorrect or incomplete templates."

vLLM's template handling has its own pathologies. Issue #9519 documented **double BOS token injection** where the offline chat API produces `[128000, 128000, 128006, 9125, 128007, ...]` — a BOS token appearing twice that the model never saw during training. Issue #4119 showed Command-R "talking only about the Jinja template" because template content was being used as the prompt instead of being rendered. Issue #25401 revealed that despite documentation recommending `--chat-template` for Mistral models, the code silently ignores any user-specified template. The ramalama project (issue #1855) found that Ollama's Go-template format chat templates were being passed directly to llama.cpp, which silently fell back to ChatML when it couldn't parse them.

RoPE scaling misconfigurations represent a particularly insidious variant. Llama 3.1 introduced a `rope_type: "llama3"` format that older framework versions couldn't parse, and users incorrectly configuring `"type": "dynamic"` as a workaround produced **silent quality degradation at longer context lengths** without any error indication. Phi-4-mini-instruct shipped with a `short_factor` field of length 48 instead of expected 64, breaking loading across vLLM, SGLang, and transformers.

---

## MLX on Apple Silicon has unique and severe quality hazards

Apple's MLX framework introduces several platform-specific failure modes that have no direct analog in CUDA-based stacks, most critically around **numerical precision in Metal shaders** and **KV-cache management**.

### Metal's floating-point non-determinism is catastrophic in float16

A September 2025 investigation found that LLMs on Apple Silicon using MLX are **not reproducible** even with temperature=0 and identical inputs. Matrix multiplication of identical data produces differences up to **~1,449 absolute error** for 2048×4096 matrices, with errors growing with matrix size. The error compounds catastrophically through transformer layers: after 20 operations the difference reaches 1e5, after 60 operations 1e25, and after 80 operations values become NaN. Float16 is described as "The Danger Zone" — catastrophically unstable with large value ranges. Critically, **quantized integer models (Q4_K_M, Q8_0) achieve perfect reproducibility**, proving the non-determinism is specific to floating-point Metal kernel operations.

MLX core developer Awni Hannun filed issue #488 acknowledging that "our reductions are quite naive and can be less accurate particularly in lower precision (mx.float16)" — this issue remains open since January 2024. Issue #2695 documented that `mx.addmm` with float16 on CPU returns **completely wrong results** (`[[1, 1], [1, 1]]` instead of `[[4.80, 11.10], [11.10, 36.31]]`). Issue #1341 showed MLX's `exp` function producing different results from NumPy/PyTorch with differences between CPU and GPU executions. Issue #2122 found that Conv1d operations show MAE ~0.04 vs PyTorch when composed in sequence, despite matching in isolation (MAE ~1e-7).

### Quantization in MLX trails GGUF and AWQ in quality

MLX's native 4-bit quantization uses a straightforward round-to-nearest INT4 scheme without per-layer sensitivity calibration. Issue #730 showed 4-bit quantized OpenELM-3B producing `<unk>` tokens (complete garbage) while full-precision works correctly. A HuggingFace discussion on DeepSeek-V3-0324-4bit revealed that **different MLX quantization code paths produce dramatically different quality** for the same model, with Hannun confirming "Quantization to 4-bit is lossy and in this case we get unlucky in terms of what we are losing."

A November 2025 benchmark of Qwen3 models across quantization formats found:

- **ExLlamaV3-4bpw** beat even BF16 on LiveBench accuracy (60.0 vs 58.2)
- **GGUF formats** (Q4_K_M, UD-Q4_K_XL) are "accuracy monsters" but slower
- **MLX 4-bit DWQ** (Distillation-aware Weight Quantization) is the best Apple option, beating standard MLX-4bit
- **Standard MLX-4bit is inferior** to both DWQ and GGUF K-quants for accuracy

A separate issue compounds this: MLX only natively supports Q4_0, Q4_1, and Q8_0 from GGUF format. **Unsupported quantizations are silently cast to float16** with no error or warning, meaning users loading K-quant GGUF models (Q4_K_M, Q5_K_S) get float16 models using ~4x more memory than expected. Issue #2962 revealed that MLX's nvfp4 implementation uses signed E4M3 scales instead of unsigned UE4M3 (as NVIDIA Blackwell specifies), causing **137x less dynamic range**.

### KV-cache lacks paging and has a critical trim bug

MLX does not implement paged KV cache, unlike vLLM's PagedAttention or MLC-LLM's paged design. For long-context scenarios (32k–128k tokens), this creates memory fragmentation and potential quadratic slowdowns. The `--max-kv-size` parameter controls a rotating cache where "smaller values like 512 will use very little RAM but result in worse quality" — a silent quality knob many users don't understand.

The LM Studio MLX engine (issue #177) exposed a **critical RotatingKVCache trim bug**: when tokens generated exceed `max_kv_size`, the cache rejects trim requests, causing context overflow policies to erase the entire cache instead of selectively trimming. Multi-turn conversations that exceed the context window lose all previous context rather than intelligently managing it. Apple Silicon's unified memory architecture adds another dimension: the GPU cannot use more than ~75% of system RAM, and models consuming too much memory trigger disk swapping. Issue #2254 found MLX allocating 214 GB for a 100 GB tensor (more than 2x expected), unexpectedly triggering swap on systems where models should fit.

---

## Precision format determines whether models work at all

The **Gemma 3 BF16-to-FP16 incident** is the most dramatic documented case of a precision mismatch causing complete model failure that users attributed to model quality.

Gemma 3 (all sizes: 1B, 4B, 12B, 27B) produces **empty output or complete nonsense** when run in FP16 instead of BF16. Researcher Daniel Han's analysis found that after each layer norm, activations reach values of **800,000** — but FP16's maximum representable value is only 65,504. Values overflow to infinity, propagate through the network, and produce NaN. Even with activation clamping workarounds, FP16 showed a **12.6% average quality degradation** across benchmarks (GSM8K dropped 26%, ARC dropped 14.5%, HellaSwag dropped 11.9%). Without the fix, the model produces no usable output. This affected every user on older GPUs (T4, V100, RTX 20xx) that lack native BF16 tensor cores.

BF16 introduces its own precision trap. HuggingFace PR #29285 documented that BF16 RoPE embeddings lose precision at longer context lengths — at 8,192 tokens, BF16 thinks positions [8188, 8189, 8190, 8191] are **all position 8192**, because BF16's 7 mantissa bits cannot distinguish adjacent large integers. This is now mitigated by forcing RoPE computation to FP32. A June 2025 paper studying Qwen-3-235B found that running the same prompt 1,000 times under greedy decoding produced **80 unique outputs** in BF16, while FP32 provided near-perfect determinism.

Cross-runtime comparisons confirm that the same model weights produce systematically different outputs across frameworks. vLLM generated 188,582 tokens from 1,000 requests while TGI generated 180,307 tokens — a **4.6% difference** from identical prompts. A developer documented that Ollama (llama.cpp/Metal/GGUF) vs vLLM (CUDA/Safetensors) showed consistent minor divergence for generative text but "divergence increases substantially for JSON extraction and classification tasks." The SGLang/vLLM Qwen2-VL bug showed **25% quality degradation** (633.57 → 474.29 on MME cognition) compared to the Transformers baseline, purely from the serving framework.

---

## Anthropic's postmortem proves the hypothesis at scale

Anthropic's September 2025 engineering postmortem provides the **definitive industry evidence** that infrastructure defects mimic model quality issues at production scale.

Three simultaneous infrastructure bugs degraded Claude's output quality for weeks. A **context window routing error** (August 5–September 4) sent short-context requests to 1M-token servers; "sticky" routing meant affected users kept getting degraded results, with 16% of Sonnet 4 requests affected at peak. A **TPU token generation corruption** (August 25–September 2) caused a runtime optimization misconfiguration that "occasionally assigned a high probability to tokens that should rarely be produced" — English prompts generated Thai and Chinese characters. An **XLA compiler miscompilation** (August 25–September 16) was triggered when Anthropic rewrote sampling code to fix bf16/fp32 precision mismatches; the compiler's approximate top-k operation excluded the most probable token.

The most revealing aspect: **standard evaluations failed to detect these issues** because "Claude often recovers well from isolated mistakes." The multi-platform deployment across Trainium, GPUs, and TPUs complicated diagnosis, and privacy controls limited engineers' access to problematic user interactions. Users perceived "model quality degradation" when the model weights were entirely unchanged. Anthropic's remediation included more sensitive evaluations differentiating "working" from "broken" implementations, continuous quality evaluations on true production systems rather than staging, and switching to exact top-k operations with higher-precision arithmetic.

---

## Systematic methods exist to isolate the cause

Several practical diagnostic frameworks have emerged for distinguishing infrastructure from model defects.

**Token-DiFR** (November 2025) exploits the fact that LLM inference is ~98% deterministic when sampling seed is fixed. By comparing generated tokens against a trusted reference implementation, it detects KV cache quantization with ~10,000 tokens, 4-bit model quantization with ~1,000 tokens, and incorrect sampling seeds with ~100 tokens. The authors audited multiple Llama-3.1-8B providers and found that Groq and SiliconFlow matched advertised configurations while others showed higher divergence — sometimes from outdated chat templates rather than actual model issues.

The **Thinking Machines Lab's batch invariance testing** provides a direct diagnostic: run the same prompt individually and in a batch; if outputs differ, batching is involved. Setting vLLM's `max_num_seqs=1` to force single-request batches eliminates batch-related issues. Comparing sequential versus concurrent requests isolates continuous batching effects. A batch size sweep (1, 10, 50) reveals quality degradation correlated with batch size.

**Meta's HawkEye toolkit** uses decision-tree-based debugging to systematically narrow from top-line metric anomalies to specific serving models, model snapshots, features, and upstream data pipelines. It correlates degradation with model snapshot rollouts to determine if issues stem from bad snapshots or infrastructure, and uses model graph tracing to explain why models get corrupted. Meta reports a **50% reduction** in time-to-root-cause for ML prediction issues. Databricks explicitly recommends using quality benchmarks like the Mosaic Eval Gauntlet to "evaluate the quality of the inference system, not just the quality of the model" — recognizing that the inference system has quality properties distinct from the model itself.

A practical diagnostic checklist synthesized from the research includes:

- **Precision verification**: Confirm BF16 vs FP16 vs FP32 matches training dtype, especially for TPU-trained models
- **Batch isolation**: Compare single-request vs batched output to detect batch invariance failures
- **Template auditing**: Verify chat template rendering matches the model's expected format, checking for double-BOS, missing stop tokens, and template fallback warnings
- **Cross-runtime comparison**: Test identical prompts across at least two serving frameworks to distinguish framework-specific bugs
- **Progressive deployment**: Shadow deployments, canary releases (1–5% traffic), and automated rollback on metric degradation

## Conclusion

The evidence forms a clear taxonomy of infrastructure defects mimicking model degradation: **numerical precision mismatches** (Gemma 3 FP16 failure, BF16 RoPE precision loss), **batch non-invariance** (98/100 identical requests returning different outputs), **template and tokenization corruption** (Ollama's 400+ context truncation issues), **stop token mishandling** (every framework affected), and **platform-specific kernel bugs** (MLX's catastrophic float16 error accumulation, XLA compiler miscompilation). The most important insight from Anthropic's postmortem is that standard evaluation suites may not catch these failures — models can "recover well from isolated mistakes," meaning infrastructure bugs produce subtle, intermittent degradation that evades benchmarks while frustrating users. The emerging solution architecture combines Token-DiFR for statistical inference verification, batch-invariant kernels for deterministic serving (at ~34% overhead), and continuous production-grade evaluation distinct from staging-environment benchmarks. For MLX specifically, quantized integer models offer perfect reproducibility where floating-point operations do not, DWQ quantization substantially outperforms naive 4-bit, and the absence of paged KV cache remains a significant gap for long-context workloads.