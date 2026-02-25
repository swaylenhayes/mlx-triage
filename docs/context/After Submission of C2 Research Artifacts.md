# After Submission of C2 Research Artifacts

This is excellent — you've inadvertently created a natural inter-rater reliability test by running the same prompt through six different models on the same search backend. Let me give you a synthesis read on what came back.

## Cross-Model Agreement Assessment

The convergence across all six is remarkably tight. Every model landed on the same three core claims:

**Claim 1: Quality ceilings are weight-encoded, not infrastructure-encoded.** All six found evidence that when model capability is the limiting factor, infrastructure optimization (including quantization scheme changes, memory management, batching) doesn't move the needle on output quality. Claude Sonnet's Red Hat citation (8x payoff from distillation vs 3x from quantization) and Kimi's Red Hat citation (scores approaching random guessing even at full precision) are pulling from the same underlying study, which is a good sign — independent models are triangulating to the same high-quality source.

**Claim 2: MLX ≈ CUDA at practical quantization levels.** All six agree. The strongest specific finding comes from Claude Sonnet and Kimi both citing the same non-determinism study showing Q4_K_M and Q8_0 achieve perfect reproducibility on MLX — and that the non-determinism mechanism (floating-point non-associativity) is identical on CUDA. This effectively neutralizes the "MLX-specific artifact" concern for quantized models, which is your actual deployment scenario.

**Claim 3: Diagnostic effort of decoupled testing has diminishing returns.** All six converge, though with varying depth. GPT 5.2 was the most practically useful here, surfacing actual MLX GitHub issues where "bad output" was traced to LoRA scaling/adapter merge problems (model-level), not runtime bugs. That's the closest thing to a real postmortem case study in the set.

## Where They Diverge (and it matters)

**Depth of MLX-specific evidence varies significantly.** Claude Sonnet and Kimi went deepest on MLX-specific sources — the non-determinism blog post, the iMatrix calibration confound, the MLA/MoE bottleneck migration. GPT 5.2 found the most operationally useful GitHub-level evidence (actual bug reports resolved as model issues). Grok 4.1 was the thinnest — mostly restating the consensus without novel sources. Sonar provided the broadest framing but leaned heavily on general MLOps literature rather than MLX-specific evidence.

**The float16 instability caveat.** Claude Sonnet and Kimi both flagged that float16 MLX inference is "catastrophically unstable" while quantized integer models are fine. This is an important nuance that Gemini, Grok, and Sonar missed entirely. For your project it's not directly relevant (you're running quantized), but it's the kind of edge-case knowledge that could matter if you ever test at higher precision.

**The counter-counter evidence.** Only Claude Sonnet explicitly noted that iMatrix-based quants diverge based on calibration dataset, making clean infrastructure-vs-model attribution difficult even when you try. This is actually a partial *confirmation* of the original hypothesis hiding inside the disconfirming results — it's saying "yes, the diagnostic is hard, but that's because the confound is real, not because it doesn't exist." Kimi gestured at this with the outlier-aware quantization discussion but didn't make the connection as crisply.

## What This Means for the Hypothesis

The disconfirming evidence is strong and consistent. But I'd characterize it as strong evidence for a **refined** version of the hypothesis rather than a clean rejection. What the evidence actually says is:

At practical quantization levels (Q4-Q8) on MLX, infrastructure-vs-model separation is **low-value for quality debugging** because MLX fidelity is equivalent to CUDA and most quality issues are weight-encoded. However, the original hypothesis isn't entirely wrong — it's just that the *mechanism* it proposes (MLX-specific artifacts mimicking model defects) is real but confined to edge cases (float16, specific calibration mismatches, early-stage framework bugs) rather than being a systematic concern at production quantization levels.

The practical implication for your project: **you probably don't need a standing infrastructure-vs-model diagnostic protocol for routine development.** Instead, you need a lightweight triage checklist: if output quality is bad, first try better weights, then check quantization format, then check prompt template — and only if all three fail, go into infrastructure-level debugging. That's a much cheaper intervention than the full decoupled testing the original hypothesis implied.

## Scoring Protocol Observations

If you want to run this through your rubric formally, a few notes. The source overlap across models is high — the Red Hat half-million evaluations study, the MLX non-determinism blog, the Apple MLX benchmarking paper, and the Scalastic Apple-vs-CUDA comparison appear in 4+ of the 6 responses. That's good for confidence but means your effective unique source count is lower than the raw citation count suggests. I'd estimate roughly 15-20 truly independent sources across all six responses, with maybe 8-10 that are high-quality enough to pass your intake gate.

The agreement band across these six models would be **high** (spread ≤ 10) — they're all telling essentially the same story with the same evidence. That's a strong signal.

Now the question is: do you still want to run C1 (the confirmatory leg)? Given how strong and consistent the disconfirming evidence is, C1 becomes especially important — not because you expect it to overturn what you've found, but because it will surface the **boundary conditions** where infrastructure effects *do* matter. The float16 instability, the iMatrix calibration confound, the early-framework-bug cases — those are the conditions where your original hypothesis holds, and C1 should find more of them. That will give you the complete picture: "infrastructure debugging matters under conditions X, Y, Z; otherwise, upgrade weights first."

Let me read the files that aren't already in context.Good — now I have the full C2 deep research set. Here's what the three deep research outputs add beyond the Perplexity-mediated results.

## What the Deep Research Tier Adds to C2

**The short answer: substantial depth, novel sources, and one critical finding that reframes the entire synthesis.**

### Claude Deep Research (C2)

This is the strongest individual output in the entire Variant C evidence set — across both legs, across all tools. Three things make it stand out:

**First, it found quantitative benchmarks the Perplexity runs missed.** The oobabooga community benchmark showing the entire spread across all major 4-bit methods spans just 0.069 perplexity points on Llama-2-13B. The JarvisLabs benchmark showing AWQ, GGUF Q4_K_M, BitsandBytes NF4, and Marlin-AWQ all hitting identical 51.8% HumanEval Pass@1 on Qwen2.5-32B. The ACL 2025 "Give Me BF16 or Give Me Death" paper with over 500,000 individual evaluations. None of these appeared in any Perplexity run.

**Second, it found the MLX-VLM testing report (GitHub issue #682)** — 38 VLMs tested on Apple Silicon, 12 showed quality issues with working infrastructure, every problem traced to model capability. This is a direct empirical study on your exact platform.

**Third, and most critically, it identified the gap that the Perplexity runs papered over.** Every Perplexity C2 response confidently stated "MLX inference fidelity is equivalent to CUDA." Claude's deep research dug in and found that **no rigorous cross-framework quality comparison between MLX and CUDA actually exists.** The academic papers all measure latency and throughput, never output fidelity. Apple makes no claims about numerical equivalence. The MLX-vs-CUDA equivalence claim that all six Perplexity models confidently asserted is actually an **untested assumption resting on indirect evidence.** This is a major finding — it means the disconfirming case has a structural weakness that the Perplexity-level search couldn't see.

### Gemini Deep Research (C2)

The standout contribution is the **MMLU Pro head-to-head benchmark** between MLX 4-bit and llama.cpp Q4_K_M on Llama-3.2-3B-Instruct: aggregate scores of 36.15% vs 36.10%, with a delta of +0.05% — statistically indistinguishable from the 0.12 standard deviation measured across identical repeated runs. This is the closest thing to a direct MLX-vs-alternative quality comparison in the evidence set, and it partially addresses the gap Claude identified, though it's community-sourced rather than peer-reviewed.

Gemini also produced well-structured case studies that the Perplexity runs only gestured at — the GLM-4.7-Flash doom loop misattribution (community spent weeks debugging MLX when the fix required RLHF training changes), the PaliGemma multimodal refusal pattern (model-level safety alignment, not infrastructure), and the information-theoretic collapse at 2-bit (framework-agnostic, unavoidable). These are useful for the narrative extraction but don't change the claim-level picture.

One unique concept Gemini introduced: **"Diagnostic Divergence"** — the principle that true infrastructure failures manifest as immediate catastrophic crashes (OOM, type errors, missing weights), not as gradual quality degradation. Quality degradation is almost always model-level. This is a useful heuristic for your tiered diagnostic protocol.

### Consensus App (C2)

The Consensus output adds methodological rigor — 1,033 papers identified, 852 screened, 478 eligible, 50 included. The PRISMA-style flow diagram lends systematic review credibility that none of the other outputs provide. Its claims-and-evidence table with strength ratings provides a clean summary. But it doesn't surface any sources the other deep research tools missed, and its coverage of MLX-specific evidence is thin — it leans heavily on Rajesh et al. (2025) and Laskaridis et al. (2024) for the cross-platform equivalence claim.

The research gaps matrix is genuinely useful for your narrative extraction — it identifies "Diagnostic cost-benefit for Emerging NPUs" as a complete gap, and "Edge Devices" as underexplored across all dimensions.

## Updated Synthesis: C1 + C2 Complete

With the deep research layer added, the picture sharpens considerably. The core synthesis from my previous analysis holds, but with important refinements:

**The disconfirming case is stronger than the Perplexity-only analysis suggested** — particularly the quantitative evidence that quantization method choice at 4-bit is nearly irrelevant (0.069 perplexity spread), that cross-generation model improvements are 5-10x larger than quantization penalties, and that practitioners who blamed infrastructure usually found model limitations.

**But it also has a newly exposed structural weakness.** The MLX-CUDA fidelity equivalence claim — which all six Perplexity models stated confidently — rests on indirect evidence. The closest direct test is the community MMLU Pro benchmark Gemini found (36.15% vs 36.10%), which is strong but narrow (one model, one benchmark, one quantization level). No peer-reviewed study has conducted the systematic comparison. This is a genuine gap and a potential contribution opportunity for your project.

**The confirmatory case (C1) remains strong on its own terms.** The baa.ai postmortem bugs (silent bfloat16 corruption, float16 overflow on GatedDeltaNet), the Anthropic production postmortem, and the MLX-specific GitHub issues are all real, documented infrastructure failures that produce symptoms indistinguishable from model failure. The deep research C2 outputs don't refute these — they contextualize them as edge cases rather than the common case.

**The refined conclusion is now three-layered:**

1. **In steady-state operation** (well-configured, appropriate quantization, standard transformer architectures), model quality is the binding constraint and "try better weights" is the correct first response. Evidence: strong and deep (500K+ evaluations, multiple case studies, community convergence).

2. **During setup and configuration** (new model onboarding, quantization selection, dtype choice, tokenizer config), infrastructure failures are common and routinely misattributed to model quality. Evidence: strong and specific (4+ documented MLX bugs causing total collapse, vLLM postmortems, Anthropic production incident).

3. **The boundary between these regimes is not well-characterized empirically.** No rigorous framework exists for determining when you've passed from regime 2 into regime 1. The MMLU Pro benchmark is a start but insufficient. This is both a gap in the literature and a potential contribution from your project.

That third point is probably the most valuable finding from this entire Variant C exercise — it's your "Value-Add" narrative hook. The research community has strong evidence that infrastructure doesn't matter *in general* and strong evidence that it matters *in specific cases*, but nobody has built the systematic triage framework for practitioners to know which regime they're in. Your tiered diagnostic protocol could be exactly that contribution.

Ready for the C1 deep research uploads whenever you have them, or if you want to move to formal rubric scoring on what we have.