# C2 Cross-Analysis Research Synthesis

This document synthesizes findings from multiple independent research analyses conducted on the disconfirming leg (C2) of Variant C — examining whether quality ceilings are primarily weight-encoded rather than infrastructure-encoded.

## Cross-Analysis Agreement Assessment

The convergence across all analyses is remarkably tight. Every analysis landed on the same three core claims:

**Claim 1: Quality ceilings are weight-encoded, not infrastructure-encoded.** All analyses found evidence that when model capability is the limiting factor, infrastructure optimization (including quantization scheme changes, memory management, batching) doesn't move the needle on output quality. Multiple analyses independently cited the Red Hat study — one highlighting the 8x payoff from distillation vs 3x from quantization, another noting scores approaching random guessing even at full precision — triangulating to the same high-quality source.

**Claim 2: MLX ≈ CUDA at practical quantization levels.** All analyses agree. The strongest specific finding is the non-determinism study showing Q4_K_M and Q8_0 achieve perfect reproducibility on MLX — and that the non-determinism mechanism (floating-point non-associativity) is identical on CUDA. This effectively neutralizes the "MLX-specific artifact" concern for quantized models, which is the actual deployment scenario.

**Claim 3: Diagnostic effort of decoupled testing has diminishing returns.** All analyses converge, though with varying depth. The most practically useful contribution here surfaced actual MLX GitHub issues where "bad output" was traced to LoRA scaling/adapter merge problems (model-level), not runtime bugs. That's the closest thing to a real postmortem case study in the set.

## Where the Analyses Diverge (and it matters)

**Depth of MLX-specific evidence varies significantly.** Two analyses went deepest on MLX-specific sources — the non-determinism blog post, the iMatrix calibration confound, the MLA/MoE bottleneck migration. One found the most operationally useful GitHub-level evidence (actual bug reports resolved as model issues). Another was the thinnest — mostly restating the consensus without novel sources. The broadest framing leaned heavily on general MLOps literature rather than MLX-specific evidence.

**The float16 instability caveat.** Two analyses flagged that float16 MLX inference is "catastrophically unstable" while quantized integer models are fine. This is an important nuance that the other analyses missed entirely. For the project it's not directly relevant (the target is quantized models), but it's the kind of edge-case knowledge that could matter when testing at higher precision.

**The counter-counter evidence.** One analysis explicitly noted that iMatrix-based quants diverge based on calibration dataset, making clean infrastructure-vs-model attribution difficult even when attempted. This is actually a partial *confirmation* of the original hypothesis hiding inside the disconfirming results — it's saying "yes, the diagnostic is hard, but that's because the confound is real, not because it doesn't exist." Another analysis gestured at this with an outlier-aware quantization discussion but didn't make the connection as crisply.

## What This Means for the Hypothesis

The disconfirming evidence is strong and consistent. But I'd characterize it as strong evidence for a **refined** version of the hypothesis rather than a clean rejection. What the evidence actually says is:

At practical quantization levels (Q4-Q8) on MLX, infrastructure-vs-model separation is **low-value for quality debugging** because MLX fidelity is equivalent to CUDA and most quality issues are weight-encoded. However, the original hypothesis isn't entirely wrong — it's just that the *mechanism* it proposes (MLX-specific artifacts mimicking model defects) is real but confined to edge cases (float16, specific calibration mismatches, early-stage framework bugs) rather than being a systematic concern at production quantization levels.

The practical implication for the project: **a standing infrastructure-vs-model diagnostic protocol for routine development is probably unnecessary.** Instead, a lightweight triage checklist is more appropriate: if output quality is bad, first try better weights, then check quantization format, then check prompt template — and only if all three fail, go into infrastructure-level debugging. That's a much cheaper intervention than the full decoupled testing the original hypothesis implied.

## Source Overlap Observations

The source overlap across analyses is high — the Red Hat half-million evaluations study, the MLX non-determinism blog, the Apple MLX benchmarking paper, and the Scalastic Apple-vs-CUDA comparison appear in 4+ of the 6 analyses. That's good for confidence but means the effective unique source count is lower than the raw citation count suggests. Roughly 15-20 truly independent sources across all analyses, with maybe 8-10 high-quality enough to pass intake.

The agreement band across all analyses is **high** (spread ≤ 10) — they're all telling essentially the same story with the same evidence. That's a strong signal.

Given how strong and consistent the disconfirming evidence is, the confirmatory leg (C1) becomes especially important — not to overturn these findings, but to surface the **boundary conditions** where infrastructure effects *do* matter. The float16 instability, the iMatrix calibration confound, the early-framework-bug cases — those are the conditions where the original hypothesis holds. C1 surfaces more of these, completing the picture: "infrastructure debugging matters under conditions X, Y, Z; otherwise, upgrade weights first."

## What the Deep Research Tier Adds to C2

The three deep research outputs add substantial depth, novel sources, and one critical finding beyond the initial analyses.

### In-Depth Analysis (C2)

This is the strongest individual output in the entire Variant C evidence set — across both legs, across all analyses. Three things make it stand out:

**First, it found quantitative benchmarks the initial analyses missed.** The oobabooga community benchmark showing the entire spread across all major 4-bit methods spans just 0.069 perplexity points on Llama-2-13B. The JarvisLabs benchmark showing AWQ, GGUF Q4_K_M, BitsandBytes NF4, and Marlin-AWQ all hitting identical 51.8% HumanEval Pass@1 on Qwen2.5-32B. The ACL 2025 "Give Me BF16 or Give Me Death" paper with over 500,000 individual evaluations. None of these appeared in any initial analysis.

**Second, it found the MLX-VLM testing report (GitHub issue #682)** — 38 VLMs tested on Apple Silicon, 12 showed quality issues with working infrastructure, every problem traced to model capability. This is a direct empirical study on the exact target platform.

**Third, and most critically, it identified the gap that the initial analyses papered over.** Every initial C2 analysis confidently stated "MLX inference fidelity is equivalent to CUDA." The deep research dug in and found that **no rigorous cross-framework quality comparison between MLX and CUDA actually exists.** The academic papers all measure latency and throughput, never output fidelity. Apple makes no claims about numerical equivalence. The MLX-vs-CUDA equivalence claim is actually an **untested assumption resting on indirect evidence.** This is a major finding — it means the disconfirming case has a structural weakness that the initial-level search couldn't see.

### Complementary Deep Analysis (C2)

The standout contribution is the **MMLU Pro head-to-head benchmark** between MLX 4-bit and llama.cpp Q4_K_M on Llama-3.2-3B-Instruct: aggregate scores of 36.15% vs 36.10%, with a delta of +0.05% — statistically indistinguishable from the 0.12 standard deviation measured across identical repeated runs. This is the closest thing to a direct MLX-vs-alternative quality comparison in the evidence set, and it partially addresses the gap the in-depth analysis identified, though it's community-sourced rather than peer-reviewed.

This analysis also produced well-structured case studies that the initial analyses only gestured at — the GLM-4.7-Flash doom loop misattribution (community spent weeks debugging MLX when the fix required RLHF training changes), the PaliGemma multimodal refusal pattern (model-level safety alignment, not infrastructure), and the information-theoretic collapse at 2-bit (framework-agnostic, unavoidable). These are useful for the narrative extraction but don't change the claim-level picture.

One unique concept introduced: **"Diagnostic Divergence"** — the principle that true infrastructure failures manifest as immediate catastrophic crashes (OOM, type errors, missing weights), not as gradual quality degradation. Quality degradation is almost always model-level. This is a useful heuristic for the tiered diagnostic protocol.

### Systematic Review (C2)

The systematic review adds methodological rigor — 1,033 papers identified, 852 screened, 478 eligible, 50 included. The PRISMA-style flow diagram lends systematic review credibility that none of the other analyses provide. Its claims-and-evidence table with strength ratings provides a clean summary. But it doesn't surface any sources the other deep research analyses missed, and its coverage of MLX-specific evidence is thin — it leans heavily on Rajesh et al. (2025) and Laskaridis et al. (2024) for the cross-platform equivalence claim.

The research gaps matrix is genuinely useful for narrative extraction — it identifies "Diagnostic cost-benefit for Emerging NPUs" as a complete gap, and "Edge Devices" as underexplored across all dimensions.

## Updated Synthesis: C1 + C2 Complete

With the deep research layer added, the picture sharpens considerably. The core synthesis from my previous analysis holds, but with important refinements:

**The disconfirming case is stronger than the initial analyses suggested** — particularly the quantitative evidence that quantization method choice at 4-bit is nearly irrelevant (0.069 perplexity spread), that cross-generation model improvements are 5-10x larger than quantization penalties, and that practitioners who blamed infrastructure usually found model limitations.

**But it also has a newly exposed structural weakness.** The MLX-CUDA fidelity equivalence claim — which all initial analyses stated confidently — rests on indirect evidence. The closest direct test is the community MMLU Pro benchmark (36.15% vs 36.10%), which is strong but narrow (one model, one benchmark, one quantization level). No peer-reviewed study has conducted the systematic comparison. This is a genuine gap and a potential contribution opportunity for the project.

**The confirmatory case (C1) remains strong on its own terms.** The baa.ai postmortem bugs (silent bfloat16 corruption, float16 overflow on GatedDeltaNet), the Anthropic production postmortem, and the MLX-specific GitHub issues are all real, documented infrastructure failures that produce symptoms indistinguishable from model failure. The deep research C2 outputs don't refute these — they contextualize them as edge cases rather than the common case.

**The refined conclusion is now three-layered:**

1. **In steady-state operation** (well-configured, appropriate quantization, standard transformer architectures), model quality is the binding constraint and "try better weights" is the correct first response. Evidence: strong and deep (500K+ evaluations, multiple case studies, community convergence).

2. **During setup and configuration** (new model onboarding, quantization selection, dtype choice, tokenizer config), infrastructure failures are common and routinely misattributed to model quality. Evidence: strong and specific (4+ documented MLX bugs causing total collapse, vLLM postmortems, Anthropic production incident).

3. **The boundary between these regimes is not well-characterized empirically.** No rigorous framework exists for determining when the system has passed from regime 2 into regime 1. The MMLU Pro benchmark is a start but insufficient. This is both a gap in the literature and a potential contribution from this project.

That third point is probably the most valuable finding from this entire Variant C exercise — it's the core "Value-Add" narrative hook. The research community has strong evidence that infrastructure doesn't matter *in general* and strong evidence that it matters *in specific cases*, but nobody has built the systematic triage framework for practitioners to know which regime they're in. The tiered diagnostic protocol could be exactly that contribution.