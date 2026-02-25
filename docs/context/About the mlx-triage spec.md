# About the mlx-triage spec

Let me build this out as a proper spec you can hand to your multi-agent development team.Architected tiered diagnostic protocol with phased implementation roadmapArchitected tiered diagnostic protocol with phased implementation roadmapCheck docx skill for creating a professional documentThe user wants me to create a project spec/plan for building the MLX diagnostic tool. They're excited but acknowledge they're at the edge of their knowledge. I should create a comprehensive but approachable spec document that bridges from what they know (product management, Apple Silicon/MLX stack, Python) to what they'd need to build.
Let me create a proper docx spec document using the skill. This should be structured as a product spec they can follow, with clear phases, architecture, and implementation details that are grounded in the evidence we collected.
Let me think about what this tool actually is:

Name: Something catchy - maybe "mlx-triage" or "mlx-diagnose" or building on the GridBae ecosystem
Core concept: A tiered diagnostic protocol that helps practitioners determine whether a quality issue on MLX is infrastructure or model, with four escalating levels—quick sanity checks for dtype and tokenizer config, automated statistical monitoring similar to Token-DiFR, batch invariance testing, and finally deep activation diffing with layer-wise comparisons.

Now I'm deciding on the format—markdown would be more flexible for iteration, but a docx feels more professional and portable for them to share with stakeholders.
Actually, thinking about their workflow, they'll want to hand this off to coding agents and iterate on it, so a well-structured markdown document makes more sense. The tool itself should be a Python CLI that takes a model path as input, runs through the diagnostic tiers in sequence, and outputs a structured report. The key diagnostic techniques are Token-DiFR-style statistical comparison of token outputs against a reference, batch invariance testing by comparing single-batch versus multi-batch runs, and activation diffing to examine layer-by-layer behavior.
Now I'm mapping out the full diagnostic suite: activation comparison between MLX and reference implementations, checksum validation to catch silent bfloat16 corruption, dtype verification to confirm the model's expected precision, tokenizer and EOS token auditing for chat template correctness, and memory pressure monitoring to track quality degradation. For Swaylen's M2 Max setup with 96GB, this is practical since he can run models locally and compare outputs directly.
I'll organize this into phases: Phase 0 for scaffolding and quick checks as a weekend project, Phase 1 for statistical divergence detection using a lightweight Token-DiFR approach, Phase 2 for batch and memory diagnostics, Phase 3 for cross-runtime comparison, and Phase 4 for reporting and packaging contributions.Show moreDoneAbsolutely. Let me build this as a proper spec you can hand to your coding agents or work through yourself. Given you're at the edge of your domain knowledge, I'll make it bridge from what you already know to what you'd need to build — with clear phases so you can ship incrementally.Synthesized technical requirements into actionable project frameworkSynthesized technical requirements into actionable project frameworkCreating the MLX diagnostic tool project spec as a comprehensive markdown documentmlx-triage-spec.mdNow let me copy to outputs and present it.Copy spec to outputsScriptDone. Let me present the file and give a concise summary.Presented fileGood. Let me give a concise summary.DoneHere's your mlx-triage project spec. A few things I want to highlight about how it's structured for where you are right now:
Phase 0 is deliberately a "warm-up." It's all config parsing — no inference, no ML math. You'll learn how MLX models are structured (config.json, safetensors, tokenizer files) just by writing checks against them. The dtype check alone would have caught two of the most catastrophic bugs in the evidence set.
Each tier is independently useful and shippable. You don't need to build all four tiers to have something worth sharing. A Tier 0 + Tier 1 tool that correctly flags dtype mismatches, tokenizer issues, and non-determinism at temperature=0 is already more than anything that currently exists for MLX practitioners.
The "What You Already Know" section is real. Your product management instinct — give the fast answer first, offer depth when needed — is exactly the design philosophy here. And the multi-agent bench you've been building is the perfect way to implement it: hand each tier's module to Claude Code, Codex, and Gemini CLI and compare their approaches.
The known_bugs.yaml is the secret weapon. That curated database of documented MLX failures — drawn directly from the evidence we collected — is where most of the value lives. Every time someone files an MLX issue, you add it to the database, and the tool gets smarter without any code changes.
The spec is written to be handed directly to a coding agent for Phase 0 scaffolding whenever you're ready to start building.



Good question. Not all of them serve the same purpose in the project. Here's how I'd sort them:

**Must-include (direct reference for `known_bugs.yaml` and `METHODOLOGY.md`):**

- **Claude C1** (`claude_compass_artifact...md`) — This is your single best document. It has the specific bug IDs, issue numbers, reproduction steps, and diagnostic methods that map directly to the tool's checks. The Token-DiFR details, the Thinking Machines batch invariance analysis, the Ollama template catalog, the Anthropic postmortem specifics — all of this becomes code.

- **Gemini C1** (`gemini_-_MLX_Infrastructure_vs__Model_Defects.md`) — The 929-bug taxonomy table and the defect category classification (State Corruption, Arithmetic Drift, Tokenization Desync, Scheduling Paradoxes, Quantization Faults) is basically the schema for your diagnostic categories. The cross-runtime comparison table is also directly useful for Tier 3 design.

**Include for methodology/credibility (goes in `docs/evidence/`):**

- **Consensus C1 PDF** (`concensus__Infrastructure...pdf`) — The PRISMA flow diagram and research gaps matrix are what legitimize the contribution claim. When you write the blog post or METHODOLOGY.md, you cite this to show the gap is real and peer-review-confirmed.

**Skip:**

- **Consensus C1 CSV** — This is just the raw data table behind the PDF. You already have the PDF. Unless you want to do your own analysis on the paper metadata, it doesn't add value to the project directory.

For the **C2 outputs** from earlier in our conversation — those are synthesized into the spec already (the three-layered conclusion, the thresholds, the "try better weights first" default). You don't need the raw C2 files in the project directory because their contribution is the *calibration* of the tool's pass/fail thresholds, which is already baked into the spec. But if you still have the Claude C2 deep research and Gemini C2 deep research files, they'd be worth keeping in a `docs/research/` folder for your own reference.

So the project directory structure would look like:

```
docs/
├── METHODOLOGY.md
├── evidence/
│   ├── claude-c1-infrastructure-defects.md      ← primary reference
│   ├── gemini-c1-infrastructure-defects.md      ← taxonomy + cross-runtime
│   └── consensus-c1-prisma-review.pdf           ← gap validation
└── research/
    └── (your C2 files, Perplexity outputs, etc. — personal reference)
```

Want me to set up that directory structure with the uploaded files renamed and organized?