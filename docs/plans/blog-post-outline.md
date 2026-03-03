# Blog Post Outline — Introducing mlx-triage

> For authoring by Swaylen Hayes. This outline provides structure; voice and details are yours.

---

## Title Options

1. "Your MLX Model Isn't Broken — Your Infrastructure Might Be"
2. "I Built a Diagnostic Tool for MLX Model Quality (Here's What I Found)"
3. "Stop Guessing Why Your MLX Model Outputs Garbage"

---

## Structure

### Hook (2-3 sentences)

The problem: you download an MLX model, run it, get bad output. Is it the model? The quantization? An MLX bug? You have no way to know without hours of debugging.

**Guidance:** Lead with the frustration. This is the "I've been there" moment that establishes credibility and hooks readers who have experienced the same thing. Keep it concrete — mention a specific symptom (garbled output, repetition loops, wrong language) rather than staying abstract.

---

### The Diagnostic Ambiguity Problem (1-2 paragraphs)

Key points to cover:

- MLX is great but young — real bugs exist (link to `known_bugs.yaml` examples)
- Model quality and infrastructure quality are confounded — when output is bad, you don't know which layer is at fault
- Practitioners waste time blaming the wrong thing (swapping models when the real issue is a dtype mismatch or a known MLX bug)
- This is not a criticism of MLX — it's a recognition that fast-moving infrastructure needs diagnostic tooling

**Guidance:** Frame this as an industry gap, not an MLX complaint. The tone should be "this is a hard problem that nobody has tooling for" rather than "MLX is buggy."

---

### What mlx-triage Does (2-3 paragraphs + terminal screenshot)

Key points to cover:

- Tiered protocol: cheap checks first, expensive checks only if needed
- Tier 0 runs in under 30 seconds without even loading the model into memory — reads safetensors headers, config JSON, and tokenizer files directly
- Tier 1 loads the model and runs statistical validation: determinism (10 runs at temp=0), reference comparison against PyTorch, perplexity measurement
- If Tier 0 finds critical issues, Tier 1 is skipped — fix the fundamentals first
- Ships with a curated known bugs database that cross-references your MLX version and model architecture

**Guidance:** Include a terminal screenshot here showing a real Tier 0 + Tier 1 run with Rich-formatted output. This is the "show, don't tell" moment. The screenshot should show a healthy model passing all checks — readers want to see what success looks like.

`[TODO: Capture terminal screenshot of a real mlx-triage check run]`

---

### What I Found Testing 15 Models (2-3 paragraphs)

Key data points from validation:

- 15 models tested across 5 families (Qwen, Llama, Phi, LiquidAI, Nanbeige)
- 4 quantization levels (bf16, 8-bit, 4-bit, 4-bit DWQ)
- Sizes from 0.6B to 30B parameters
- 100% determinism at temp=0 across all tested models
- Perplexity range 1.8 -- 4.8 (all within healthy PASS threshold of <15)
- Correctly caught an incomplete download (Qwen2-7B — missing safetensors/tokenizer files)
- Correctly caught an incompatible model type (Qwen3.5-27B VL model with vision tower weights)
- Zero false negatives — all known-good models pass

**Guidance:** The key takeaway to land here is that most issues are configuration or environment, not model quality. This validates the tool's thesis: the diagnostic ambiguity problem is real, and systematic checks resolve it quickly. Use the specific numbers — they establish rigor. Link to the full validation results for readers who want the raw data.

---

### The Known Bugs Database (1 paragraph)

Key points to cover:

- Curated from MLX GitHub issues and community postmortems
- Each entry includes affected versions, architectures, severity, symptoms, and remediation
- Stored as YAML — community-contributable without code changes
- The tool gets smarter as the community contributes, without any new releases needed
- Link to `CONTRIBUTING.md` for how to add an entry

**Guidance:** Frame this as the community flywheel. The tool improves passively as people contribute bugs they've encountered. This is the "help us help you" moment.

---

### Try It (call to action)

```bash
pip install mlx-triage
mlx-triage check ./your-model
```

Include:

- Link to GitHub repo
- Link to issues page for bug reports and feature requests
- Mention Tier 2-3 roadmap (isolation tests, layer-wise activation comparison) as what's coming next
- Invitation to contribute to the known bugs database

**Guidance:** Keep the CTA simple and actionable. One command to install, one command to run. Everything else is secondary.

---

## Target Venues

| Venue | Format | Notes |
|-------|--------|-------|
| dev.to | Full article | Primary publication venue. Good MLX/Apple Silicon audience. |
| Personal blog / site | Full article | Canonical URL if you have one. |
| Hacker News | Show HN post | Title: "Show HN: mlx-triage — diagnose MLX model quality on Apple Silicon". Keep the HN comment brief — link to blog post for detail. |
| MLX GitHub Discussions | Discussion post | Link to blog post + repo. This is where the core MLX community lives. |
| Reddit r/LocalLLaMA | Post | Strong overlap with target audience. Cross-post or link to blog. |
| Apple Silicon ML Discord communities | Message | Brief intro + link. Follow community norms on self-promotion. |

---

## Assets Needed

- [ ] Terminal screenshot of a Tier 0 + Tier 1 run (capture manually with a real model)
- [ ] GitHub repo link (after push to `github.com/swaylenhayes/mlx-triage`)
- [ ] PyPI package link (after `uv build` + publish)
- [ ] Link to validation results (`docs/validation-results.md` on GitHub)
- [ ] Link to known bugs database (`src/mlx_triage/data/known_bugs.yaml` on GitHub)

---

## Timing Notes

- Publish blog post after GitHub repo is public and PyPI package is live
- Post to HN during US morning (Pacific time) for best visibility
- Space out venue posts over 2-3 days rather than blasting everything at once
