# Docs Audit Design — mlx-triage

_Created: 2026-03-02_

---

## Goal

Clean up `docs/` so it's public-ready for long-term project commitment. Remove personal references, AI tool attributions, and raw session artifacts. Keep the evidence trail and design rationale intact.

---

## Principles

1. `docs/` is public-facing — everything there should look intentional and professional
2. Dev workflow files live in `_private/` (gitignored), not in the public tree
3. No AI tool attribution anywhere — all work is authored by the developer
4. Research and evidence stays in-repo (scrubbed), preserving the project's credibility
5. All changes are local-only until explicitly reviewed and approved for push

---

## Structure After Audit

```
docs/
├── plans/                  # Design docs and implementation plans (scrubbed)
│   ├── 2026-02-25-phase-0-scaffolding.md
│   ├── 2026-02-25-phase-1-statistical-core.md
│   ├── 2026-02-28-v01-integration-and-launch.md
│   ├── 2026-02-28-v01-launch-design.md
│   └── 2026-03-02-docs-audit-design.md
├── context/                # Research context summaries (scrubbed)
│   ├── About the mlx-triage spec.md
│   ├── After Submission of C1 Research Artifacts.md
│   └── After Submission of C2 Research Artifacts.md
├── evidence/               # Primary evidence artifacts (scrubbed)
│   ├── claude-c1-infrastructure-defects.md
│   ├── consensus-c1-prisma-review.pdf
│   ├── experiment-consolidation.md
│   ├── gemini-c1-infrastructure-defects.md
│   └── How the UI Assistant Project connects to mlx-triage as evidence.md
├── research/               # Research outputs (scrubbed markdown, raw PDFs/CSV)
│   ├── claude-C2-compass_artifact_wf-*.md
│   ├── consensus-c1-paper-metadata.csv
│   ├── Elicit-C1-*.pdf
│   ├── Elicit-C2-*.pdf
│   └── gemini-C2-MLX Infrastructure vs. Model Limitations.md
├── roadmap.md              # Clean — no changes needed
└── validation-results.md   # Clean — no changes needed

_private/                   # Gitignored — dev workflow only
├── agents.md               # Agent orchestration patterns
└── current-state.md        # Session handoff notes
```

---

## Scrubbing Rules

### Remove

- AI tool names used as authors/contributors (e.g., "Claude Deep Research produced this")
- "For Claude" instructions, "Primary agent: Claude Code", agent workflow language
- Personal identifiers where they add no technical value ("Swaylen's M2 Max" → "M2 Max")
- Session-specific workflow references ("In this session we asked Claude to...")

### Keep

- Methodology descriptions ("systematic review", "deep research synthesis") — these describe process, not attribution
- Technical content, citations, analysis, and findings
- Tool names when used descriptively in methodology context (e.g., "used for literature search") — but NOT as author/contributor credits
- Hardware references where technically relevant ("tested on M2 Max 96GB")

### Don't touch

- PDF files (binary, not scrubable in-place)
- CSV files (raw data)
- `roadmap.md` and `validation-results.md` (already clean)

---

## File-by-file Actions

### Move to `_private/`

| File | Reason |
|------|--------|
| `docs/agents.md` | Internal dev workflow, not public content |
| `docs/current-state.md` | Session handoff notes, not public content |

### Scrub in `docs/`

| File | What to scrub |
|------|---------------|
| `docs/context/About the mlx-triage spec.md` | Personal project references, AI session framing |
| `docs/context/After Submission of C1 Research Artifacts.md` | AI agent names as authors, session framing |
| `docs/context/After Submission of C2 Research Artifacts.md` | Same |
| `docs/evidence/How the UI Assistant Project connects...` | Cross-project personal references, generalize |
| `docs/evidence/claude-c1-infrastructure-defects.md` | AI tool attribution |
| `docs/evidence/experiment-consolidation.md` | Personal experiment references, AI workflow language |
| `docs/evidence/gemini-c1-infrastructure-defects.md` | AI tool attribution |
| `docs/plans/2026-02-25-phase-0-scaffolding.md` | "For Claude" instructions |
| `docs/plans/2026-02-25-phase-1-statistical-core.md` | Same |
| `docs/plans/2026-02-28-v01-integration-and-launch.md` | AI tool references throughout |
| `docs/plans/2026-02-28-v01-launch-design.md` | "Primary agent: Claude Code" |

### No changes needed

| File | Reason |
|------|--------|
| `docs/roadmap.md` | Already scrubbed in session 3 |
| `docs/validation-results.md` | Technical data, no personal/AI refs |
| `docs/research/*.pdf` | Binary files |
| `docs/research/consensus-c1-paper-metadata.csv` | Raw data |

### Needs review (markdown in research/)

| File | Assessment needed |
|------|-------------------|
| `docs/research/claude-C2-compass_artifact_wf-*.md` | Check for AI attribution |
| `docs/research/gemini-C2-MLX Infrastructure vs. Model Limitations.md` | Check for AI attribution |

---

## Housekeeping

- Add `_private/` to `.gitignore`
- Add `.DS_Store` to `.gitignore`
- Remove `docs/.DS_Store` from git tracking
- Remove `_private/` dir entries from git tracking via `git rm --cached`

---

## Review Gate

All scrubbed files will be diffed and presented for review before any commit. Nothing gets pushed to GitHub without explicit approval.
