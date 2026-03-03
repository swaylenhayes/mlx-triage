# Docs Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean up `docs/` for long-term public readiness — remove AI attribution, personal references, and raw session artifacts. Move dev workflow files to `_private/`.

**Architecture:** Seven tasks executed sequentially. Housekeeping first (gitignore), then file moves, then renames, then scrubs by severity (lightest to heaviest). Final review gate before commit.

**Tech Stack:** git, bash (file moves/renames), editor (scrubs)

**Design doc:** `docs/plans/2026-03-02-docs-audit-design.md`

---

## Task 1: Housekeeping

**Files:**
- Modify: `.gitignore`
- Remove from tracking: `docs/.DS_Store`

**Step 1: Update .gitignore**

Append these lines to `.gitignore`:

```
# Dev workflow (not public)
_private/

# macOS
.DS_Store
```

**Step 2: Remove DS_Store from tracking**

Run: `git rm --cached docs/.DS_Store`
Expected: `rm 'docs/.DS_Store'`

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: add _private/ and .DS_Store to gitignore"
```

---

## Task 2: Move Dev Workflow Files to _private/

**Files:**
- Move: `docs/agents.md` → `_private/agents.md`
- Move: `docs/current-state.md` → `_private/current-state.md`
- Move: `docs/context/About the mlx-triage spec.md` → `_private/context/About the mlx-triage spec.md`

**Why move "About the mlx-triage spec.md":** This file is a raw session transcript ("Let me build this out...", "The user wants me to..."). It's not research — it's a planning conversation. Its useful content (which artifacts to include, directory structure) is already implemented. Not scrubable into professional shape.

**Step 1: Create _private directory structure**

Run: `mkdir -p _private/context`

**Step 2: Remove from git tracking**

Run:
```bash
git rm --cached docs/agents.md
git rm --cached docs/current-state.md
git rm --cached "docs/context/About the mlx-triage spec.md"
```

**Step 3: Move files physically**

Run:
```bash
mv docs/agents.md _private/agents.md
mv docs/current-state.md _private/current-state.md
mv "docs/context/About the mlx-triage spec.md" "_private/context/About the mlx-triage spec.md"
```

**Step 4: Commit**

```bash
git commit -m "chore: move dev workflow files and raw session transcript to _private/"
```

---

## Task 3: Rename Files to Remove AI Tool Names

**Files to rename:**

| Old name | New name | Reason |
|----------|----------|--------|
| `docs/evidence/claude-c1-infrastructure-defects.md` | `docs/evidence/c1-infrastructure-defects-analysis.md` | Remove "claude-" prefix |
| `docs/evidence/gemini-c1-infrastructure-defects.md` | `docs/evidence/c1-infrastructure-defects-taxonomy.md` | Remove "gemini-" prefix, differentiate by content type |
| `docs/research/claude-C2-compass_artifact_wf-01ce8ff7-1c27-4352-8e33-9c0e6d722b3d.md` | `docs/research/c2-model-quality-dominance.md` | Remove "claude-", remove UUID, use descriptive name |
| `docs/research/gemini-C2-MLX Infrastructure vs. Model Limitations.md` | `docs/research/c2-mlx-infrastructure-constraints.md` | Remove "gemini-", simplify name |
| `docs/context/After Submission of C1 Research Artifacts.md` | `docs/context/c1-cross-model-synthesis.md` | Descriptive name reflecting content |
| `docs/context/After Submission of C2 Research Artifacts.md` | `docs/context/c2-cross-model-synthesis.md` | Same |
| `docs/evidence/How the UI Assistant Project connects to mlx-triage as evidence.md` | `docs/evidence/first-party-experiment-analysis.md` | Remove project name, describe content |

**Step 1: Rename tracked files via git mv**

Run:
```bash
git mv docs/evidence/claude-c1-infrastructure-defects.md docs/evidence/c1-infrastructure-defects-analysis.md
git mv docs/evidence/gemini-c1-infrastructure-defects.md docs/evidence/c1-infrastructure-defects-taxonomy.md
git mv "docs/research/claude-C2-compass_artifact_wf-01ce8ff7-1c27-4352-8e33-9c0e6d722b3d.md" docs/research/c2-model-quality-dominance.md
git mv "docs/research/gemini-C2-MLX Infrastructure vs. Model Limitations.md" docs/research/c2-mlx-infrastructure-constraints.md
git mv "docs/context/After Submission of C1 Research Artifacts.md" docs/context/c1-cross-model-synthesis.md
git mv "docs/context/After Submission of C2 Research Artifacts.md" docs/context/c2-cross-model-synthesis.md
git mv "docs/evidence/How the UI Assistant Project connects to mlx-triage as evidence.md" docs/evidence/first-party-experiment-analysis.md
```

**Step 2: Commit**

```bash
git commit -m "chore: rename docs files to remove AI tool names and use descriptive names"
```

---

## Task 4: Scrub Plans (light)

Four plan files need "For Claude" headers and AI workflow references removed.

**Files:**
- `docs/plans/2026-02-25-phase-0-scaffolding.md`
- `docs/plans/2026-02-25-phase-1-statistical-core.md`
- `docs/plans/2026-02-28-v01-integration-and-launch.md`
- `docs/plans/2026-02-28-v01-launch-design.md`

**Step 1: Scrub phase-0-scaffolding.md**

Remove line: `> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.`

**Step 2: Scrub phase-1-statistical-core.md**

Remove same "For Claude" line.

**Step 3: Scrub v01-integration-and-launch.md**

Remove "For Claude" line.
Line 7: Change `Stream A (Tasks 8-11) runs in Claude Code. Stream B (Tasks P1-P6) runs in Codex and can execute in parallel with Stream A.` → `Stream A (Tasks 8-11) handles integration. Stream B (Tasks P1-P6) handles packaging and can run in parallel.`
Line 1070-1071: Change `← Claude Code, Sessions S1-S3` → remove these annotations.
Lines referencing "Claude Code" or "Codex" in the execution order: remove agent names, keep task descriptions.
Line ~1084: Remove `Inter-agent memo checkpoint` line.

**Step 4: Scrub v01-launch-design.md**

Line 6: Change `> **Primary agent:** Claude Code | **Parallel agent:** Codex (packaging tasks)` → remove this line entirely.
Table row "Agent model": Change `Claude Code primary, Codex for packaging, memo checkpoints for ecosystem` → `Primary and parallel workstreams with checkpoints`
Table row "Context resilience": Change `Belt and suspenders until memory migration` → `Multiple redundant state preservation mechanisms`
Section "Stream A" header: Remove `(Claude Code)` from `### Stream A: Implementation (Claude Code)`
Section "Stream B" header: Remove `(Codex — parallel with S2/S3)` → `(Packaging — parallel)`
Section "Stream C": Remove entire "Stream C: Inter-Agent Coordination" section (it's agent-to-agent messaging, internal workflow only).
Section "Stream D" header: Remove `(Claude Code — final session)` → `(Publish)`
Remove subagent usage section (lines about "feature-dev", "code-reviewer", "code-explorer" agents).
Context Resilience section: Remove "Sediment store", "server-memory update" references; keep git commit + docs update steps.
Remove line: `This protocol holds until the Falcor DB memory migration is complete.`

**Step 5: Verify no "Claude", "Codex", "subagent", "agent" references remain in plans/**

Run: `grep -ri "claude\|codex\|subagent\|for claude\|primary agent" docs/plans/`
Expected: No matches (except this plan file and the design doc, which are new).

**Step 6: Commit**

```bash
git add docs/plans/
git commit -m "chore: remove AI workflow references from implementation plans"
```

---

## Task 5: Scrub Context Files (medium)

Two files remain in `docs/context/` after the Task 2 move: the C1 and C2 synthesis notes.

**Files:**
- `docs/context/c1-cross-model-synthesis.md` (renamed in Task 3)
- `docs/context/c2-cross-model-synthesis.md` (renamed in Task 3)

### c1-cross-model-synthesis.md

This is a research synthesis document with genuine analytical value. Scrub:

1. **Remove model-as-author language.** E.g., "Claude Sonnet 4.6 was the clear standout on C1" → "The most comprehensive C1 analysis found..." Focus on what was found, not who found it.
2. **Replace all AI model names used as researchers.** "Claude Sonnet", "Grok 4.1", "GPT 5.2", "Kimi 2.5", "Gemini 3.1 Pro", "Sonar" → use descriptive labels like "one analysis", "a separate analysis", "another research output" or just state the findings directly.
3. **Remove scoring/rubric language.** The V3.2 rubric scoring section at the bottom (lines ~134-146) is internal methodology — remove it.
4. **Remove "your project" / "your hypothesis" / "your stack" language.** Replace with neutral framing: "the project", "the hypothesis", "the target stack."
5. **Keep all technical findings, citations, source references, and analytical conclusions.** The five validated claims, three contested claims, and contribution opportunity sections are excellent and should stay.

### c2-cross-model-synthesis.md

Same treatment:

1. **Remove model-as-researcher names.** "Claude Sonnet and Kimi both flagged..." → "Multiple analyses confirmed..."
2. **Remove cross-model scoring protocol language.** The "Scoring Protocol Observations" section (lines ~32-36) is internal.
3. **Replace "your project" / "your hypothesis" language** with neutral framing.
4. **The deep research sections (Claude/Gemini/Consensus subsections)**: Rename subsections from "Claude Deep Research (C2)" to "Deep Research Analysis 1 (C2)" or similar. Remove provenance attribution, keep the findings.
5. **Keep the analytical substance.** The updated three-layered synthesis, contested claims, and gap identification are all valuable.

**Step 1: Edit c1-cross-model-synthesis.md** per rules above.

**Step 2: Edit c2-cross-model-synthesis.md** per rules above.

**Step 3: Verify no AI model names remain as researchers**

Run: `grep -i "claude sonnet\|grok 4\|gpt 5\|kimi 2\|gemini 3.*pro\|sonar" docs/context/`
Expected: No matches.

**Step 4: Commit**

```bash
git add docs/context/
git commit -m "chore: scrub research synthesis notes of AI tool attribution"
```

---

## Task 6: Scrub Evidence Files (medium)

Four markdown files in `docs/evidence/`.

**Files:**
- `docs/evidence/c1-infrastructure-defects-analysis.md` (renamed in Task 3)
- `docs/evidence/c1-infrastructure-defects-taxonomy.md` (renamed in Task 3)
- `docs/evidence/experiment-consolidation.md`
- `docs/evidence/first-party-experiment-analysis.md` (renamed in Task 3)

### c1-infrastructure-defects-analysis.md

This is a well-written research document about infrastructure defects. Read through carefully.
- Check title and headers — should not attribute authorship to an AI tool.
- Content is professional research writing. Likely minimal changes needed.
- Verify: no "This analysis was produced by..." or similar.

### c1-infrastructure-defects-taxonomy.md

Large research taxonomy document. Same treatment.
- Verify no authorship attribution.
- Content should be clean — it's structured as an academic-style analysis.

### experiment-consolidation.md

First-party experimental evidence. Scrub:
1. **Line 12**: "an interactive UI assistant on Apple Silicon (M2 Max, 96 GB)" → "an interactive application on Apple Silicon (M2 Max, 96 GB)" (generalize "UI assistant" to "application")
2. **References to specific project names**: "the UI assistant project" → "the application project" throughout.
3. **Fork references**: Keep technical details (commit hashes, branch names, serve flags) — these are evidence, not personal info.
4. **The content is excellent first-party evidence.** Keep all experimental findings, data tables, and architectural conclusions.

### first-party-experiment-analysis.md

This file has the heaviest conversational AI framing:
1. **Line 1**: "This is an exceptional piece of work, Swaylen." → Remove.
2. **Throughout**: Replace "your experiment" / "your write-up" / "you went from" → neutral framing ("the experiment", "the write-up", "the project went from").
3. **"For the 'Connecting to the Broader Literature' section you've left as reserved"** → Remove (meta-commentary).
4. **"I'd include it in `docs/evidence/`"** → Remove (meta-commentary about filing).
5. **Keep all the analytical connections** between the experiments and the C1/C2 evidence.

**Step 1: Scrub c1-infrastructure-defects-analysis.md** — read and verify clean.

**Step 2: Scrub c1-infrastructure-defects-taxonomy.md** — read and verify clean.

**Step 3: Scrub experiment-consolidation.md** — generalize project name references.

**Step 4: Scrub first-party-experiment-analysis.md** — remove conversational AI framing, neutralize voice.

**Step 5: Verify**

Run: `grep -i "swaylen\|UI assistant project\|your.*experiment\|your.*write" docs/evidence/`
Expected: No matches (except technical hardware references like "M2 Max").

**Step 6: Commit**

```bash
git add docs/evidence/
git commit -m "chore: scrub evidence files of personal references and conversational framing"
```

---

## Task 7: Scrub Research Files (light)

Two markdown files remain in `docs/research/`.

**Files:**
- `docs/research/c2-model-quality-dominance.md` (renamed in Task 3)
- `docs/research/c2-mlx-infrastructure-constraints.md` (renamed in Task 3)

### c2-model-quality-dominance.md

This is professionally written research ("The model is almost always the problem, not the runtime"). Read through and verify:
- No authorship attribution.
- No conversational framing.
- Content is clean research. Likely no changes needed.

### c2-mlx-infrastructure-constraints.md

Large academic-style research document. Same verification:
- Check for any AI tool attribution in intro/conclusion.
- Content should be clean.

**Step 1: Read and verify c2-model-quality-dominance.md** — fix if needed.

**Step 2: Read and verify c2-mlx-infrastructure-constraints.md** — fix if needed.

**Step 3: Commit (if changes made)**

```bash
git add docs/research/
git commit -m "chore: verify research files are clean of AI attribution"
```

---

## Task 8: Review Gate + Final Commit

**Step 1: Run full diff review**

Run: `git diff --stat HEAD~7` (or however many commits since Task 1)
Run: `git log --oneline` to verify commit history is clean.

**Step 2: Verify no AI attribution remains**

Run: `grep -ri "claude code\|codex\|co-authored-by.*claude\|for claude\|primary agent\|subagent" docs/`
Expected: No matches except in this plan file and the design doc.

Run: `grep -ri "swaylen" docs/`
Expected: No matches (or only in technical hardware context that was explicitly kept).

**Step 3: Verify directory structure matches design**

Run: `find docs/ -type f | sort`
Expected: Matches the structure in the design doc, plus the new plan files.

Run: `ls _private/`
Expected: `agents.md`, `current-state.md`, `context/About the mlx-triage spec.md`

**Step 4: Present diff to user for review before any push**

Show the full diff of all changes. User reviews and approves.

---

## Execution Order Summary

```
Task 1: Housekeeping (.gitignore, .DS_Store)
Task 2: Move 3 files to _private/
Task 3: Rename 7 files to remove AI tool names
Task 4: Scrub 4 plan files (light — remove "For Claude" lines)
Task 5: Scrub 2 context files (medium — remove AI-as-researcher language)
Task 6: Scrub 4 evidence files (medium — remove personal refs, conversational framing)
Task 7: Verify 2 research files (light — confirm clean)
Task 8: Review gate — full diff presented to user before commit/push
```

All changes are local-only until user reviews and approves.
