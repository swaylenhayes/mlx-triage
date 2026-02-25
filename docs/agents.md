# Agents — mlx-triage

_Last updated: 2026-02-25_

> Specialist agents live in `~/.claude/agents/`. This file documents which agents apply
> to this project, when to delegate, and how to orchestrate multi-agent workflows.

---

## Active Agents for This Project

| Agent | Domain | When to use |
|-------|--------|-------------|
| feature-dev | Guided feature development | Each Phase 0 task (TDD cycle) |
| code-reviewer | Code review + quality | After completing each task or logical chunk |
| code-explorer | Deep codebase analysis | When tracing MLX internals for Tier 1+ |

---

## Delegation Guidelines

**Delegate when:**
- Implementing a single task from the plan (each task is self-contained)
- Running TDD cycle (test → implement → verify) for a check function
- Reviewing completed work against the plan

**Don't delegate when:**
- Making cross-task architectural decisions (handle inline)
- Updating project docs (quick updates, handle inline)
- Quick one-liner fixes or typo corrections

---

## Multi-Agent Workflows

### Subagent-Driven Development (Phase 0)

**When to use**: Executing the Phase 0 implementation plan task-by-task.
**Sequence**:
1. Fresh subagent per task from `docs/plans/2026-02-25-phase-0-scaffolding.md`
2. Subagent implements one task (TDD: test → implement → verify → commit)
3. Code-reviewer agent reviews the task output
4. Repeat for next task

### Parallel Check Development

**When to use**: Tasks 5-8 (the four Tier 0 checks) are independent and can run in parallel.
**Sequence**:
1. Ensure Tasks 1-3 are complete (shared dependencies)
2. Launch up to 4 subagents, one per check (dtype, tokenizer, weight, version)
3. Each writes its own test + implementation
4. Merge and run full test suite
5. Review all four together

---

## Notes

- No custom agents in `~/.claude/agents/` for this project yet
- Phase 1+ may need an ML inference specialist agent for Tier 1 tests
- The subagent-driven workflow matches the plan's TDD structure naturally
