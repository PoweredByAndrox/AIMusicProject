---
name: aris-skill-router
description: Use when a non-trivial project task may need ARIS skills, workflow routing, skill-chain selection, or the user asks to use ARIS Skill Router before solving a task.
---

# ARIS Skill Router

Select the smallest effective ARIS skill chain before starting a non-trivial task.

## Required Reference

Read this file before selection:

- `docs/ARIS_skills_usage_map_ru.md`

If the reference is missing, stop and ask for it unless the user explicitly allows a conservative fallback.

## Output

Before acting, write an **ARIS Skill Selection Block**:

1. **Task classification**
   - Choose one or more: bugfix, frontend/UI, backend/API, Docker/infra, ML/audio/metrics, architecture, research, refactor, performance, security, data/pipeline.
2. **Selected ARIS skills**
   - Select 3-7 skills maximum.
   - For each skill, state: name, why it is needed, when it will be used, expected output.
3. **Rejected skills**
   - List at least 3 tempting but rejected skills and explain why they are not needed now.
4. **Execution order**
   - Write the exact chain: selected skill order, concrete action, verification, audit, result-to-claim.
5. **Stop conditions**
   - Define when to stop and ask instead of guessing.
6. **Verification gates**
   - Define what evidence must pass before the task can be called done.

## Selection Rules

- Prefer fewer skills; do not select skills just because they exist.
- Do not exceed 7 selected ARIS skills.
- The 3-7 cap overrides broader reference presets that list more skills; trim those presets to the smallest useful chain.
- If a selected skill is unavailable in the current session, report it and choose the closest available fallback only when it preserves the verification contract.
- Do not use research-heavy skills for a simple bugfix.
- Do not use `auto-review-loop` before reproduction or first concrete evidence exists.
- Do not use architecture/refactor skills unless source-of-truth evidence shows architecture is the problem.
- Do not use fallback, retry, sleep, or silent catch as a fix unless it is proven architecture-correct.
- Use one patch per hypothesis.
- If a hypothesis fails, write a failed-hypothesis note before trying another patch.
- After selection, read the selected skills' `SKILL.md` files before executing them.

## Common Shortlists

| Task | Prefer |
| --- | --- |
| General bugfix | `/system-profile`, `/experiment-plan`, `/analyze-results`, `/experiment-audit`, `/result-to-claim` |
| Frontend/UI bug | `playwright`, `/experiment-plan`, `/ablation-planner`, `/experiment-audit`, `/result-to-claim` |
| Backend/API bug | `/system-profile`, `/experiment-plan`, `/analyze-results`, `/experiment-audit`, `/result-to-claim` |
| Docker/infra | `/system-profile`, `/experiment-plan`, `/analyze-results`, `/experiment-audit`, `/result-to-claim` |
| ML/audio/metrics | `/experiment-plan`, `/ablation-planner`, `/run-experiment`, `/monitor-experiment`, `/analyze-results`, `/experiment-audit`, `/result-to-claim` |
| Architecture | `/idea-creator`, `/research-refine`, `/kill-argument`, `/dse-loop`, `/experiment-plan`, `/ablation-planner`, `/result-to-claim` |
| Repeated failure | `/experiment-audit`, `/kill-argument`, `/result-to-claim`, `/research-wiki`, `/auto-review-loop` with max 3 rounds |

## Hard Limits

- Maximum 7 skills.
- Maximum 3 repair rounds unless the user explicitly extends the loop.
- No patch before reproduction for bugfixes.
- No broad rewrite without explicit approval.
- No done claim without evidence.
