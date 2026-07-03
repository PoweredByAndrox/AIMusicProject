---
name: aris-skill-router
description: Use when a non-trivial project task may need ARIS skills, workflow routing, engineering route selection, architecture gates, TDD, browser/runtime verification, ML/audio experiment routing, or the user asks to use ARIS Skill Router before solving a task.
---

# ARIS Skill Router

Select the smallest defensible skill chain and engineering route before starting any non-trivial task. The router exists to prevent guessing, broad rewrites, pattern-for-pattern's-sake, fallback fixes, fake verification, and overloaded skill chains.

## Required References

Before routing, read:

- `docs/ARIS_skills_usage_map_ru.md`
- the current task evidence: user files, logs, tests, manifests, source files, or screenshots relevant to the task
- the available skill inventory from project-local `.agents/skills/*/SKILL.md` frontmatter or the session skill list

If maintaining this router, also inspect:

- `AGENTS.md`
- `.agents/skills/aris-skill-router/SKILL.md`

Do not bulk-load every skill body. After route selection, read only the selected skills' `SKILL.md` files before executing them. If a required reference is missing, report:

- Unavailable reference:
- Replacement / fallback:
- Coverage loss:
- Confidence impact:

## Output Contract

Before implementation or deep analysis, output a **Professional Skill Routing Block**. In projects that require an **ARIS Skill Selection Block**, this same block satisfies that requirement when it includes the selected skills, rejected skills, execution order, stop conditions, and verification gates.

Use this format:

```md
# Professional Skill Routing Block

## 1. Task classification
- Primary type:
- Secondary types:
- Risk level: low / medium / high
- Expected blast radius:

## 2. Evidence available
- User-provided evidence:
- Logs/tests/screenshots/files available:
- Missing evidence:

## 3. Architecture Gate
Decision: negative / positive / unknown
Reason:
Architecture smells checked:
- business logic mixed with UI/controller/infrastructure: PASS/FAIL/UNKNOWN
- duplicated logic in 3+ places: PASS/FAIL/UNKNOWN
- wrong dependency direction: PASS/FAIL/UNKNOWN
- tight coupling blocks tests: PASS/FAIL/UNKNOWN
- shared mutable state issue: PASS/FAIL/UNKNOWN
- circular dependencies: PASS/FAIL/UNKNOWN
- generic utils/helpers/common hiding domain logic: PASS/FAIL/UNKNOWN
- local fix would require fallback/retry/sleep/silent catch: PASS/FAIL/UNKNOWN
- same bug class appeared before: PASS/FAIL/UNKNOWN

## 4. OOP / Design Pattern Gate
Decision: no pattern / use pattern / investigate
Reason:
Candidate pattern, if any:
Why a simpler function/module is insufficient:
Tradeoff:
Testability impact:

## 5. Selected route
- Primary route:
- Secondary route, if any:

## 6. Selected skills
Select 3-7 skills in this project; use fewer when enough.
For each:
- skill name
- source: ARIS / external / custom / browser
- why needed
- when used
- expected output

## 7. Rejected skills
List at least 3 tempting but rejected skills.
For each:
- skill name
- why rejected
- what would make it relevant later

## 8. Execution plan
- Step 1:
- Step 2:
- Step 3:
- Patch policy:
- Verification policy:

## 9. Stop conditions
Stop and ask/escalate if:
- reproduction cannot be created
- root cause is not proven
- same metric fails twice
- architecture change is needed but ADR cannot be justified
- tests are impossible without changing public behavior
- fix requires fallback/retry/sleep/silent catch without architectural justification

## 10. Definition of done
- bug reproduced before fix, when this is a bugfix
- root cause proven
- minimal patch or justified ADR
- build passes, when applicable
- relevant tests pass
- runtime/E2E check passes, when runtime is involved
- regression matrix passes
- logs clean
- result-to-claim supports the exact claim, when results/quality/metrics are claimed
```

The router must not start implementation before this block unless the user explicitly asks for planning only or a tiny direct answer.

## Global Engineering Rules

- Do not edit before reproducing the issue, unless the task is explicitly planning-only.
- One patch = one hypothesis.
- Do not change unrelated files.
- Do not use broad rewrites for local bugs.
- Do not use architecture refactor before Architecture Gate is positive.
- Do not use design patterns by default.
- Do not use OOP where simple modules/functions are cleaner.
- Do not add fallback/retry/sleep/silent catch as a fix unless root cause proves it is architecture-correct.
- Do not suppress errors to make tests pass.
- Do not change expected behavior to match the bug.
- Do not trust source files alone; verify the running artifact when runtime is involved.
- Do not claim done without before/after evidence.
- After two failed attempts on the same metric, stop and write a Failed Hypothesis Report.
- After three failed repair rounds, stop and write an engineering report instead of continuing to guess.

## Skill Selection Rules

- Prefer fewer skills; do not select skills just because they exist.
- Project default is 3-7 selected skills. If a reference preset lists 8+ skills, trim it to the smallest useful chain.
- If a selected skill is unavailable, report unavailable/fallback/coverage loss/confidence impact and use a fallback only when it preserves the verification contract.
- Do not use research-heavy skills for a simple bugfix.
- Do not use `auto-review-loop` before reproduction or concrete source-of-truth evidence exists.
- Do not use architecture/refactor skills until source-of-truth evidence makes Architecture Gate positive, or use Architecture Verification Path while the gate is unknown.
- After selection, read selected skills' `SKILL.md` files before executing them.

## Route Definitions

Use one primary route and, only when the evidence demands it, one secondary route.

### Local Root-Cause Bugfix Path

Use when root cause appears local and does not require architecture change.

Prefer skills: `/system-profile`, `/experiment-plan`, `/ablation-planner`, `/analyze-results`, `/experiment-audit`, `/result-to-claim`; add a debugging/checklist skill only if available.

Rules:

- Reproduce first.
- Identify the exact failing function/file/rule.
- Add or run a failing test if possible.
- Patch minimal surface.
- Verify the exact claim.
- Do not refactor unrelated code.

### Architecture Verification Path

Use when it is unclear whether the root cause is local or architectural.

Prefer skills: `/experiment-plan`, `/analyze-results`, `/experiment-audit`, `/result-to-claim`, `/kill-argument` if available, plus codebase-orientation/codebase-recon/software-architecture equivalents if available.

Rules:

- Do not refactor yet.
- Map module boundaries.
- Identify dependency direction.
- Identify duplicated business logic.
- Identify coupling that blocks tests.
- Decide local fix vs architecture refactor.

### Architecture Refactor Path

Use only when Architecture Gate is positive.

Prefer skills: `/experiment-plan`, `/ablation-planner`, `/dse-loop`, `/experiment-audit`, `/result-to-claim`, `/kill-argument` if available, and TDD/architecture skills if available.

Before implementation, write an ADR:

```md
# ADR: <title>

## Problem
## Current architecture issue
## Options considered
## Selected option
## Rejected options
## Pattern/OOP decision
- Pattern:
- Why justified:
- Why simpler module/function is insufficient:
- Tradeoff:
## Migration plan
## Tests
## Regression matrix
## Rollback plan
```

Prohibitions:

- No broad rewrite without ADR.
- No new abstraction without a concrete variation point.
- No generic Manager/Helper/Utils classes.
- No class wrapper around one function.
- No pattern for pattern's sake.

### TDD Bugfix Path

Use when behavior can be checked by a test.

Prefer skills: `superpowers:test-driven-development` when available, `/experiment-plan`, `/ablation-planner`, `/analyze-results`, `/result-to-claim`.

Rules:

- First write or identify a failing test.
- Run it and confirm red.
- Implement the smallest fix.
- Confirm green.
- Refactor only after green.
- Keep behavior contract explicit.

### Frontend Browser Verification Path

Use for UI/CSS/drag-drop/layout/runtime browser bugs.

Prefer skills: `playwright` or browser skill, frontend/UI best-practice skill if available, `/experiment-plan`, `/ablation-planner`, `/experiment-audit`, `/result-to-claim`.

Rules:

- Reproduce in browser before editing.
- Capture screenshot and console errors.
- Inspect DOM and computed styles.
- Verify served app, not only source.
- Use Playwright/E2E check after patch.
- For CSS bugs, measure bounding boxes if layout is involved.

Served-artifact check:

- source file changed
- build artifact changed
- running frontend/container uses new artifact
- browser no-cache reload confirms fix

### Backend/API Log Forensics Path

Use for API bugs, worker bugs, integration errors, wrong responses, queue/runtime failures.

Prefer skills: `/system-profile`, `/experiment-plan`, `/analyze-results`, `/experiment-audit`, `/result-to-claim`, `auto-review-loop` only after reproduction and defined gates; add debugging/checklist skill only if available.

Rules:

- Reproduce request/job/event.
- Capture input, output, logs, stack trace.
- Identify service boundary where behavior diverges.
- Check env/config/database/queue assumptions.
- Add integration test if possible.
- Do not swallow exceptions.

### Docker/Infra Served Artifact Path

Use when the issue may be container, stale build, wrong env, wrong port, volume, cache, service graph, or served artifact mismatch.

Prefer skills: `/system-profile`, `/experiment-plan`, `/analyze-results`, `/experiment-audit`, `/result-to-claim`.

Rules:

- Identify actual running process/container.
- Compare source vs served artifact.
- Check image rebuild/recreate.
- Check env vars and mounted volumes.
- Check ports and healthchecks.
- Do not restart unrelated services unless required.

### ML/Audio Metrics Experiment Path

Use for music/radio/stem separation/LLM metrics/model quality.

Prefer skills: `/experiment-plan`, `/ablation-planner`, `/run-experiment`, `/monitor-experiment`, `/analyze-results`, `/experiment-audit`, `/result-to-claim`; add `/dse-loop` only for controlled design-space exploration.

Rules:

- One experiment = one changed variable.
- Lock evaluation dataset/holdout.
- Do not trust a single metric.
- Produce metrics + artifacts + failed cases.
- Use audit before quality claims.
- Use result-to-claim before saying improved.

For audio/stem separation, require these evidence surfaces when applicable:

- `manifest.json`
- `metrics.json`
- `leakage_matrix.json`
- `source_conservation.json`
- `listen_pack_manifest.json`
- `rejected_cases.json`

### Performance Profiling Path

Use for speed, memory, CPU/GPU, latency, throughput, or blocking IO problems.

Prefer skills: `/system-profile`, `/experiment-plan`, `/analyze-results`, `/ablation-planner`, `/result-to-claim`.

Rules:

- Measure before optimizing.
- Identify bottleneck with profiler/logs/trace.
- Optimize one bottleneck at a time.
- Compare before/after numbers.
- Do not trade correctness for speed.

### Security/Safety Audit Path

Use for auth, tokens, permissions, injection, secrets, unsafe file/network operations.

Prefer skills: codex-security skills if available, `/experiment-plan`, `/kill-argument` if available, `/experiment-audit`, `/result-to-claim`.

If no security-specific skill is available, write:

```text
Security-specific skill unavailable. Proceeding with conservative manual security checklist.
```

Rules:

- Do not log secrets.
- Do not weaken validation.
- Do not bypass auth for tests.
- Do not store tokens insecurely.
- Add regression checks for exploit path.

### Research/Unknown Problem Path

Use when the problem is poorly defined or a new method/approach is needed.

Prefer skills: `/idea-creator`, `/research-refine`, `/research-review`, `/experiment-plan`, `/dse-loop`, `/kill-argument` if available.

Rules:

- Do not implement during research step.
- Generate options.
- Rank by feasibility, risk, blast radius, and testability.
- Convert the best option into an experiment plan.
- Ask before broad architecture change.

## OOP / Design Pattern Gate

Patterns are optional tools, not defaults.

Use OOP/patterns when:

- domain invariants must be protected;
- behavior varies by provider/strategy/state;
- lifecycle/state needs encapsulation;
- dependency inversion reduces coupling;
- tests become simpler;
- there is a real variation point.

Avoid OOP/patterns when:

- a pure function is enough;
- the class only wraps one function;
- abstraction has only one implementation and no credible second implementation;
- pattern increases test complexity;
- naming becomes generic: Manager, Helper, Utils, Common, Misc;
- pattern hides instead of clarifies business logic.

Common allowed patterns:

- Strategy: multiple interchangeable algorithms/providers.
- Adapter: isolate external APIs/libraries.
- Facade: simplify complex subsystem boundary.
- Repository: isolate persistence from domain/application logic.
- Factory: object creation has variants or invariants.
- State: behavior depends on explicit state transitions.
- Command: operations need queueing, undo, logging, or delayed execution.
- Observer/Event: decouple event producers and consumers.
- Dependency Injection: remove hardcoded dependencies and improve tests.

Before using a pattern, write:

- Pattern:
- Problem it solves:
- Why simpler function/module is insufficient:
- Alternatives rejected:
- Tradeoff:
- Test plan:

## Clean Code Gate

Before finalizing a patch, check:

- Function/module has one reason to change.
- Business logic is not hidden in UI/controller/infrastructure.
- Names are domain-specific, not generic.
- No duplicated logic introduced.
- No dead code.
- No broad catch without handling.
- No silent fallback.
- No global mutable state unless justified.
- No circular dependency.
- Public behavior is explicit and tested.
- Error states are represented intentionally.
- New abstraction has a real purpose.

## Failed Hypothesis Report

Required after every failed attempt:

```md
## Failed hypothesis
What I believed:

## Evidence against it
What disproved it:

## Metric/check still red

## Patch attempted

## Why it failed

## Banned repetition
What I will not try again:

## Next hypothesis
Exact next check:
```

## Auto Review Loop Policy

Use `/auto-review-loop` only after:

- bug is reproduced;
- first patch failed or high-risk patch needs review;
- verification gates are defined.

Max rounds: 3.

Do not use `auto-review-loop` to endlessly guess. After 3 failed rounds, stop, write an engineering report, list unresolved hypotheses, and ask for missing evidence or a human decision.

## Final Engineering Report

End non-trivial work with:

```md
# Final Engineering Report

## Summary

## Root cause

## Route selected

## Skills used

## Architecture Gate result

## OOP/Pattern decision

## Files changed

## Why this is not a workaround

## Verification performed
Commands/tests/checks:

## Before/after evidence

## Regression matrix

## Remaining risks

## What was intentionally not changed

## Confidence
0-100:
```

## Router Self-Check Cases

When editing this router, validate it mentally or with a non-production dry run on these cases:

- CSS layout bug where source changed but browser still shows old UI: Frontend Browser Verification Path, with Docker/Infra Served Artifact Path as secondary when stale artifact is plausible.
- Backend API returns wrong result in one validator function: TDD Bugfix Path or Local Root-Cause Bugfix Path.
- Same business-rule bug appears in API, worker, and UI: Architecture Verification Path first, then Architecture Refactor Path only if the gate becomes positive.
- Audio stem separation metric improves but sound is worse: ML/Audio Metrics Experiment Path with `/experiment-audit` and `/result-to-claim`; do not claim quality from one metric.
