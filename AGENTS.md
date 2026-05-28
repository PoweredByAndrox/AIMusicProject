Ты должен использовать следующие MCP + используй сабагентов для ВСЕХ ДЕЙСТВИЙ но не ту же самую модель! + Прочитай правило про сабагентов в текущем проекте
ДЛЯ САБАГЕНТОВ:
1. ДОЛЖНЫ указывать в промпт ЧТОБЫ САБАГЕНТ каждый читал ОТ НАЧАЛА и до КОНЦА файлы для полного контекста данных!!!
2. ДОЛЖНЫ ИСПОЛЬЗОВАТЬ MCP доступные ПРИНУДИТЕЛЬНО А НЕ ПОВЫБОРУ!!!

Работай в режиме **forced source-of-truth execution**.

Твоя задача:
НЕ опираться на память, summaries, старые отчёты или поверхностный просмотр.
Сначала ты обязан прочитать код, тесты, docs и артефакты, и только потом делать выводы.

==================================================
ALWAYS-APPLY PRODUCTION ADAPTIVE AUDIO RULE
==================================================

Любая работа по audio separation должна делаться как обобщаемая production-система, а не как разовый фикс под один трек.

RALPH-AI delivery contract:
- Reference: если есть verified reference stem, он задает objective success criteria для конкретного stem;
- Audit: каждый candidate должен проходить objective gate + exact-reference waveform audit перед выдачей как результат;
- Lock: rejected/failed/skipped candidates запрещено представлять пользователю как production/listen result, даже если это лучший из плохих candidates;
- Produce: выдавать можно только artifact со статусом `delivery_status=accepted`, `delivery_allowed=true`, `presentation_status=accepted_for_listen_pack`;
- Halt/iterate: если exact/reference gate failed, продолжать model routing / training / repair loop или фиксировать explicit rejection reason, но не отдавать загрязненный candidate ради результата;
- diagnostic artifacts допустимы только как forensic/debug evidence с явным blocked/rejected статусом и не считаются результатом для прослушивания.

Обязательный contract:
- не добавлять per-track hardcodes: имена треков, фиксированные timestamps, ручные EQ/threshold values или special-case ветки допустимы только как diagnostic artifacts, но не как production route;
- не подменять adaptive routing быстрым single-candidate shortcut, если задача требует качества на разных треках;
- не использовать fallback/фолбек как implementation strategy: не маскировать неудачный inference как успешный результат, не подставлять “что-нибудь похожее” вместо решения, не объявлять degraded route успешным baseline;
- использовать quality-controlled model routing: несколько candidates, segment-level router, proxy metrics, issue diagnostics, feedback priors, provenance-safe reuse и repeatable Docker-only artifacts;
- если качество недостаточно, добавлять production-grade route/candidate с явным contract, manifest provenance, proxy scoring, tests и docs; proxy/model-routing paths (`--pure-base`, `--ir-instrumental-minus-path`, `--external-vocal-candidate`, `quality_consistent_ir_router`, provenance-safe reuse rejection) допустимы только как проверяемые архитектурные входы, а не как скрытая подмена baseline;
- для clean-minus baseline не использовать `build_vocal_guided_minus.py` как основной путь; baseline должен идти через documented separation model route (`export_instrumental_resurrection_minus.py`) и self-only/pure-base repair, если repair вообще нужен;
- для vocal stem использовать обобщаемый adaptive route: BS/FT/IR-residual/external candidates, debleed gates, segment-router, issue reports и feedback store; backing vocals считаются vocal content;
- если качество недостаточно, улучшать общую архитектуру/метрики/router/scoring/tests/docs, а не добавлять ручной фикс под один файл.

==================================================
ALWAYS-APPLY DOCKER EXECUTION RULE
==================================================

Любая project execution работа должна выполняться через Docker, а не через host environment.

Обязательный contract:
- не устанавливать и не подтягивать project dependencies на host;
- не чинить host PATH, Python, ffmpeg, CUDA, Node, package managers или системные библиотеки как способ выполнить project task;
- все project scripts, tests, builds, audio/ML inference, ffmpeg decode/encode, metrics, benchmark, artifact generation и validation запускать внутри Docker/container routes;
- host shell допустим только для легкой read-only инспекции, проверки наличия файлов, просмотра маленьких фрагментов, управления Docker-командами и редактирования файлов в workspace;
- если проверка или task требует runtime dependency, использовать существующий Docker image/service либо обновлять Docker config/image contract, а не host;
- если Docker недоступен, явно сообщить blocker и не подменять Docker-run host-run результатом;
- WSL не использовать для project execution, если пользователь явно не попросил обратное;
- результаты, полученные на host для project runtime команд, считать diagnostic-only и не использовать как production validation.

==================================================
ALWAYS-APPLY AUDIO OBSERVABILITY AND ACCELERATOR RULE
==================================================

Любая audio/ML/separation работа должна быть observable и accelerator-aware. Запрещено запускать долгий route, benchmark, inference, repair, scoring или artifact generation как "черный ящик" без понятного progress trail.

Обязательный contract:
- перед запуском long-running audio route явно зафиксировать execution plan: route name, input mix/reference paths, candidate/model list, Docker service/image, expected output manifest, expected acceleration mode (`gpu_inference`, `cpu_dsp_scoring`, `external_only_cpu_scoring`, etc.);
- каждый long-running script обязан писать structured progress logs с flush: start/end phase, elapsed_sec, current candidate/model/stem, segment/window index или percent, GPU/CPU phase, output manifest path, temporary artifact count, rejection/acceptance gate phase;
- если route запущен с `--external-only` или другим режимом без model inference, лог обязан явно писать: `gpu_inference_skipped=true`, reason, и что текущая стадия является CPU/DSP/scoring/repair, чтобы не гадать по CPU/RAM;
- ML model inference, model candidate generation, neural separation, GPU-capable batch evaluation и heavy tensor operations должны использовать GPU Docker route по умолчанию; CPU допустим только для явно CPU-only DSP, metadata, manifest, lightweight scoring, filesystem orchestration или когда GPU недоступен и это зафиксировано как blocker/diagnostic-only;
- запрещено молча падать на CPU для GPU-capable задач. Если CUDA/GPU/model device недоступны, route должен fail-fast или записать explicit blocker, а не продолжать как production result;
- monitoring должен опираться на source-of-truth evidence: logs, manifest heartbeat/progress file, Docker stats, process state и partial phase markers. CPU/RAM без phase logs не считается достаточным доказательством прогресса;
- если progress не обновляется дольше разумного timeout для текущей фазы, нужно остановить blind run, зафиксировать last known phase/log/process evidence, добавить observability, и только потом перезапускать;
- accepted/listen artifacts запрещено выдавать из route, у которого отсутствует достаточный progress/provenance trail для воспроизведения candidate generation, scoring, gating и accelerator mode.

==================================================
MANDATORY EXECUTION PROTOCOL
==================================================

Ты ОБЯЗАН работать в таком порядке:

## Phase 0 — forced source-of-truth pass
Перед любыми выводами ты ОБЯЗАН:

1. Проверить текущий checkpoint по:
- actual files on disk
- uploaded archive/snapshot if present
- source-of-truth docs
- tests / manifests / runner scripts

2. Прочитать не “пару файлов”, а **полный релевантный срез** по задаче:
- production code files, где реально живёт логика
- связанные test files
- docs/passports/high-risk docs
- manifests / runner scripts / config if task touches flow or baseline

3. Если production code не прочитан, ты НЕ ИМЕЕШЬ ПРАВА:
- говорить DONE
- говорить root cause found
- предлагать final implementation route
- говорить, что понимаешь checkpoint

==================================================
MANDATORY MCP / SKILLS / PLUGIN USAGE
==================================================

Если эти MCP/skills доступны в среде, они ОБЯЗАТЕЛЬНЫ.
Если недоступны — ты обязан явно сказать, что именно недоступно, и чем заменяешь.

### 1. Sequential Thinking MCP — MANDATORY
Используй обязательно:
- для decomposition задачи
- для reconciliation
- для root-cause ordering
- для separation: semantic bug vs threshold issue vs infra issue
- для build-up of exact implementation plan

### 2. Beads / task tracker — MANDATORY
Создай и веди минимум такие task buckets:
- reconcile:source_of_truth
- inspect:production_code
- inspect:tests_and_docs
- implement_or_audit:<task_name>
- artifact_updates:<task_name>
- review:<task_name>

### 3. Repo/context/file tools — MANDATORY
Используй обязательно для чтения:
- actual production code
- relevant tests
- docs/passports
- manifests / scripts / configs

Нельзя ограничиваться:
- filenames
- summaries
- 2–5 sample files
если задача про целый слой / pipeline / subsystem.

### 4. Code review tool — MANDATORY
После изменений или после серьёзных audit-выводов:
- обязателен code review
- с обязательной фиксацией minimum findings / or explicit PASS with reasons

### 5. Archive/file tools — MANDATORY
Если есть архив/снапшот:
- открой и проверь его содержимое
- сравни archive vs disk
- явно перечисли:
  - что подтверждено архивом
  - что подтверждено диском
  - что не подтверждено ничем
  - где есть drift

==================================================
FORCED CODE READING RULES
==================================================

Если задача про баг / interaction / threshold / pipeline / backtest / dashboard / replay / AI integration:

ты ОБЯЗАН прочитать:
1. точку входа
2. центральный orchestrator / pipeline file
3. affected modules
4. affected tests
5. related docs/passports

И в ответе ты обязан явно перечислить:
- какие production files реально прочитал
- какие tests реально прочитал
- какие docs реально прочитал

Если этого списка нет — задача считается выполненной плохо.

==================================================
MINIMUM EVIDENCE REQUIREMENT
==================================================

Каждое сильное утверждение должно быть подкреплено чем-то из этого:
- code path
- test path
- doc statement
- artifact/log evidence

Запрещено:
- “скорее всего”
- “похоже”
- “я думаю”
без явного указания, что это hypothesis и чем она подтверждается/не подтверждается.

==================================================
RECONCILIATION CONTRACT
==================================================

Перед любой реализацией или серьёзным audit ты обязан дать:

1. current checkpoint
2. что подтверждено production code
3. что подтверждено tests
4. что подтверждено docs
5. есть ли drift / contradictions
6. что НЕ подтверждено

Если этого нет — не переходи к реализации.

==================================================
IF TASK IS IMPLEMENTATION
==================================================

Ты обязан работать строго так:

1. short execution plan
2. reconciliation decision
3. formal contract / exact scope
4. mandatory MCP/skill-backed code reading summary
5. implementation
6. tests / validation
7. docs / artifact updates
8. code review
9. final structured report

==================================================
IF TASK IS AUDIT / ANALYSIS
==================================================

Ты обязан работать строго так:

1. short execution plan
2. source-of-truth reconciliation
3. code-backed layered audit
4. severity-ranked issues
5. exact recommendations
6. docs update only if status/route wording changed
7. final structured report

==================================================
HARD PROHIBITIONS
==================================================

- не опирайся только на summary
- не делай выводы только по docs, если есть code
- не делай выводы только по tests, если есть code
- не говори “checked” если реально только посмотрел filenames
- не делай broad refactor без доказанной необходимости
- не меняй бизнес-логику скрытно
- не предлагай следующий шаг по инерции
- не пропускай docs/artifact updates, если они должны измениться
- не заканчивай task без перечисления реально прочитанных production files

==================================================
FAIL CONDITIONS
==================================================

Твоя работа считается плохой и незавершённой, если:
- не использованы MCP/skills, хотя они доступны
- не прочитан реальный production code
- checkpoint не reconciled
- docs/code/tests contradictions не перечислены явно
- выводы сделаны по summary вместо файлов
- нет списка реально прочитанных файлов

==================================================
REQUIRED FINAL FORMAT
==================================================

В конце всегда отвечай структурировано:

1. short execution plan;
2. what is confirmed vs unconfirmed;
3. what was created/updated/changed;
4. what the next safe step is.

Начинай с forced source-of-truth pass.
Если production code не прочитан — сначала читай код, не делай выводы.
<!-- ARIS-CODEX:BEGIN -->
## ARIS Codex Skill Scope
ARIS Codex packages installed in this project: skills-codex
Managed entries: 69
Manifest: `.aris/installed-skills-codex.txt`
ARIS repo root: `/c/Users/PBAX/aris_repo`
Project skill path: `.agents/skills/<skill-name>`
For ARIS Codex workflows, prefer the project-local skills under `.agents/skills/`.
When a skill needs ARIS helper scripts, resolve the repo root from the manifest or set it explicitly:
`ARIS_REPO=$(awk -F'\t' '$1=="repo_root"{print $2; exit}' "/d/AIMusicProject/.aris/installed-skills-codex.txt")`
Do not edit or delete symlinked skills in place; update upstream or rerun:
`bash /c/Users/PBAX/aris_repo/tools/install_aris_codex.sh "/d/AIMusicProject" --reconcile`
For copied Codex installs, use:
`bash /c/Users/PBAX/aris_repo/tools/smart_update_codex.sh --project "/d/AIMusicProject"`
<!-- ARIS-CODEX:END -->

==================================================
AI GIT COMMIT RULES
==================================================

All commits created by an AI agent in this project must follow Conventional Commits.

Format:

```text
<type>(<scope>): <summary>
```

Allowed types:

```text
feat, fix, refactor, perf, docs, test, style, chore, build, ci, revert
```

Commit summary rules:

- write the summary in English;
- use imperative mood;
- keep it concise and specific;
- keep it no longer than 72 characters;
- do not end it with a period;
- do not mention AI, ChatGPT, Codex, Cursor, prompts, or conversations;
- do not use generic summaries like `update`, `fix`, `changes`, `final`, `wip`, `misc`, `save`, or `checkpoint`.

Before any commit, the agent must run and inspect:

```bash
git status --short
git diff
git diff --staged
```

Commit workflow:

- one commit must contain one logical change;
- split unrelated changes into separate commits or ask before committing;
- never commit files that were not inspected;
- do not use `git add .` unless every changed file was inspected and confirmed relevant;
- prefer explicit staging paths or `git add -p`.

Never commit:

- secrets;
- API keys;
- `.env` files;
- debug logs;
- temporary files;
- local IDE/cache files;
- generated junk;
- unrelated changes;
- broken experiments.
