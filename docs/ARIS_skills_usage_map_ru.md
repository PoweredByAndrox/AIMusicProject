# ARIS Skills Usage Map — практическая карта применения

Версия документа: 2026-07-02
Источник структуры: публичный каталог ARIS `docs/SKILLS_CATALOG.md` на GitHub.
Назначение: быстро выбирать ARIS skill под задачу, а не ставить всё подряд и не перегружать агента.

---

## Как пользоваться этим файлом

1. Сначала определи тип задачи: багфикс, аудит, эксперимент, архитектура, статья, патент, презентация, поиск литературы.
2. Выбери 3–8 skills максимум. Не надо подключать 30 skills в один запуск.
3. Для инженерных задач адаптируй research-термины:
   - `experiment` = воспроизводимый тест / benchmark / проверка бага;
   - `claim` = утверждение “исправлено”, “ускорилось”, “качество лучше”;
   - `audit` = проверка, что доказательство не фейковое;
   - `ablation` = менять один фактор за раз.
4. Если задача про код/продукт, добавь свои project-specific gates: build, tests, logs, runtime check, regression matrix.

---

## Быстрые пресеты

### Production bugfix без костылей

```text
/system-profile
/experiment-plan
/research-review
/kill-argument
/ablation-planner
/analyze-results
/experiment-audit
/result-to-claim
/auto-review-loop
/research-wiki
```

Использовать, когда: что-то не работает, есть логи/ошибки/симптомы, нужен root-cause fix.

Главное правило: не редактировать код до воспроизведения бага.

---

### Frontend / UI / CSS bugfix

```text
/system-profile
/experiment-plan
/ablation-planner
/experiment-audit
/result-to-claim
/auto-review-loop
```

Плюс не-ARIS skill/tool: `playwright-interactive`.

Использовать, когда: layout ломается, drag/drop глючит, старый bundle/container отдаёт stale CSS, визуальный баг надо доказать через браузер.

---

### ML/audio/radio/stem separation quality loop

```text
/research-refine
/experiment-plan
/ablation-planner
/run-experiment
/monitor-experiment
/analyze-results
/experiment-audit
/result-to-claim
/kill-argument
/research-wiki
```

Использовать, когда: нужно улучшать качество модели/пайплайна по метрикам, hardcase holdout, leakage matrix, listen-pack, а не “на глаз”.

---

### Архитектурное улучшение

```text
/idea-creator
/research-refine
/dse-loop
/kill-argument
/experiment-plan
/ablation-planner
/result-to-claim
```

Использовать, когда: текущая архитектура тупиковая, надо выбрать между несколькими подходами и доказать, что выбранный лучше.

---

### Поиск литературы / новых методов

```text
/research-lit
/arxiv
/semantic-scholar
/deepxiv
/exa-search
/openalex
/novelty-check
/research-review
```

Использовать, когда: надо найти fresh methods, papers, аналоги, доказать novelty или не изобретать велосипед.

---

### Paper / scientific output

```text
/paper-plan
/paper-write
/paper-figure
/paper-claim-audit
/citation-audit
/proof-checker
/paper-compile
/auto-paper-improvement-loop
```

Использовать, когда: надо превратить результаты в нормальную статью, а не просто текст.

---

## Полная карта ARIS skills

Ниже — все skills из каталога, с практическим назначением, примерами применения и ограничениями.

---

# 1. Workflow Orchestrators

## `/research-pipeline`

**Для чего:** полный end-to-end research pipeline: идея → литература → эксперименты → review loop → paper.

**Когда использовать:**
- нужен полный цикл исследования;
- есть research direction, но ещё нет готовой статьи;
- надо связать discovery, experiment bridge, review и writing.

**Пример запроса:**
```text
/research-pipeline "Improve 6-stem music separation with leakage-aware evaluation and hardcase holdout"
```

**Не использовать:** для простого багфикса. Слишком тяжёлый.

---

## `/idea-discovery`

**Для чего:** workflow поиска research ideas: literature → idea generation → novelty check → review → refinement.

**Когда использовать:**
- надо найти новые методы;
- есть область, но нет конкретной идеи;
- нужна проверка novelty перед экспериментами.

**Пример:**
```text
/idea-discovery "local LLM-assisted radio automation and stem separation quality control"
```

**Не использовать:** когда задача уже конкретная и надо чинить код.

---

## `/idea-discovery-robot`

**Для чего:** адаптер `idea-discovery` для robotics / embodied AI.

**Когда использовать:**
- роботы, симуляторы, embodied AI, manipulation, navigation;
- идеи должны быть benchmark-aware.

**Пример:**
```text
/idea-discovery-robot "improve imitation learning robustness under camera noise"
```

**Не использовать:** для обычного frontend/backend/audio проекта.

---

## `/experiment-bridge`

**Для чего:** взять experiment plan, реализовать код, sanity-check, deploy на GPU, собрать первые результаты.

**Когда использовать:**
- план уже есть;
- надо быстро перейти от идеи к running experiment;
- есть GPU/remote/Vast/Modal.

**Пример:**
```text
/experiment-bridge "Run the approved ablation plan for stem-router-v2 on the locked hardcase set"
```

**Не использовать:** пока нет чёткого experiment plan.

---

## `/auto-review-loop`

**Для чего:** review → fix → re-review до PASS или лимита раундов.

**Когда использовать:**
- есть draft/code/result;
- нужен внешний критический reviewer;
- надо избежать self-approval.

**Пример:**
```text
/auto-review-loop "Review the bugfix for log panel row overlap. Max 3 rounds. Reject fallbacks and unrelated rewrites."
```

**Не использовать:** без лимита раундов. Иначе агент может уйти в бесконечный цикл.

---

## `/auto-review-loop-llm`

**Для чего:** то же, что `/auto-review-loop`, но reviewer через OpenAI-compatible LLM route.

**Когда использовать:**
- хочешь использовать другой LLM как reviewer;
- Codex MCP не подходит;
- нужен независимый reviewer.

**Пример:**
```text
/auto-review-loop-llm "Audit the backend fix using external reviewer; reject hidden retries and swallowed exceptions."
```

**Не использовать:** если нет настроенного `llm-chat` MCP.

---

## `/auto-review-loop-minimax`

**Для чего:** variant review loop, закреплённый за MiniMax API.

**Когда использовать:**
- хочешь MiniMax как external reviewer;
- есть ключ API и проверенный маршрут.

**Пример:**
```text
/auto-review-loop-minimax "Critically review the experiment results and patch proposal."
```

**Не использовать:** если нет причины именно MiniMax.

---

## `/paper-writing`

**Для чего:** orchestrator для статьи: plan → figures → illustrations → writing → compile → improvement loop.

**Когда использовать:**
- есть результаты экспериментов;
- надо собрать submission-ready paper.

**Пример:**
```text
/paper-writing "NARRATIVE_REPORT.md"
```

**Не использовать:** если результаты ещё не подтверждены `/result-to-claim`.

---

## `/rebuttal`

**Для чего:** workflow для ответа reviewers: parse reviews → strategy → draft → safety check → stress test.

**Когда использовать:**
- есть официальные reviews;
- надо написать rebuttal без эмоциональных ошибок.

**Пример:**
```text
/rebuttal "paper/ + reviews" — venue: ICML
```

**Не использовать:** для обычного code review.

---

## `/resubmit-pipeline`

**Для чего:** перенос статьи в новый venue без новых экспериментов: isolation, anonymity, audits, microedit, compile.

**Когда использовать:**
- статью надо адаптировать под другой venue;
- нельзя менять результаты/эксперименты.

**Пример:**
```text
/resubmit-pipeline "paper/" — venue: NeurIPS
```

**Не использовать:** если нужны новые эксперименты.

---

## `/paper-talk`

**Для чего:** paper → slide outline → Beamer/PPTX → polish → assurance audits → final report.

**Когда использовать:**
- надо сделать доклад по статье;
- нужны слайды, speaker notes, script.

**Пример:**
```text
/paper-talk "paper/" — venue: ICLR
```

**Не использовать:** для обычной коммерческой презентации, если нет paper context.

---

## `/research-refine-pipeline`

**Для чего:** sub-pipeline: refine method + plan experiments.

**Когда использовать:**
- идея есть, но она сырая;
- надо получить метод + experiment roadmap.

**Пример:**
```text
/research-refine-pipeline "Leakage-aware stem router for 6-stem radio pipeline"
```

**Не использовать:** когда уже есть готовый план и надо просто запускать.

---

## `/patent-pipeline`

**Для чего:** полный патентный workflow: invention → claims → specification → jurisdiction format.

**Когда использовать:**
- есть изобретение;
- надо подготовить патентную заявку.

**Пример:**
```text
/patent-pipeline "Adaptive audio stem quality routing system for radio automation"
```

**Не использовать:** как замену патентному поверенному для финальной подачи.

---

## `/dse-loop`

**Для чего:** design-space exploration: run → analyze → tune → iterate until objective met.

**Когда использовать:**
- много вариантов архитектуры/параметров;
- есть objective metric;
- надо системно искать лучшее, а не brainstorming без проверки.

**Пример:**
```text
/dse-loop "Compare stem separation routing strategies by leakage, artifact score, runtime, and source conservation"
```

**Не использовать:** если нет метрики успеха.

---

## `/meta-optimize`

**Для чего:** анализ usage logs ARIS и предложение улучшений SKILL.md / prompts / defaults.

**Когда использовать:**
- агент регулярно повторяет ошибки;
- надо улучшить сам workflow;
- есть накопленные logs.

**Пример:**
```text
/meta-optimize "Analyze failed bugfix loops and propose stricter no-fallback verification gates"
```

**Не использовать:** для текущего срочного bugfix вместо самого fix.

---

## `/meta-apply`

**Для чего:** privileged landing gate, который может изменять skill corpus после human approval и fresh jury PASS.

**Когда использовать:**
- есть approved patch к skills;
- надо безопасно применить изменения в corpus.

**Пример:**
```text
/meta-apply "Apply the approved root-cause-audit skill patch after jury PASS"
```

**Не использовать:** без human-in-loop. Это опасный skill.

---

# 2. Literature & Search

## `/research-lit`

**Для чего:** multi-source literature search: Zotero, Obsidian, local PDFs, web, arXiv, Semantic Scholar, DeepXiv, Exa, Gemini, OpenAlex.

**Когда использовать:**
- нужно изучить область;
- надо собрать источники;
- требуется dedup и краткие выводы.

**Пример:**
```text
/research-lit "query-based source separation and leakage-aware audio evaluation"
```

**Не использовать:** если нужен быстрый ответ по одному известному документу.

---

## `/arxiv`

**Для чего:** поиск, скачивание и summarization arXiv papers.

**Когда использовать:**
- нужен arXiv-first поиск;
- надо быстро получить таблицу свежих papers.

**Пример:**
```text
/arxiv "music source separation 2025 RoFormer stem separation"
```

**Не использовать:** для industrial docs/blogs — тогда лучше `/exa-search`.

---

## `/semantic-scholar`

**Для чего:** поиск published-venue papers с citation counts, venues, TLDR.

**Когда использовать:**
- нужны более зрелые published works;
- важно видеть citation/venue metadata.

**Пример:**
```text
/semantic-scholar "source separation leakage evaluation"
```

**Не использовать:** если нужен только preprint-heavy arXiv scan.

---

## `/deepxiv`

**Для чего:** progressive paper reading: search → brief → head → section → trending → web search.

**Когда использовать:**
- надо читать paper глубже, но постепенно;
- нужна структура по sections.

**Пример:**
```text
/deepxiv "RoFormer music source separation"
```

**Не использовать:** без установленного `deepxiv-sdk`.

---

## `/exa-search`

**Для чего:** broad AI web search с content extraction: blogs, docs, news, papers.

**Когда использовать:**
- нужны GitHub/blog/docs/examples;
- research не только academic.

**Пример:**
```text
/exa-search "best open source 6 stem separation model inference pipeline"
```

**Не использовать:** без `EXA_API_KEY`.

---

## `/openalex`

**Для чего:** OpenAlex search по огромному citation graph, affiliations, funding data.

**Когда использовать:**
- нужно картировать область;
- важны citations, institutions, funding.

**Пример:**
```text
/openalex "audio source separation evaluation metrics"
```

**Не использовать:** для быстрого implementation fix.

---

## `/gemini-search`

**Для чего:** Gemini-driven literature discovery: decomposition into sub-problems, aliases, variants.

**Когда использовать:**
- topic широкий и надо разбить его на поисковые ветки;
- нужны альтернативные термины.

**Пример:**
```text
/gemini-search "stem separation artifacts leakage residual reconstruction"
```

**Не использовать:** без `gemini-cli`.

---

## `/alphaxiv`

**Для чего:** быстрый single-paper lookup через AlphaXiv: overview → markdown → LaTeX source fallback.

**Когда использовать:**
- есть конкретная paper;
- надо быстро вытащить readable summary/source.

**Пример:**
```text
/alphaxiv "arXiv:2501.xxxxx"
```

**Не использовать:** для широкого literature survey.

---

## `/comm-lit-review`

**Для чего:** communications-domain literature review: wireless, networking, satellite, Wi‑Fi, cellular.

**Когда использовать:**
- тема из communications/networking;
- нужны domain-specific retrieval assumptions.

**Пример:**
```text
/comm-lit-review "LLM-assisted optimization for wireless resource allocation"
```

**Не использовать:** для audio/frontend/backend задач.

---

## `/novelty-check`

**Для чего:** проверка, насколько идея новая относительно recent literature, с closest-prior-work table.

**Когда использовать:**
- есть идея, но непонятно, не сделано ли это уже;
- перед экспериментами/патентом.

**Пример:**
```text
/novelty-check "Leakage-aware issue-to-action contract for production stem separation pipelines"
```

**Не использовать:** если идея ещё не сформулирована.

---

# 3. Ideation & Method Design

## `/idea-creator`

**Для чего:** brainstorm 8–12 ideas, filter feasibility, pilot on GPU, rank by signal.

**Когда использовать:**
- надо найти новые варианты решения;
- текущий подход тупиковый;
- есть бюджет на pilots.

**Пример:**
```text
/idea-creator "Generate 12 architectures for reducing vocal leakage in 6-stem separation"
```

**Не использовать:** когда нужен один срочный bugfix.

---

## `/research-refine`

**Для чего:** iterative method refinement: problem anchor → review rounds → score threshold.

**Когда использовать:**
- идея есть, но слабая;
- нужен конкретный метод с reviewer pressure.

**Пример:**
```text
/research-refine "Convert a vague stem-quality pipeline idea into a concrete reproducible method"
```

**Не использовать:** если нет problem anchor.

---

## `/experiment-plan`

**Для чего:** claim-driven roadmap: ablations, budgets, run order.

**Когда использовать:**
- надо превратить идею/bugfix в проверяемый план;
- нужно определить gates до правки.

**Пример:**
```text
/experiment-plan "Plan verification for a production bugfix: reproduce, patch, build, E2E, regression, result-to-claim"
```

**Не использовать:** как замену реальному запуску tests.

---

## `/ablation-planner`

**Для чего:** спроектировать ablations с позиции reviewer.

**Когда использовать:**
- надо понять, какой фактор реально влияет;
- агент меняет слишком много сразу.

**Пример:**
```text
/ablation-planner "Test one variable at a time: model route, chunk size, overlap, normalization, postprocess"
```

**Не использовать:** если нет baseline и метрики.

---

## `/formula-derivation`

**Для чего:** структурировать теоретические derivations, assumptions, equation chains.

**Когда использовать:**
- ML/theory/math paper;
- формулы разбросаны и нужно собрать coherent derivation.

**Пример:**
```text
/formula-derivation "Derive the loss decomposition for leakage-aware source separation"
```

**Не использовать:** для обычного кода без математики.

---

# 4. Experiments & Infrastructure

## `/run-experiment`

**Для чего:** deploy experiments to local/remote/Vast/Modal GPU.

**Когда использовать:**
- experiment plan готов;
- надо запустить batch jobs.

**Пример:**
```text
/run-experiment "Run ablation A03 on local GPU; save manifest, metrics, logs, artifacts"
```

**Не использовать:** до sanity-check кода.

---

## `/monitor-experiment`

**Для чего:** monitor running experiments, progress, collect results.

**Когда использовать:**
- долгоиграющие jobs;
- надо ловить dead/idle/crashed runs.

**Пример:**
```text
/monitor-experiment "Poll stem separation benchmark until all hardcase tracks finish or fail"
```

**Не использовать:** для мгновенных команд.

---

## `/analyze-results`

**Для чего:** statistics, comparison tables, insights from results.

**Когда использовать:**
- есть metrics/logs/CSV/JSON;
- нужно понять, что реально изменилось.

**Пример:**
```text
/analyze-results "Compare baseline vs candidate by leakage, artifacts, runtime, failed cases"
```

**Не использовать:** без сохранённых результатов.

---

## `/experiment-queue`

**Для чего:** SSH job queue для sweeps: OOM retry, stale-screen cleanup, wave gating, crash-safe state.

**Когда использовать:**
- много multi-seed/multi-config jobs;
- remote GPU по SSH.

**Пример:**
```text
/experiment-queue "Queue 24 stem-router configs with crash-safe state and OOM retry"
```

**Не использовать:** для одного локального запуска.

---

## `/vast-gpu`

**Для чего:** rent/manage/destroy Vast.ai GPU.

**Когда использовать:**
- локального GPU мало;
- нужны дешёвые on-demand эксперименты.

**Пример:**
```text
/vast-gpu "Rent a 24GB GPU for separation benchmark and destroy after results are collected"
```

**Не использовать:** без контроля бюджета.

---

## `/serverless-modal`

**Для чего:** GPU workloads на Modal: serverless, auto scale-to-zero.

**Когда использовать:**
- хочешь serverless GPU;
- эксперименты можно упаковать в Modal job.

**Пример:**
```text
/serverless-modal "Run batch separation eval on Modal with saved artifacts"
```

**Не использовать:** без Modal account/setup.

---

## `/qzcli`

**Для чего:** manage GPU compute jobs на Qizhi platform через qzcli.

**Когда использовать:**
- ты реально используешь Qizhi;
- нужен kubectl-style job control.

**Пример:**
```text
/qzcli "Submit and monitor training job on Qizhi"
```

**Не использовать:** если Qizhi не часть инфраструктуры.

---

## `/training-check`

**Для чего:** poll W&B metrics during training, catch NaN/loss divergence/idle GPUs.

**Когда использовать:**
- есть W&B;
- идёт training/fine-tuning.

**Пример:**
```text
/training-check "Monitor QLoRA run for divergence, idle GPU, eval regression"
```

**Не использовать:** для inference-only pipeline.

---

## `/system-profile`

**Для чего:** profile target: script/process/GPU/memory/interconnect, instrumentation, actionable report.

**Когда использовать:**
- непонятно, где bottleneck/bug;
- нужно проверить runtime, process, контейнер, память, GPU.

**Пример:**
```text
/system-profile "Profile why the frontend served bundle does not reflect source CSS changes"
```

**Не использовать:** если причина уже доказана и нужен только маленький patch.

---

# 5. Review, Audit & Assurance

## `/research-review`

**Для чего:** single-round deep critical review from external LLM.

**Когда использовать:**
- нужен независимый критик плана/кода/метода;
- надо разнести слабые места до patch.

**Пример:**
```text
/research-review "Critically review this root-cause hypothesis and minimal patch before editing code"
```

**Не использовать:** как формальность после уже сделанного костыля.

---

## `/experiment-audit`

**Для чего:** integrity check experiment code + results: fake ground truth, score normalization fraud, phantom results, scope inflation.

**Когда использовать:**
- метрики выглядят слишком хорошо;
- агент мог проверить не тот artifact;
- надо доказать, что тест честный.

**Пример:**
```text
/experiment-audit "Audit whether the bugfix verification used the served app, not stale source/build artifacts"
```

**Не использовать:** без raw evidence/logs/results.

---

## `/result-to-claim`

**Для чего:** маппинг результатов к claims: что подтверждено, что нет, что нужно ещё.

**Когда использовать:**
- агент говорит “готово”, “улучшилось”, “исправлено”;
- надо проверить, доказан ли claim.

**Пример:**
```text
/result-to-claim "Claim: log panel row overlap is fixed. Evidence: Playwright geometry, screenshot, console logs, build output."
```

**Не использовать:** если claim не сформулирован.

---

## `/paper-claim-audit`

**Для чего:** zero-context numeric verification для paper: все числа/сравнения/scope claims against raw result files.

**Когда использовать:**
- статья содержит метрики;
- надо исключить confirmation bias.

**Пример:**
```text
/paper-claim-audit "Verify all numeric claims in the stem separation paper draft against raw metrics files"
```

**Не использовать:** для обычного bugfix report.

---

## `/citation-audit`

**Для чего:** audit bibliography: existence, metadata correctness, context appropriateness for every citation.

**Когда использовать:**
- paper/grant/rebuttal;
- важна корректность ссылок.

**Пример:**
```text
/citation-audit "Audit every citation in paper.tex; verify context and metadata"
```

**Не использовать:** если текст без ссылок.

---

## `/proof-checker`

**Для чего:** rigorous mathematical proof verification with issue taxonomy and proof-obligation ledger.

**Когда использовать:**
- есть theorem/lemma/proof;
- нужны side-condition checks/counterexamples.

**Пример:**
```text
/proof-checker "Verify the proof of convergence for the proposed optimization method"
```

**Не использовать:** для инженерного bugfix без математического proof.

---

## `/kill-argument`

**Для чего:** adversarial two-thread review: rejection memo + defense + unresolved issues.

**Когда использовать:**
- предлагается крупное архитектурное изменение;
- надо проверить, не костыль ли это;
- нужен “адвокат дьявола”.

**Пример:**
```text
/kill-argument "Attack this proposed fallback-based fix. Is it hiding the root cause?"
```

**Не использовать:** для тривиальных однострочных правок, где overhead не нужен.

---

# 6. Paper Writing & Figures

## `/paper-plan`

**Для чего:** structured paper outline from review conclusions + experiment results.

**Когда использовать:**
- результаты уже есть;
- нужна claims-evidence matrix, section structure, figure plan.

**Пример:**
```text
/paper-plan "Build a paper outline from the final stem separation experiment report"
```

**Не использовать:** без проверенных результатов.

---

## `/paper-write`

**Для чего:** section-by-section LaTeX generation with anti-hallucination BibTeX.

**Когда использовать:**
- есть план статьи;
- надо писать LaTeX секциями.

**Пример:**
```text
/paper-write "Write Methods and Experiments sections from the approved outline"
```

**Не использовать:** чтобы придумывать неподтверждённые results.

---

## `/paper-figure`

**Для чего:** publication-quality plots/tables from experiment results.

**Когда использовать:**
- есть CSV/JSON metrics;
- нужны matplotlib/seaborn figures и LaTeX tables.

**Пример:**
```text
/paper-figure "Generate leakage matrix heatmap and ablation comparison table"
```

**Не использовать:** без raw data.

---

## `/figure-spec`

**Для чего:** deterministic JSON → SVG renderer for architecture/workflow/pipeline/audit diagrams.

**Когда использовать:**
- нужна воспроизводимая схема;
- Mermaid недостаточно контролируемый.

**Пример:**
```text
/figure-spec "Render the bugfix verification pipeline as a deterministic SVG"
```

**Не использовать:** для художественных иллюстраций.

---

## `/paper-illustration`

**Для чего:** AI architecture/method illustrations через Gemini image generation + iterative refinement.

**Когда использовать:**
- нужна красивая method illustration для paper.

**Пример:**
```text
/paper-illustration "Create a clean architecture illustration for the 6-stem quality gate pipeline"
```

**Не использовать:** без Gemini API key.

---

## `/paper-illustration-image2`

**Для чего:** Codex-native image generation alternative через local Codex app-server bridge.

**Когда использовать:**
- не хочешь Gemini;
- есть Codex app-server + `codex-image2` bridge.

**Пример:**
```text
/paper-illustration-image2 "Generate a paper-style figure for audio issue-to-action contract"
```

**Не использовать:** без bridge setup.

---

## `/mermaid-diagram`

**Для чего:** Mermaid diagrams: flowchart, sequence, class, ER, Gantt with syntax verification.

**Когда использовать:**
- docs/README/architecture;
- нужна быстро читаемая схема.

**Пример:**
```text
/mermaid-diagram "Create a sequence diagram for frontend rebuild and served artifact verification"
```

**Не использовать:** если нужен pixel-perfect publication figure.

---

## `/pixel-art`

**Для чего:** pixel-art SVG illustrations for README/docs/slides.

**Когда использовать:**
- нужен декоративный пиксельный стиль;
- branding/README.

**Пример:**
```text
/pixel-art "Create a small pixel-art icon for the research wiki"
```

**Не использовать:** для строгих технических схем.

---

## `/paper-compile`

**Для чего:** compile LaTeX paper to PDF, auto-fix errors, submission readiness checks.

**Когда использовать:**
- paper already written;
- надо собрать PDF и пройти checks.

**Пример:**
```text
/paper-compile "Compile paper.tex and fix LaTeX errors without changing claims"
```

**Не использовать:** если нет LaTeX toolchain.

---

## `/auto-paper-improvement-loop`

**Для чего:** 2-round content review + format check, improve paper quality.

**Когда использовать:**
- есть draft;
- надо поднять качество текста/структуры.

**Пример:**
```text
/auto-paper-improvement-loop "Improve the paper draft while preserving verified claims"
```

**Не использовать:** до `/paper-claim-audit` для чисел.

---

## `/proof-writer`

**Для чего:** draft rigorous mathematical proofs for ML/AI theory.

**Когда использовать:**
- надо написать theorem/lemma/corollary proof.

**Пример:**
```text
/proof-writer "Draft a proof sketch for stability of the proposed routing objective"
```

**Не использовать:** без последующего `/proof-checker`.

---

## `/writing-systems-papers`

**Для чего:** paragraph-level blueprint for systems papers: OSDI/SOSP/ASPLOS/NSDI/EuroSys.

**Когда использовать:**
- пишешь systems paper;
- нужна page allocation и paragraph templates.

**Пример:**
```text
/writing-systems-papers "Plan a 12-page systems paper about an autonomous audio evaluation harness"
```

**Не использовать:** для ML theory paper или коммерческой документации.

---

## `/grant-proposal`

**Для чего:** structured grant proposal drafting: KAKENHI, NSF, NSFC, ERC, DFG и др.

**Когда использовать:**
- нужно оформить research grant;
- есть тема, цели, impact, methodology.

**Пример:**
```text
/grant-proposal "Draft an NSF-style proposal for autonomous audio quality evaluation systems"
```

**Не использовать:** для внутренних задач проекта.

---

# 7. Talks, Posters & Resubmission

## `/paper-slides`

**Для чего:** conference presentation: Beamer → PDF + editable PPTX + speaker notes + script.

**Когда использовать:**
- нужен доклад по paper;
- важны заметки и сценарий.

**Пример:**
```text
/paper-slides "Create conference slides from paper/ with speaker notes"
```

**Не использовать:** для обычной бизнес-презентации без paper.

---

## `/slides-polish`

**Для чего:** per-page Codex review + targeted fixes for PPTX/Beamer.

**Когда использовать:**
- слайды уже есть, но надо polish;
- проблемы с layout, font scaling, overflow.

**Пример:**
```text
/slides-polish "Polish slides for font scaling, frame resize, and anonymity placeholders"
```

**Не использовать:** если слайды ещё не созданы.

---

## `/paper-poster-html`

**Для чего:** default conference poster: single HTML/CSS file with hard gates → print-ready PDF via Chromium.

**Когда использовать:**
- нужен poster;
- важны measurement-driven layout gates.

**Пример:**
```text
/paper-poster-html "Create a print-ready poster from the verified paper figures"
```

**Не использовать:** если нет Playwright/Chromium/PyMuPDF.

---

## `/paper-poster`

**Для чего:** deprecated redirect stub to `/paper-poster-html`.

**Когда использовать:**
- практически не использовать; только если legacy workflow ссылается на него.

**Пример:**
```text
/paper-poster "Redirect to paper-poster-html"
```

**Не использовать:** в новых проектах.

---

# 8. Patents

## `/invention-structuring`

**Для чего:** raw invention idea → formal invention disclosure.

**Когда использовать:**
- есть техническая идея;
- надо структурировать проблему, решение, novelty, embodiments.

**Пример:**
```text
/invention-structuring "Adaptive leakage-aware stem routing for radio automation"
```

**Не использовать:** если идея ещё на уровне “что-то с AI”.

---

## `/claims-drafting`

**Для чего:** draft patent claims: independent + dependent, anti-pattern checks.

**Когда использовать:**
- invention disclosure уже есть;
- надо сформировать claims.

**Пример:**
```text
/claims-drafting "Draft independent and dependent claims for the audio quality gate system"
```

**Не использовать:** без prior-art awareness.

---

## `/embodiment-description`

**Для чего:** detailed embodiment descriptions for patent specification.

**Когда использовать:**
- есть claims;
- надо описать варианты реализации.

**Пример:**
```text
/embodiment-description "Describe embodiments for local, cloud, and hybrid stem-quality routing"
```

**Не использовать:** без claims/spec direction.

---

## `/specification-writing`

**Для чего:** full patent specification from claims + invention disclosure.

**Когда использовать:**
- claims готовы;
- нужно собрать spec.

**Пример:**
```text
/specification-writing "Write full patent specification from approved claims"
```

**Не использовать:** как финальный legal filing без patent attorney review.

---

## `/figure-description`

**Для чего:** formal drawing descriptions for patent figures.

**Когда использовать:**
- есть патентные схемы/figures;
- надо описать FIG. 1, FIG. 2 и т.д.

**Пример:**
```text
/figure-description "Write patent figure descriptions for the stem-quality pipeline diagrams"
```

**Не использовать:** для обычных README diagrams.

---

## `/prior-art-search`

**Для чего:** search patent databases + academic literature for prior art.

**Когда использовать:**
- перед claims;
- надо понять, насколько идея уже покрыта.

**Пример:**
```text
/prior-art-search "Search prior art for adaptive audio stem separation quality gating"
```

**Не использовать:** если нет web access.

---

## `/patent-novelty-check`

**Для чего:** assess patent novelty and non-obviousness against prior art.

**Когда использовать:**
- prior-art найден;
- надо оценить patentability.

**Пример:**
```text
/patent-novelty-check "Evaluate novelty of the issue-to-action stem quality contract"
```

**Не использовать:** без prior-art table.

---

## `/patent-review`

**Для чего:** external patent-examiner-style critical review of patent application.

**Когда использовать:**
- draft application готов;
- нужен examiner-style attack.

**Пример:**
```text
/patent-review "Review the patent draft like an examiner; identify weak claims"
```

**Не использовать:** до claims/spec draft.

---

## `/jurisdiction-format`

**Для чего:** compile patent application into CN/US/EP filing format.

**Когда использовать:**
- draft готов;
- надо привести к jurisdiction format.

**Пример:**
```text
/jurisdiction-format "Format this patent application for US filing"
```

**Не использовать:** если юрисдикция не выбрана.

---

# 9. Meta, Utilities & Integrations

## `/research-wiki`

**Для чего:** persistent research knowledge base: papers, ideas, experiments, claims, typed relationships.

**Когда использовать:**
- проект длительный;
- надо помнить failed hypotheses, claims, papers, experiments.

**Пример:**
```text
/research-wiki init
/research-wiki "Record failed hypothesis: CSS source changed but served container still stale"
```

**Не использовать:** для одноразовой мелкой задачи.

---

## `/wiki-enrich`

**Для чего:** fill per-paper TODO sections left by `ingest_paper`; fetch chain AlphaXiv → DeepXiv → arXiv → page abstract.

**Когда использовать:**
- wiki уже есть;
- в papers есть TODO scaffolds.

**Пример:**
```text
/wiki-enrich "Enrich all TODO paper notes related to source separation metrics"
```

**Не использовать:** без initialized research wiki.

---

## `/render-html`

**Для чего:** render ARIS Markdown/JSON artifacts into reviewed single-file HTML views.

**Когда использовать:**
- надо дать человеку удобный HTML-отчёт;
- есть complex Markdown/JSON artifacts.

**Пример:**
```text
/render-html "Render the bugfix audit report and verification matrix to a single HTML file"
```

**Не использовать:** если достаточно обычного Markdown.

---

## `/overleaf-sync`

**Для чего:** two-way sync local paper directory ↔ Overleaf Git bridge.

**Когда использовать:**
- работаешь в Overleaf Premium;
- нужна аккуратная sync/pull/push.

**Пример:**
```text
/overleaf-sync status
/overleaf-sync push "Upload compiled paper changes after confirmation"
```

**Не использовать:** без Overleaf Premium/Git bridge.

---

## `/feishu-notify`

**Для чего:** notifications to Feishu/Lark.

**Когда использовать:**
- долгие эксперименты;
- надо получать push notifications.

**Пример:**
```text
/feishu-notify "Send message when the hardcase benchmark finishes"
```

**Не использовать:** без webhook или Feishu setup.

---

## `/interview-cheatsheet`

**Для чего:** long-form Chinese ML/LLM interview prep cheat sheets with formulas, code, Q&A, review, HTML output.

**Когда использовать:**
- нужна подготовка к интервью по ML/LLM;
- формат — китайский cheat sheet.

**Пример:**
```text
/interview-cheatsheet "Transformer attention, LoRA, quantization, and RAG"
```

**Не использовать:** для production engineering tasks.

---

# 10. Мои рекомендуемые shortlists

## Для обычного проекта с багами

```text
/system-profile
/experiment-plan
/research-review
/kill-argument
/ablation-planner
/analyze-results
/experiment-audit
/result-to-claim
/auto-review-loop
/research-wiki
```

## Для проекта с множеством логов и 5–10 багами

```text
/system-profile
/analyze-results
/experiment-plan
/ablation-planner
/experiment-audit
/result-to-claim
/auto-review-loop
/research-wiki
```

## Для архитектуры

```text
/idea-creator
/research-refine
/dse-loop
/kill-argument
/experiment-plan
/ablation-planner
/result-to-claim
```

## Для ML/audio

```text
/research-lit
/research-refine
/experiment-plan
/ablation-planner
/run-experiment
/monitor-experiment
/analyze-results
/experiment-audit
/result-to-claim
/research-wiki
```

## Для paper

```text
/paper-plan
/paper-write
/paper-figure
/paper-claim-audit
/citation-audit
/proof-checker
/paper-compile
/auto-paper-improvement-loop
```

## Для патента

```text
/invention-structuring
/prior-art-search
/patent-novelty-check
/claims-drafting
/specification-writing
/patent-review
/jurisdiction-format
```

---

# 11. Анти-паттерны

## Не делай так

```text
Use all ARIS skills to fix this bug.
```

Почему плохо: агент утонет в routing и будет делать research вместо фикса.

## Лучше так

```text
Use only:
/system-profile
/experiment-plan
/experiment-audit
/result-to-claim
/auto-review-loop

Goal: reproduce and fix one bug. No unrelated rewrites. Max 3 repair rounds.
```

---

# 12. Универсальный prompt-шаблон

```text
Use ARIS-style workflow, but keep scope narrow.

Task:
[описать проблему]

Allowed skills:
[выбрать 3–8 skills из этого файла]

Rules:
1. Reproduce before editing.
2. One patch = one hypothesis.
3. No fallback/sleep/retry/silent catch unless proven architecture-correct.
4. Verify running artifact, not only source code.
5. If a hypothesis fails, write failed-hypothesis note before next patch.
6. Stop after 3 failed rounds and write engineering report.
7. Done only when result-to-claim confirms the exact claim.

Required final report:
- root cause
- changed files
- why this is not a workaround
- verification commands
- before/after evidence
- regression checks
- remaining risks
```

---

# 13. Критерий выбора skill

Задай себе вопрос:

- Нужно найти источники? → `/research-lit`, `/arxiv`, `/semantic-scholar`, `/exa-search`.
- Нужно придумать новые варианты? → `/idea-creator`, `/research-refine`, `/dse-loop`.
- Нужно спланировать проверку? → `/experiment-plan`, `/ablation-planner`.
- Нужно запустить/следить за экспериментом? → `/run-experiment`, `/monitor-experiment`, `/experiment-queue`.
- Нужно разобрать результаты? → `/analyze-results`.
- Нужно проверить честность доказательств? → `/experiment-audit`, `/result-to-claim`.
- Нужно разнести плохое решение? → `/kill-argument`, `/research-review`.
- Нужно улучшать через циклы? → `/auto-review-loop`, `/auto-review-loop-llm`.
- Нужно помнить историю? → `/research-wiki`.
- Нужно писать статью? → paper skills.
- Нужно патентовать? → patent skills.

---

# 14. Главный вывод

ARIS полезен не потому, что “чинит лучше”, а потому что заставляет агента работать как инженер/исследователь:

```text
hypothesis → test → evidence → audit → claim → next action
```

Для твоих проектов главное — не максимальное количество skills, а правильный shortlist под конкретную задачу.
