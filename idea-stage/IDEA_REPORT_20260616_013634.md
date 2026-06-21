# ARIS Idea Report: AIMusicProject Source Separation Recovery

**Дата:** 2026-06-16 01:36  
**Skill route:** project-local ARIS `/idea-creator` with local `/novelty-check` and local adversarial review pass  
**Direction:** найти не очередной косметический слой, а проверяемые идеи, которые реально могут вывести систему music source separation из текущего тупика с no-reference routing и shadow-only critic.

## A. Понимание проекта

Проект сейчас является локальной Docker-first системой для music source separation и no-reference evaluation. Основные компоненты:

- autonomous runtime: `ml_worker_improved/tools/run_autonomous_no_reference_pipeline.py`, docs: `ml_worker_improved/docs/AUTONOMOUS_NO_REFERENCE_RUNTIME_README.md`;
- candidate/job wave, score matrices, selector, pair compatibility, preview contracts: docs в `ml_worker_improved/docs/NO_REFERENCE_*`;
- audio error grounding dataset: `ml_worker_improved/tools/build_audio_error_grounding_dataset.py`, docs: `ml_worker_improved/docs/AUDIO_ERROR_GROUNDING_DATASET.md`;
- leak-safe split/eval harness: `ml_worker_improved/tools/build_leak_safe_audio_eval_harness.py`, docs: `ml_worker_improved/docs/LEAK_SAFE_AUDIO_EVAL_HARNESS.md`;
- tensor critic/localizer: `ml_worker_improved/tools/train_tensor_audio_error_critic.py`, `ml_worker_improved/tools/compare_tensor_audio_error_critic_variants.py`, docs: `ml_worker_improved/docs/TENSOR_AUDIO_ERROR_CRITIC.md`;
- generic trainable teacher-mask adapter/fine-tune matrix: `ml_worker_improved/tools/train_generic_audio_teacher_mask_adapter.py`, `ml_worker_improved/tools/run_generic_audio_finetune_matrix.py`, `ml_worker_improved/tools/validate_generic_audio_finetune_holdout.py`;
- reference/teacher/repair utilities already exist: `build_reference_teacher_stem_pack.py`, `build_complex_oracle_mask_teacher.py`, `train_teacher_mask_separator.py`, `build_stem_delta_profile_repair.py`, `teacher_calibrated_instrumental_selector.py`.

Подтверждённый текущий тупик:

- `lab_artifacts/audio_error_critic/tensor_r1_train_ready_loop_001/train_ready_blocker_report.md` фиксирует, что R1 tensor critic не может дать train-ready promotion artifact.
- `compare_tensor_audio_error_critic_variants.py::compare_tensor_audio_error_critic_variants` жёстко пишет `promotion_allowed=False`.
- `compare_tensor_audio_error_critic_variants.py::choose_decision` может дать только `keep_as_specialist_shadow_model`, `mine_next_dataset_slice`, `candidate_rejected_runtime`.
- locked hardcase holdout уже saturated для сравниваемых вариантов: `accuracy=1.0`, `hard_negative_count=0`, `false_reject_count=0`, поэтому текущий hardcase-improvement gate нечем улучшать.

## B. Слабые места

1. **Shadow critic не является repair/separator model.**  
   `train_tensor_audio_error_critic.py` обучает `logistic_regression_balanced` и `random_forest_balanced` по tensor maps. Это классификатор/локализатор, а не модель, которая исправляет вокал или минус.

2. **Decision contract R1 не может дать promotion.**  
   `promotion_allowed=False` в comparator записан независимо от метрик. Повторение R1 training не может стать production-result.

3. **Hardcase holdout metric saturated.**  
   Если holdout уже имеет 1.0 accuracy и 0 hard negatives/false rejects, gate "improve hardcase" становится математически непроходимым.

4. **Данных с проверенными target stems недостаточно формализовано.**  
   Есть playbook `docs/generic_finetuning_playbook/*` и scripts для teacher-mask adapter, но нужен строгий pipeline из user-verified/FL Studio/reference stems в train/eval JSONL.

5. **Много routing/repair scripts, но мало доказательств, что они улучшают cross-track audio quality.**  
   Есть многочисленные `apply_*repair.py`, но для production нужно оценивать их через fixed holdout, а не по одному listen preview.

6. **No-reference metrics помогают блокировать мусор, но не создают clean stems.**  
   `AUDIO_ERROR_GROUNDING_DATASET.md` прямо говорит: mix используется как mixture consistency constraint, не как true source reference.

7. **Subagent external reviewer недоступен в текущем thread.**  
   Попытка `spawn_agent` для ARIS review завершилась `agent thread limit reached`; adversarial pass выполнен локально.

## C. Таблица статусов сырых идей

| # | Идея | Категория | Статус | Причина для REJECT/MERGE/DEFER |
|---|------|-----------|--------|--------------------------------|
| 1 | Перестроить train-ready decision contract поверх существующего R1, где допускается validated train-ready artifact | evaluation | KEEP | |
| 2 | Пересобрать locked hardcase holdout так, чтобы метрики были несатурированными и включали positive/negative hardcases | evaluation | KEEP | |
| 3 | Запустить generic teacher-mask adapter matrix как основной trainable repair path | ML/audio | KEEP | |
| 4 | Построить verified reference corpus из user-approved кейсов и FL Studio stems | data | KEEP | |
| 5 | Сделать oracle upper-bound/gap report перед любым fine-tune | evaluation | KEEP | |
| 6 | Использовать tensor critic как training-data miner, а не как production gate | architecture | KEEP | |
| 7 | Обучать только metadata/controller policy на routing decisions | ML/controller | MERGE | weaker than another idea: полезно, но слабее trainable audio repair и должно быть вторым уровнем после verified corpus |
| 8 | Добавить ещё один no-reference score matrix | evaluation | REJECT | already exists: score matrices уже есть, проблема не в отсутствии ещё одной матрицы |
| 9 | Просто повысить mining weight multiplier | ML | REJECT | low impact: model_003/model_004 показали одинаковый сигнал на тех же rows |
| 10 | Использовать текущий R1 comparator как promotion gate | architecture | REJECT | already exists: он уже существует и явно запрещает promotion |
| 11 | Добавить BS-RoFormer/Mamba2 route как ещё один candidate generator | model routing | DEFER | too expensive: может помочь, но без verified eval/targets снова даст только candidates |
| 12 | Fine-tune separator целиком на всех lab_artifacts | ML/audio | REJECT | high risk: lab artifacts содержат dirty/blocked outputs, нельзя считать их ground truth |
| 13 | Учить repair adapter на accepted windows как positives и blocked as negatives | ML/audio | MERGE | weaker than another idea: это часть verified teacher-mask/corpus route, отдельно слишком расплывчато |
| 14 | Ввести per-track ручные EQ/patch recipes | repair | REJECT | high risk: нарушает production adaptive audio rule и плохо обобщается |
| 15 | Сделать dataset fingerprint/retrain-if-needed state для audio fine-tune | automation | KEEP | |
| 16 | Сделать promotion quarantine: модель может быть "train-ready shadow" без runtime activation | safety | KEEP | |
| 17 | Улучшить observability GPU/CPU phase logs для fine-tune matrix | observability | KEEP | |
| 18 | Свести все successful listen previews в recipe bank без targets | data | DEFER | needs missing data: без verified target stems это только priors, не supervised training |
| 19 | Перейти сразу к melody/music intelligence | analysis | DEFER | low impact: не решает качество stems до появления clean stems |
| 20 | Сделать multi-objective holdout: SDR/improvement/bleed/phase/consistency вместо одного hardcase decision | evaluation | KEEP | |
| 21 | Автоматически генерировать synthetic mixtures из clean stems для supervised training | data | KEEP | |
| 22 | Сделать active-learning queue для user verification только спорных windows | data/UX | KEEP | |
| 23 | Переписать все repair scripts в один большой pipeline | architecture | REJECT | high risk: broad refactor без доказанной необходимости |
| 24 | Использовать external closed API separation как reference | data | REJECT | not testable: нельзя считать закрытый output ground truth без verification |
| 25 | Добавить regression dashboard для artifacts и gates | observability | MERGE | duplicate: dashboard/control-plane уже частично есть, нужно объединить с train-ready manifest |
| 26 | Делать release_ready после best available candidate | delivery | REJECT | high risk: против RALPH-AI contract, dirty candidates нельзя выдавать как result |

## D. Scoring matrix серьёзных идей

Формула: `Total = Impact*0.30 + Feasibility*0.20 + Novelty*0.15 + Testability*0.20 - Risk*0.10 - TestCost*0.05`.

| Идея | Impact | Feasibility | Novelty | Testability | Risk | TestCost | Total |
|------|--------|-------------|---------|-------------|------|----------|-------|
| Несатурированный locked eval + train-ready decision contract | 9 | 9 | 5 | 10 | 4 | 4 | 6.65 |
| Teacher-mask adapter matrix как основной trainable repair path | 10 | 8 | 6 | 9 | 5 | 6 | 6.50 |
| Oracle upper-bound/gap report перед training | 8 | 8 | 6 | 9 | 3 | 4 | 6.20 |
| Verified reference corpus из user-approved/FL Studio stems | 10 | 7 | 5 | 8 | 5 | 6 | 5.95 |
| Multi-objective holdout: SDR/bleed/phase/consistency | 8 | 8 | 6 | 10 | 4 | 4 | 6.10 |
| Lab-artifact recipe mining только как priors, не labels | 8 | 8 | 5 | 8 | 4 | 4 | 5.75 |
| Supervised two-stage repair adapter for vocal/minus | 10 | 6 | 7 | 8 | 7 | 8 | 5.75 |
| Dataset fingerprint + retrain-if-needed state | 7 | 9 | 4 | 9 | 3 | 3 | 5.60 |
| Active-learning queue для user verification | 8 | 7 | 5 | 8 | 4 | 5 | 5.55 |
| Trainable controller policy from verified outcomes | 7 | 7 | 6 | 7 | 5 | 5 | 5.05 |
| Добавить новый separator family route | 9 | 5 | 4 | 7 | 7 | 8 | 4.60 |
| Music intelligence layer до clean stems | 6 | 7 | 5 | 6 | 4 | 5 | 4.50 |

## E. Проверка evidence / gap по проекту

| Идея | already implemented | Evidence | Exact missing gap |
|------|---------------------|----------|-------------------|
| Несатурированный locked eval + train-ready decision contract | partial | `build_leak_safe_audio_eval_harness.py`, `compare_tensor_audio_error_critic_variants.py`, `LEAK_SAFE_AUDIO_EVAL_HARNESS.md`, `TENSOR_AUDIO_ERROR_CRITIC.md` | текущий comparator не может promotion; hardcase metric saturated |
| Teacher-mask adapter matrix | partial | `train_generic_audio_teacher_mask_adapter.py`, `run_generic_audio_finetune_matrix.py`, `validate_generic_audio_finetune_holdout.py`, tests `test_run_generic_audio_finetune_matrix.py`, `test_validate_generic_audio_finetune_holdout.py` | нужен реальный verified audio_training_examples_jsonl и full Docker/GPU matrix run на project corpus |
| Oracle upper-bound/gap report | partial | `oracle_stem_upper_bound.py`, `build_complex_oracle_mask_teacher.py`, tests `test_oracle_stem_upper_bound.py`, `test_build_complex_oracle_mask_teacher.py` | нет обязательного pre-training gate, который говорит "эта trainable path вообще может улучшить target" |
| Verified reference corpus | partial | `build_flstudio_ground_truth_bundle.py`, `build_reference_teacher_stem_pack.py`, `docs/generic_finetuning_playbook/*` | нет единого accepted/user-verified corpus manifest, отделяющего true target stems от dirty candidates |
| Multi-objective holdout | partial | `validate_generic_audio_finetune_holdout.py` считает SDR/improvement; tensor critic считает bleed/phase proxies | нет единого holdout decision, объединяющего waveform metrics и no-reference blockers |
| Recipe priors from lab_artifacts | partial | `build_no_reference_corpus_route_priors.py`, `generate_no_reference_failure_reflections.py`, harness docs | нужно запретить превращать priors в labels без verification |
| Supervised repair adapter | partial | `train_teacher_mask_separator.py`, `build_stem_delta_profile_repair.py`, `build_reference_guided_stem_residual_repair.py` | нужен stable train/eval corpus и promotion quarantine |
| Dataset fingerprint/retrain state | partial | generic playbook описывает pattern; adapter fingerprint exists in `train_generic_audio_teacher_mask_adapter.py` | нет общего state file для "skip/retrain/promote" на audio fine-tune loop |
| Active-learning queue | no | доказательство не найдено как отдельный reviewed queue для user verification | нужен UI/manifest для ручной проверки только спорных windows |
| Controller policy | partial | `run_no_reference_learning_cycle.py`, `promote_no_reference_critic_model.py`, runtime docs | verified outcomes слишком ограничены, нельзя учить из raw predictions |

## F. Novelty / prior-art check

Сравнение с известными подходами:

- HTDemucs показывает, что extra training songs и per-source fine-tuning могут улучшать SDR; это делает "trainable audio path" не новой идеей, но подтверждает направление как технически нормальное. Source: [Hybrid Transformers for Music Source Separation](https://arxiv.org/abs/2211.08553).
- BSRNN описывает band-split modeling и semi-supervised fine-tuning; значит "ещё один separator route" не является новой идеей сам по себе. Source: [Music Source Separation with Band-split RNN](https://arxiv.org/abs/2209.15174).
- BS-RoFormer/Mel-RoFormer показывают сильные frequency-domain architectures; добавление их как route потенциально полезно, но не решает отсутствие verified eval. Sources: [BS-RoFormer](https://arxiv.org/abs/2309.02612), [Mel-Band RoFormer](https://arxiv.org/abs/2310.01809).
- Domain adaptation для source separation уже исследовалась; novelty проекта не в "domain adaptation exists", а в строгой artifact/provenance/holdout discipline для локального corpus. Source: [Adversarial Unsupervised Domain Adaptation for Harmonic-Percussive Source Separation](https://arxiv.org/abs/2101.00701).
- Recent restoration/separation systems split separation and restoration; это поддерживает идею two-stage repair, но снижает novelty. Source: [Multi-Stage Music Source Restoration with BandSplit-RoFormer](https://arxiv.org/html/2603.04032v1).

Novelty risk:

| Идея | Related known method/tool/paper | What already exists | What is different here | Novelty risk |
|------|---------------------------------|---------------------|------------------------|--------------|
| Несатурированный eval + train-ready decision | standard ML holdout/promotion gates | common in ML systems | локальный artifact contract, no-reference + reference-safe gates, explicit anti-leakage | low |
| Teacher-mask adapter matrix | teacher/student masks, separator fine-tuning | common idea | project already has concrete GPU matrix + cross-track holdout, gap is execution with verified corpus | medium |
| Verified reference corpus | supervised MSS datasets | common | user/project-specific verified stems with provenance from lab artifacts/FL Studio | low |
| Oracle gap report | oracle masks/upper bounds | common | required precondition before spending GPU on local repair path | low |
| Two-stage repair adapter | source restoration, residual repair | known | specialized to detected local blocker taxonomy and artifact contracts | medium |

## G. Adversarial reviewer pass

External subagent reviewer was attempted but blocked by `agent thread limit reached`. Local adversarial pass:

| Идея | Strongest argument against it | How it can fail | Overfit/mislead risk | Evidence that would kill it | Decision |
|------|-------------------------------|-----------------|----------------------|-----------------------------|----------|
| Несатурированный locked eval + train-ready decision | Это само по себе не улучшает audio | Можно сделать красивые gates и всё равно не получить better stems | Можно "подогнать" holdout под текущие candidates | Новый holdout показывает, что all candidates fail equally и нет trainable signal | KEEP |
| Teacher-mask adapter matrix | Может выучить artifact-specific masks и не generalize | Cross-track holdout failed или SDR improves but bleed worsens | Высокий, если target stems грязные или same-track leakage | Holdout pass rate < threshold на verified corpus | KEEP |
| Verified reference corpus | Самая трудоёмкая часть, требует чистых target stems | Недостаточно треков/стемов, labels inconsistent | Пользовательский listen approval может быть субъективным | Corpus audit finds no true targets or too few unique tracks | KEEP |
| Oracle upper-bound/gap report | Может стать ещё одним report без действия | Oracle says target path has no headroom | Низкий, если reference stems real | Oracle upper bound не превосходит текущий best candidate | KEEP |
| Multi-objective holdout | Сложнее интерпретировать, может конфликтовать | Metrics disagree and no clear decision rule | Можно выбрать удобную метрику post-hoc | Pre-registered decision rule fails to correlate with human checks | REWRITE |
| Controller policy | Не исправляет audio, только выбирает | Learns historical routing bias | High if trained from raw predictions | Offline replay improves but real runtime not improved | REWRITE |
| New separator route | Может дать immediate quality jump | Install/runtime/GPU cost, no verified promotion | High if best-of-bad is accepted | New route still blocked on same hardcases | DEFER |
| Music intelligence | Нужны clean stems first | Анализирует грязные stems как музыку | Very high until clean stems exist | Manifest has no analysis_allowed clean stems | REJECT now |

## H. Финальный рейтинг

1. **Несатурированный locked eval + train-ready decision contract.**  
   Почему: текущий gate доказанно непроходим. Без исправления eval contract любой training будет "ещё один shadow artifact".

2. **Verified reference corpus + user-approved/FL Studio target stems.**  
   Почему: supervised audio repair/fine-tune без настоящих targets невозможен; lab_artifacts нельзя массово считать truth.

3. **Teacher-mask adapter/fine-tune matrix на verified corpus.**  
   Почему: это уже существующий trainable path, который реально меняет audio output, имеет GPU execution и cross-track holdout validation.

4. Oracle upper-bound/gap report.
5. Multi-objective holdout metrics.
6. Dataset fingerprint/retrain-if-needed state.
7. Active-learning user verification queue.
8. Lab-artifact priors for routing, not labels.
9. Trainable controller policy.
10. New separator family route.

## I. Планы реализации TOP-3

### 1. Несатурированный locked eval + train-ready decision contract

Exact implementation plan:

1. Создать новый decision route, не менять R1F/R1P задним числом: `tools/compare_train_ready_audio_candidates.py` или расширить отдельным `tools/validate_train_ready_audio_model.py`.
2. Входы: `leak_safe_eval_001`, `generic_audio_finetune_holdout_manifest.json`, tensor critic score summaries, no-reference blockers.
3. Decision values: `train_ready_candidate`, `shadow_only_candidate`, `needs_more_verified_data`, `rejected_safety`.
4. Убрать saturated-only condition: gate должен требовать no regression на hardcase negative rows и improvement на at least one non-saturated metric: cross-track SDR/improvement, blocker reduction, false accept stable.
5. Добавить tests рядом с `tests/test_compare_tensor_audio_error_critic_variants.py`.

Likely affected files/areas:

- `ml_worker_improved/tools/compare_tensor_audio_error_critic_variants.py` только как reference, не править его contract;
- new tool under `ml_worker_improved/tools/`;
- `ml_worker_improved/tests/test_*train_ready*`;
- docs: `ml_worker_improved/docs/TENSOR_AUDIO_ERROR_CRITIC.md`, maybe new `TRAIN_READY_AUDIO_MODEL_DECISION.md`.

Minimum experiment/test:

- Mock baseline with saturated hardcase but failed SDR holdout.
- Candidate with same hardcase safety and improved SDR/improvement.
- Validator must output `train_ready_candidate` only when release/runtime flags stay false and leakage is zero.

Success metrics:

- `critical_leakage_count=0`;
- no `release_ready=true`;
- `false_accept_rate` not regressed;
- holdout improvement above threshold on cross-track verified stems;
- decision artifact allows train-ready quarantine without runtime activation.

Failure criteria:

- any dev/test training contamination;
- decision can pass with saturated hardcase only and no positive improvement metric;
- decision implies runtime gate update automatically.

Validation/anti-overfitting guard:

- pre-register metrics and thresholds in manifest;
- split by track_id/audio fingerprint;
- require at least N unique holdout tracks.

### 2. Verified reference corpus + user-approved/FL Studio target stems

Exact implementation plan:

1. Build a corpus manifest that separates true targets from candidates: `verified_reference_corpus_manifest.json`.
2. Inputs: `D:/AIMusicProject/testing_stuff_music`, FL Studio exports, user-approved listen packs, existing `build_flstudio_ground_truth_bundle.py`, `build_reference_teacher_stem_pack.py`.
3. Each row must include `track_id`, `stem_type`, `source_mix_path`, `target_stem_path`, `verification_source`, `verification_status`, `train/dev/test split`, audio fingerprint, and `safe_for_training`.
4. Explicitly reject `lab_artifacts` candidates unless marked user-verified or exact reference.
5. Generate `audio_training_examples.jsonl` for teacher-mask adapter.

Likely affected files/areas:

- `ml_worker_improved/tools/build_flstudio_ground_truth_bundle.py`;
- `ml_worker_improved/tools/build_reference_teacher_stem_pack.py`;
- possible new builder `build_verified_audio_training_examples.py`;
- tests for corpus provenance and leakage;
- docs under `docs/generic_finetuning_playbook/` and `ml_worker_improved/docs/`.

Minimum experiment/test:

- Build small corpus with 2-3 tracks and vocal/minus targets.
- Validate no candidate output is accepted as ground truth without explicit verification.
- Split by `track_id` and audio fingerprint.

Success metrics:

- `safe_for_training_count >= minimum`;
- no missing audio;
- no overlap train/holdout by track/fingerprint;
- every target has provenance and verification status.

Failure criteria:

- generated/candidate stems enter training as truth;
- same track leaks into train and holdout;
- missing target stems.

Validation/anti-overfitting guard:

- leakage validator similar to EVAL-1;
- corpus hash/fingerprint;
- append-only verification log.

### 3. Teacher-mask adapter/fine-tune matrix on verified corpus

Exact implementation plan:

1. Use `run_generic_audio_finetune_matrix.py` with `--device cuda --require-gpu true` on `audio_training_examples.jsonl`.
2. Start with small config matrix: baseline adapter, higher activity weight, leakage suppression, limited source-feature memory only as diagnostic.
3. Validate with `validate_generic_audio_finetune_holdout.py` using cross-track leave-one-checkpoint-out.
4. Write matrix decision artifact: `passed` means train-ready quarantine, not release-ready.
5. Only after holdout passes, evaluate on real runtime hardcases as shadow repair candidate.

Likely affected files/areas:

- `ml_worker_improved/tools/run_generic_audio_finetune_matrix.py`;
- `ml_worker_improved/tools/train_generic_audio_teacher_mask_adapter.py`;
- `ml_worker_improved/tools/validate_generic_audio_finetune_holdout.py`;
- tests already exist: `test_run_generic_audio_finetune_matrix.py`, `test_train_generic_audio_teacher_mask_adapter.py`, `test_validate_generic_audio_finetune_holdout.py`.

Minimum experiment/test:

```bash
docker compose run --rm ml-worker python tools/run_generic_audio_finetune_matrix.py \
  --audio-training-examples-jsonl /app/lab_artifacts/verified_audio_training/audio_training_examples.jsonl \
  --config-matrix-json /app/lab_artifacts/verified_audio_training/teacher_mask_matrix.json \
  --output-dir /app/lab_artifacts/generic_audio_finetune/teacher_mask_matrix_001 \
  --stem vocal \
  --scope verified_vocal_teacher_mask \
  --device cuda \
  --require-gpu true
```

Success metrics:

- `matrix_validation_status=passed`;
- `passed_config_count >= 1`;
- `holdout_validation_status=passed`;
- `min_candidate_sdr_db >= threshold`;
- `improvement_db >= threshold`;
- `release_ready=false`, `delivery_allowed=false` until separate promotion.

Failure criteria:

- GPU unavailable and route continues silently on CPU;
- holdout fails;
- train/holdout track overlap;
- preview quality improves SDR but worsens blockers.

Validation/anti-overfitting guard:

- holdout track IDs excluded at training;
- max examples per track in matrix;
- require independent hardcase replay after holdout pass;
- no source-feature memory generalization claim unless validated.

## Read files and evidence used

Code Index:

- `code-index-mcp stats`: 1459 files, 9500 functions, 389 classes, 103949 calls.
- `code-index-mcp query run_autonomous_no_reference_pipeline`.
- `code-index-mcp query compare_tensor_audio_error_critic_variants`.
- `code-index-mcp query build_audio_error_grounding_dataset`.

CodeGraph:

- `compare_tensor_audio_error_critic_variants.py::choose_decision`.
- `compare_tensor_audio_error_critic_variants.py::compare_tensor_audio_error_critic_variants`.
- `train_tensor_audio_error_critic.py::train_tensor_audio_error_critic`.
- `build_audio_error_grounding_dataset.py::build_audio_error_grounding_dataset`.
- `train_teacher_mask_separator.py::train_teacher_mask_dataset`.
- `train_generic_audio_teacher_mask_adapter.py::train_generic_audio_teacher_mask_adapter`.
- `run_generic_audio_finetune_matrix.py::run_generic_audio_finetune_matrix`.
- `validate_generic_audio_finetune_holdout.py::validate_generic_audio_finetune_holdout`.

Docs inspected:

- `ml_worker_improved/docs/TENSOR_AUDIO_ERROR_CRITIC.md`.
- `ml_worker_improved/docs/AUDIO_ERROR_GROUNDING_DATASET.md`.
- `ml_worker_improved/docs/AUTONOMOUS_NO_REFERENCE_RUNTIME_README.md`.
- `docs/generic_finetuning_playbook/README.md`.
- `docs/generic_finetuning_playbook/music_stem_separation_playbook.md`.

Tests inspected:

- `ml_worker_improved/tests/test_run_generic_audio_finetune_matrix.py`.
- `ml_worker_improved/tests/test_validate_generic_audio_finetune_holdout.py`.
- `ml_worker_improved/tests/test_train_generic_audio_teacher_mask_adapter.py`.
- plus test inventory for tensor critic, grounding, teacher, repair, finetune validators.

Artifacts inspected:

- `lab_artifacts/audio_error_critic/tensor_r1_train_ready_loop_001/train_ready_blocker_report.md`.
- `lab_artifacts/audio_error_grounding/leak_safe_eval_001/leak_safe_split_manifest.json`.
- `lab_artifacts/audio_error_grounding/leak_safe_eval_001/hardcase_holdout_manifest.json`.
- `lab_artifacts/audio_error_critic/tensor_r1_dataset_001/locked_tensor_audio_error_dataset_manifest.json`.
- `lab_artifacts/audio_error_critic/tensor_r1_model_003_source_b_grounded_decision/tensor_audio_error_critic_decision_manifest.json`.
- `lab_artifacts/audio_error_critic/tensor_r1_train_ready_loop_001/r1v_train_only_balanced_bundle_from_model_004/tensor_audio_error_train_only_mining_bundle_manifest.json`.

Drift found:

- `r1p_model_004_vs_model_003_decision` was not found under `tensor_r1_train_ready_loop_001`; it exists under the earlier hard-negative harvest root according to previous command output. Treat the train-ready loop root as incomplete for that decision artifact unless copied/backfilled.

