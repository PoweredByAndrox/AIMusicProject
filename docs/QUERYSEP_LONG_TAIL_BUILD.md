# Querysep long-tail build task

`querysep_long_tail_build` добавляет repeatable queue/runtime contract для long-tail stem-ов:

- `strings_orchestra`
- `synth_lead`
- `synth_pad`
- `arps_plucks`
- `brass_woodwinds`
- `organ_reeds`
- `fx_risers_impacts`

Это не proxy route. Основной `ml_worker_improved` вызывает отдельный `querysep-worker` service и принимает результат только если:

- backend явно `banquet_byoq`;
- `querysep-worker` вернул `status=completed`;
- `metadata.routing_mode=banquet_byoq`;
- `named_outputs[target_stem]` присутствует;
- output-файл виден `ml-worker` через shared volume / `path_mappings`;
- output не меньше `QUERYSEP_MIN_OUTPUT_BYTES` или `min_output_bytes`;
- output проходит `long_tail_acoustic_quality_gate_v1`: duration/sample/finite checks,
  minimum target presence, minimum RMS ratio vs source audio, minimum active-window audibility
  и proxy-copy correlation rejection against available `other` / `melody` / `guitar` / `piano` buses.
  Gate проверяет не только
  full-track correlation, но и active 5-секундные окна, чтобы local leakage не проходил из-за
  низкой global correlation.
- Для `other` bus есть отдельный blocker `possible_other_proxy_leakage`: если output ещё не является
  full proxy copy и не превышает windowed leakage threshold, но всё равно слишком близок к `other`
  по full/windowed correlation при сопоставимом RMS, route отклоняется как подозрительный proxy stem,
  а не считается production `strings_orchestra`/long-tail результатом.

`clap_router`, `auto`, dry-run и отсутствующий shared output считаются ошибками. Task не подставляет `other`, не делает CLAP-routed proxy успешным результатом и не объявляет degraded route baseline.
Если acoustic gate отклоняет output, task пишет `querysep_long_tail_manifest.json` со
`status: "rejected"` и затем падает вместо записи успешного baseline.

После Banquet response результат проходит `long_tail_model_router_v1`. Router оценивает:

- primary `banquet_byoq` output;
- derived `banquet_byoq_harmonic_residual_query` for harmonic long-tail classes when the
  primary Banquet output is rejected. The worker materializes an auxiliary
  `router_auxiliary/<target>/harmonic_residual_query/harmonic_residual.flac` bus from the
  input audio plus available source buses, sends a second Banquet BYOQ request with that
  bus exposed as `build_from_paths.harmonic_residual`, forces
  `raw_step.query_mode="harmonic_residual_single"` for that request, and then applies the
  same `long_tail_acoustic_quality_gate_v1`. The auxiliary bus is never selectable as a stem.
  In this mode `querysep-worker` must build the Banquet query exemplar only from
  `bus_harmonic_residual`; `other` / `guitar` / `piano` / `melody` buses are blocked for
  query selection instead of being mixed into the ensemble.
- derived `sparse_harmonic_residual_candidate` when the harmonic-residual Banquet pass completes
  but fails on target presence (`long_tail_presence_too_weak`,
  `long_tail_active_presence_too_sparse`, or `long_tail_presence_too_weak_vs_source`). This is a
  production candidate copied into `router_candidates/<target>/sparse_harmonic_residual/` from the
  model-guided harmonic residual auxiliary bus only when at least four projection references were
  used. It is not a fallback and not the auxiliary bus itself: it still passes the full
  `long_tail_acoustic_quality_gate_v1`, and windowed proxy decorrelation may create a follow-up
  `windowed_proxy_decorrelated_candidate` from it.
- optional external model candidates из `long_tail_candidate_paths`;
- derived `windowed_proxy_decorrelated_candidate` только если production-eligible model output провалился
  по локальному `windowed_proxy_correlation_too_high`, но не провалился как полный
  `proxy_copy_correlation_too_high`;
- diagnostic bus-sum candidate из доступных `other` / `melody` / `guitar` / `piano`.

Diagnostic bus/proxy candidates пишутся только для объяснения качества и не выбираются как production stem.
`windowed_proxy_decorrelated_candidate` не является fallback: он материализуется из конкретного production
candidate, сохраняет `source_candidate_id`, `source_candidate_type`, `repair_window_count` и `repaired_windows`,
а затем проходит тот же `long_tail_acoustic_quality_gate_v1`. Если после decorrelation gate не пройден,
router обязан отклонить route, а не объявлять degraded output успешным.
`banquet_byoq_harmonic_residual_query` также не является fallback: это второй model-backed Banquet pass с
явным `query_auxiliary_bus` provenance в candidate metadata. Если output слабый, sparse или похож на
`other`/`melody`/`guitar`/`piano` proxy, он получает обычный reject finding и не выбирается.
Если нет production-eligible candidate, прошедшего gate, manifest получает `status: "rejected"`.

## Python worker contract

Handler регистрируется в `ml_worker_improved/worker/main.py` как:

```text
querysep_long_tail_build -> handle_querysep_long_tail_build
```

Production adapter находится в:

```text
ml_worker_improved/worker/querysep_long_tail_build.py
```

Минимальный payload:

```json
{
  "type": "querysep_long_tail_build",
  "task_id": 123,
  "track_id": 15,
  "build_id": "track15-long-tail-20260507",
  "audio_path": "/app/lab_artifacts/run/instrumental_resurrection_minus.flac",
  "source_stem_manifest_path": "/app/lab_artifacts/run/hq_stems/stem_manifest.json",
  "output_dir": "/app/lab_artifacts/run/long_tail_querysep",
  "target_stems": ["strings_orchestra", "brass_woodwinds"],
  "querysep_worker_url": "http://querysep-worker:8000",
  "backend": "banquet_byoq",
  "dry_run": false,
  "long_tail_candidate_paths": {
    "strings_orchestra": {
      "qscnet_strings": "/app/lab_artifacts/run/qscnet/strings_orchestra.flac"
    }
  },
  "path_mappings": [
    {
      "local_prefix": "/app/lab_artifacts",
      "querysep_prefix": "/main_lab_artifacts"
    }
  ]
}
```

Если `querysep_worker_url` не задан в payload, worker читает `QUERYSEP_WORKER_URL`. Если URL отсутствует или недоступен, task падает через `QueueWorker.process_task` и получает `queue_tasks.status=failed`.

## Shared path contract

`querysep-worker` и основной `ml-worker` живут в разных Docker stack/network по текущей архитектуре. Поэтому task требует явный path contract:

- `local_prefix` — путь, видимый основному `ml-worker`;
- `querysep_prefix` — тот же физический каталог, но видимый `querysep-worker`;
- `querysep_output_root` можно переопределить, если output root отличается от `output_dir`.

После ответа `querysep-worker` путь `named_outputs[target]` переводится обратно через `path_mappings`. Если файл не найден локально, task падает. Это защищает от ситуации, где `querysep-worker` успешно записал файл в свой private container path, а основной runtime не может использовать результат.

## Registered `strings_orchestra` candidate route

`ml_worker_improved/tools/build_querysep_strings_orchestra_candidate.py` is the model-registry wrapper for
`querysep_banquet_strings_orchestra_v1`. It builds a production candidate, not a teacher copy:

- input is the full `source_mix.flac`;
- base stems are generated no-reference through `export_track_stems_aggressive_melody_first`;
- `querysep_long_tail_build` is called with `target_stems=["strings_orchestra"]`, `backend="banquet_byoq"`,
  and `dry_run=false`;
- the wrapper records `target_reference_assisted=false`, `target_reference_used_at_application=false`,
  `application_reference_free=true`, `delivery_allowed_without_reference=true`, `fallback_used=false`,
  `per_track_hardcode_allowed=false`, and `not_timestamp_hardcoded=true`;
- output remains `presentation_status=requires_objective_gate` and must not be published as a clean listen
  result until the objective/RALPH gate accepts it;
- default `querysep_output_root` is `null`, so the long-tail builder maps the current output directory through
  `path_mappings`; pass `--querysep-output-root` only for a deliberate shared-volume override.

## Laravel queue helper

`App\Services\MLQueueService::pushQuerysepLongTailBuild()` ставит payload в обычную ML Redis-очередь через `pushPriority`. Тип добавлен в `PRIORITY_TASK_TYPES`, потому что long-tail stem build не должен стоять за фоновым segment backlog.

`App\Models\QueueTask::TYPE_QUERYSEP_LONG_TAIL_BUILD` фиксирует task type для DB/status layer.

## Output manifest

При успехе worker пишет:

```text
<output_dir>/querysep_long_tail_manifest.json
```

Ключевые поля:

- `backend: "banquet_byoq"`
- `fallback_used: false`
- `dry_run: false`
- `source_audio_path`
- `source_stem_manifest_path`
- `build_from_paths`
- `querysep_build_from_paths`
- `stems[target].local_output_path`
- `stems[target].querysep_output_path`
- `stems[target].selected_output_origin`
- `stems[target].selected_output_path`
- `stems[target].primary_querysep_output_path`
- `stems[target].quality_gate`
- `stems[target].router`
- `stems[target].candidates`
- `stems[target].candidates[*].metadata.query_auxiliary_bus` для
  `banquet_byoq_harmonic_residual_query`
- `stems[target].candidates[*].candidate_type: "sparse_harmonic_residual_candidate"` when the
  harmonic residual auxiliary bus is promoted into a gated production candidate after Banquet
  presence failure
- `querysep_results[target].metadata.query_mode`; harmonic residual requests must report
  `harmonic_residual_single`
- при reject: `failure.reason: "long_tail_model_router_rejected_all_candidates"`
- `quality_gate.objective_metrics.sdr: null`
- `quality_gate.objective_metrics.si_sdr: null`
- `querysep_results[target].metadata`

Default acoustic gate thresholds can be overridden per payload or env:

- `min_quality_duration_sec` / `QUERYSEP_LONG_TAIL_MIN_DURATION_SEC`
- `max_duration_delta_sec` / `QUERYSEP_LONG_TAIL_MAX_DURATION_DELTA_SEC`
- `min_quality_rms_dbfs` / `QUERYSEP_LONG_TAIL_MIN_RMS_DBFS`
- `min_source_rms_ratio_db` / `QUERYSEP_LONG_TAIL_MIN_SOURCE_RMS_RATIO_DB`
- `min_active_presence_windows` / `QUERYSEP_LONG_TAIL_MIN_ACTIVE_PRESENCE_WINDOWS`
- `min_active_presence_rms_dbfs` / `QUERYSEP_LONG_TAIL_MIN_ACTIVE_PRESENCE_RMS_DBFS`
- `active_presence_window_sec` / `QUERYSEP_LONG_TAIL_ACTIVE_PRESENCE_WINDOW_SEC`
- `max_proxy_correlation` / `QUERYSEP_LONG_TAIL_MAX_PROXY_CORRELATION`
- `max_windowed_proxy_correlation` / `QUERYSEP_LONG_TAIL_MAX_WINDOWED_PROXY_CORRELATION`
- `windowed_proxy_correlation_sec` / `QUERYSEP_LONG_TAIL_WINDOWED_PROXY_CORRELATION_SEC`
- `min_windowed_proxy_rms_dbfs` / `QUERYSEP_LONG_TAIL_MIN_WINDOWED_PROXY_RMS_DBFS`
- `max_possible_other_proxy_correlation` / `QUERYSEP_LONG_TAIL_MAX_POSSIBLE_OTHER_PROXY_CORRELATION`
- `min_possible_other_proxy_rms_ratio_db` / `QUERYSEP_LONG_TAIL_MIN_POSSIBLE_OTHER_PROXY_RMS_RATIO_DB`

SDR/SI-SDR/SIR/SAR остаются недоступны, пока нет isolated ground-truth stem для exact target class.
Proxy reference sample-rate mismatch считается blocker-ом: gate не пропускает correlation check silently.
Active presence gate требует, чтобы output был слышимым не только по full-track RMS:
по умолчанию минимум `3` active окна по `5.0` секунд с `rms_dbfs >= -50.0`.
Для коротких файлов required active windows уменьшается до количества доступных окон. Это не нормализует
тихий stem, а отклоняет почти пустой result как `long_tail_active_presence_too_sparse`.
Windowed proxy-correlation применяется только к active output windows:
по умолчанию `max_windowed_proxy_correlation=0.55`, `windowed_proxy_correlation_sec=5.0`,
`min_windowed_proxy_rms_dbfs=-58.0`. Это защищает от локальных `other`/`melody`/`guitar`/`piano`
утечек без reject-а почти тихих окон.
`possible_other_proxy_leakage` закрывает промежуточный случай: output может быть ниже full-copy
threshold `max_proxy_correlation=0.85` и ниже windowed threshold `0.55`, но всё ещё слишком похож
на `other` для production long-tail stem. По умолчанию blocker включается при
`max_possible_other_proxy_correlation=0.42` и `output/other rms ratio >= -18.0 dB`.
Если router выбирает external candidate, `querysep_output_path` у selected stem остаётся `null`, потому что
output не был создан `querysep-worker`; исходный Banquet path сохраняется отдельно в
`primary_querysep_output_path`, а фактический выбранный файл фиксируется в `selected_output_path`.

## Docker-only validation

Unit/contract tests:

```powershell
docker compose run --rm --no-deps -T ml-worker pytest tests/test_querysep_long_tail_build.py -q
```

Related regression tests:

```powershell
docker compose run --rm --no-deps -T ml-worker pytest tests/test_querysep_long_tail_build.py tests/test_stem_quality_control.py tests/test_high_quality_stem_export.py -q
```

Real inference still requires a running/reachable `querysep-worker` with Banquet checkpoint:

```powershell
docker compose --profile ml-querysep up -d querysep-worker
docker compose --profile ml-querysep exec -T querysep-worker `
  python -c "import json; from app.banquet_runtime import banquet_ready_state; print(json.dumps(banquet_ready_state(), indent=2))"
```

The root project compose exposes this service behind profile `ml-querysep`. It joins `music-network` as
`querysep-worker` and mounts `./lab_artifacts` at `/main_lab_artifacts`, matching the default
`path_mappings` used by `build_querysep_strings_orchestra_candidate.py`.
