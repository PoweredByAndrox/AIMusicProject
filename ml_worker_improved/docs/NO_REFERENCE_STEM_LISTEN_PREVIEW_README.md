# No-Reference Stem Listen Preview

`tools/build_no_reference_stem_listen_preview_manifest.py` promotes a generated
non-vocal stem candidate into a listen-preview-only manifest.

Supported target stems:

- `melody`
- `drums`
- `bass`
- `other`

This route deliberately blocks `vocal`, `minus`, and `both` so it cannot bypass
their dedicated mode-specific delivery contracts.

## Contract

- input is an existing `stem_manifest.json` from the Docker job-matrix
  candidate-generation route;
- for multistem model exports, `--target-stem` may select any supported
  non-vocal entry already present in `stems` even when the source manifest's
  own `target_stem` was produced for another stem;
- the selected target audio path must exist and be readable inside Docker;
- audio integrity gate recomputes duration, RMS, and peak from the file;
- GPU provenance is required by default:
  `gpu_inference_required=true`, `gpu_inference_skipped=false`, and
  `cuda_preflight.cuda_available=true`;
- source manifests that already claim delivery/release readiness are blocked;
- source manifests with fallback, hidden inference, per-track hardcode, safety
  violations, training, fine-tune, model registry edits, cleanup routes, copied
  audio, exported audio, threshold relaxation, or release readiness are blocked;
- passed output is semantic-judge input only:
  `delivery_status=pending_semantic_judge`, `delivery_allowed=false`,
  `presentation_status=semantic_judge_input_not_for_listen_pack`,
  `release_status=semantic_judge_input_only`, and `semantic_judge_input=true`;
- passed output still sets `semantic_quality_verified=false` and
  `release_ready=false`, so it is not user-presentable until a separate
  semantic stem-quality judge clears it.

This contract is not a semantic stem-quality proof and must not be presented as
`release_ready` or as a user listen/result artifact.

## Short-Name Stem Audio Pack

When a registered model `stem_manifest.json` already contains generated full
6s-model stem audio, use `tools/build_stem_manifest_audio_pack.py` to create
real listenable stem files with short Windows-safe names. The current production
default set is seven files:

- `audio/t001/vocal.flac`
- `audio/t001/minus.flac`
- `audio/t001/drums.flac`
- `audio/t001/bass.flac`
- `audio/t001/other.flac`
- `audio/t001/guitar.flac`
- `audio/t001/piano.flac`

The packer does not run inference, train, download, edit registries, relax
thresholds, or mark release-ready. It copies only existing source stem audio and
stores the raw track id, model id, source stem key, source path, sha256, and
duration in the manifest.

If a candidate wave writes a normalized audio artifact that is the actual
accepted candidate path, pass it explicitly with `--override-stem-path
stem=/app/path/file.flac`. The packer still requires the registered
`stem_manifest.json`, copies the override without modification, writes
`audio_override_applied=true`, preserves the original `source_manifest_audio_path`,
and keeps the override path in manifest provenance instead of the filename.

If the registered manifest was produced on a remote Mac and still contains
Mac-local source paths, use `--path-prefix-map FROM=TO` instead of per-stem
overrides. The packer resolves the source path into the mounted/local artifact
mirror before checking or copying audio, while `source_manifest_audio_path` keeps
the original path and `source_audio_path_resolution` records the mapping.

The generated pack uses short metadata names:

- `pack.json`
- `selected.jsonl`
- `progress.jsonl`

`tools/build_machine_stem_acceptance_gate.py` is fail-closed for final machine
acceptance: by default it requires the full production stem set
`vocal, minus, drums, bass, other, guitar, piano`. A pack containing only
`vocal` and `minus` remains a diagnostic/listen-preview artifact and is blocked
with `required_stem_missing:<stem>` reasons.

Full coverage is necessary but not sufficient. Required non-minus stems must
also carry machine semantic-quality evidence; raw listen-preview rows with
`semantic_quality_verified=false` remain blocked with
`required_stem_quality_not_verified:<stem>`. The `minus` stem is covered by the
separate minus quality audit.

All new listen/review/feedback/repair artifacts must use short Windows-safe
file and directory segments. Raw track titles, raw candidate ids, full route
ids, and full hashes stay in manifests only. Review and feedback outputs use
compact metadata names such as `tpl.json`, `items.jsonl`, `handoff.json`,
`playlist.m3u`, `checklist.md`, `validation.json`, `feedback.json`,
`queue.json`, `rows.jsonl`, `packet.json`, `cycle.json`, `summary.md`, and
`progress.jsonl`.
Short artifact paths must be contained relative paths: `.` / `..` traversal
segments, absolute paths, Windows reserved device names such as `CON` or
`COM1`, overlong path segments, and Windows-unsafe characters are rejected.

To combine already accepted vocal/minus/stem listen packs into one
user-facing folder, use `tools/build_aggregate_short_listen_pack.py`. It copies
only existing audio from source manifests, writes `pack.json`, `coverage.json`,
`selected.jsonl`, `progress.jsonl`, and `README.md`, and blocks the pack if any
source manifest or row is release-ready, publishable, threshold-relaxed, missing
audio, or uses a non-short alias/path segment.

If a source manifest or CLI argument declares required instrument stems, aggregate
packaging also fails closed until those present required stems carry machine
semantic-quality evidence. Unverified required instrument rows are blocked with
`required_stem_quality_not_verified:<stem>` by both the builder and validator.
The partial vocal/minus preview flow stays preview-only unless a caller declares
full required-stem coverage.

The aggregate pack also accepts an accepted no-reference vocal/minus master
manifest that exposes `accepted_listen_preview_rows` with `selected_audio_path`.
Those rows are copied without modification using short stem filenames such as
`audio/mst/vocal.flac` and `audio/mst/minus.flac`; raw route filenames remain in
`pack.json` provenance only.

Validate the aggregate pack with `tools/validate_aggregate_short_listen_pack.py`
before handing it off for listening. The validator rechecks the short-name
policy, source/listen contracts, missing audio, copied audio sha256 and audio
metadata, coverage rows, promised `selected.jsonl` / `progress.jsonl` /
`README.md` artifacts, release/publish flags, and writes
`validation/validation.json` plus `validation/progress.jsonl`.

After validation, use `tools/audit_short_listen_goal_status.py` to write a
machine-readable goal-status scorecard. This audit is deliberately not a listen
verdict: it separates machine-proved packaging/coverage/safety checks from the
remaining human listening blocker, and keeps `release_ready=false` and
`can_publish_as_result=false`. Six-stem/6s packs may prove coverage, but they
remain `semantic_quality_verified=false` unless a separate semantic stem-quality
gate proves otherwise.

When listening rejects a short pack item, first write item-level rows with the
short `audio_path`, `verdict`, `issue_tags`, and optional note, then build
`feedback.json` with `tools/build_short_listen_feedback_manifest.py`.
The feedback manifest copies provenance from `pack.json`, rejects unknown audio
paths, and remains non-deliverable. Passing that manifest back into
`tools/audit_short_listen_goal_status.py` changes the goal status to
`human_listen_rejected`, so repair/mining can target exact short-path items
instead of ambiguous long filenames.

Use `tools/build_short_listen_feedback_template.py` to create the item-level
review rows directly from a validated aggregate pack. It writes `items.jsonl`
with every short `audio_path`, empty `verdict`, and stem-specific suggested
issue tags. The template manifest is `tpl.json`; it is deliberately not a
human verdict and sets `ready_for_feedback_manifest=false`; fill each row with
`accepted` or `rejected` before passing it to
`tools/build_short_listen_feedback_manifest.py`. The template route also
accepts an audit-gated `short_listen_subset_pack_manifest.v1`, using its
`listen_audio_files` rows and `artifact_relative_path` values. It also accepts
`short_listen_repair_selection_pack_manifest.v1` so accepted repair-selection
candidates can get their own A/B listen handoff without being merged into the
main aggregate pack first.

Use `tools/build_short_listen_review_handoff.py` after the feedback template to
write a review-only handoff folder with `playlist.m3u`, `checklist.md`, and
`handoff.json`. The playlist uses
relative short paths from the handoff folder to the selected pack audio files.
This keeps `playlist.m3u` openable after the handoff is copied next to the pack,
without embedding raw track names, candidate ids, or long hashes in file names.
The checklist keeps candidate provenance out of visible file names and points
reviewers back to the feedback template for the actual verdict rows.

Validate the review handoff with
`tools/validate_short_listen_review_handoff.py`. It checks that the handoff
manifest, playlist, checklist, aggregate pack, and feedback template all agree
on the same short audio paths and pack id, that every playlist entry resolves to
an existing audio file from the handoff folder, and that no release/publish/
training flags were introduced. It also rejects handoff item-level release/
publish flags, playlist targets outside the source pack root, and `#EXTINF`
labels that leak raw ids or long hashes. For audit-gated subsets, the validator
compares the handoff against `listen_audio_files` instead of aggregate `items`.

Use `tools/run_short_listen_review_packet.py` when the aggregate pack is ready
and the goal is to produce the complete review handoff from one Docker command.
It runs the feedback template, review handoff, handoff validation, and goal
audit stages in order. It stops fail-closed if any stage reports failed checks
and still keeps `release_ready=false`.

After the review rows are filled, use
`tools/run_short_listen_feedback_cycle.py` to run the feedback manifest, goal
audit, and repair-queue planning from one Docker command. It only builds a
repair queue when the human verdict is rejected; accepted full-pack feedback
does not start repair. This command does not repair audio, train, publish, or
mark release-ready. The feedback JSONL reader accepts Windows-authored UTF-8
files with BOM so PowerShell/Notepad-saved verdict rows do not break the cycle.
The cycle manifest is `cycle.json`.

Use `tools/build_short_listen_repair_queue.py` after rejected feedback to create
a non-deliverable repair/mining queue. It maps rejected short audio paths to
route intents such as `minus_repair` and `vocal_candidate_replacement`,
preserves source provenance and judge evidence, and still does not train,
repair, publish, or mark release-ready. Explicit `stem_type` from the feedback
manifest is authoritative for mode routing: a rejected vocal item remains a
vocal replacement request even when its issue tags overlap with minus bleed
diagnostics.
The repair queue writes `queue.json`, `rows.jsonl`, `summary.md`, and
`progress.jsonl`.

Use `tools/build_short_listen_repair_candidate_pack.py` to turn that queue into
a short-name candidate listen pack. For vocal replacement requests it mines
existing candidate-conditioned vocal sources for the same track and records
exact-reference metrics when verified targets exist. For minus repair requests
it packages existing no-reference minus sweep candidates from the source render
manifest, full-track switch lineage, stem audio pack lineage, or phase-repair
selection lineage. Phase-repair lineage reads the original source minus plus
`variants[].path` from `no_reference_minus_phase_repair_manifest.v1`, then
writes `minus_judge_candidates.jsonl` for follow-up no-reference minus ranking.
It does not run inference, train, change models, relax thresholds, repair audio
in-place, publish, or mark release-ready.

After both minus judges run, use
`tools/build_short_listen_repair_selection_pack.py` to build a concise
short-name repair selection pack. The selector keeps only source rows with
`delivery_status=accepted` and `delivery_allowed=true`, excludes any
release/publish-ready or blocked rows, ranks vocal replacements by verified
reference SDR when available, and ranks minus replacements only when the
candidate passed both result and windowed no-reference minus judges. Any judge
warning reasons are preserved as `quality_warning_reasons`. The output remains
listen-preview only: it is not semantic proof, not a repaired stem, not
publishable, and not `release_ready`.

When the selection pack will be consumed by
`tools/build_short_listen_replacement_pack.py`, use
`--dedupe-minus-by-track` to keep at most one judge-ranked minus replacement per
source track identity, and use repeated `--exclude-minus-reason <reason>` flags
for known-bad machine warning families such as `silence_hallucination`. The
repeated `--include-minus-source-key <key>` flag narrows a diagnostic selection
to one provenance-safe source such as `track:trk008:minus`, which prevents
window comparison or segment switching from mixing candidates across different
tracks. The
replacement pack still matches by specific source-path provenance first and
uses track id only as a fallback, so one replacement candidate cannot silently
overwrite unrelated clean minus rows that merely share a track id.

Use `tools/build_short_minus_replacement_sweep.py` when the idea funnel has
identified accepted-with-warning minus rows. The sweep tries each accepted
repair candidate as a short-name replacement, runs the same minus quality audit,
and writes `sweep.json`, `rows.jsonl`, `summary.md`, and `progress.jsonl`.
It is an idea-funnel outcome artifact, not a listen result: it reuses existing
candidates/judges, records `gpu_inference_skipped=true`, blocks release-ready
inputs, and only promotes a replacement for review when warnings or blockers
improve without changing runtime gates.

Phase-repair selection is stricter than the raw no-reference judge decision:
pair-incompatible rows or rows with warning reasons such as
`phase_or_stereo_inconsistency` are excluded from replacement candidates.

`tools/build_minus_cmp.py` can compare diagnostic mid-window clips against
either `short_listen_repair_selection_pack_manifest.v1` or
`short_candidate_wave_listen_pack_manifest.v1`. This lets the same comparison
gate evaluate registered-model candidate-wave outputs without converting them
into repair-selection artifacts or mixing candidates across unrelated tracks.

For fresh candidate-wave outputs that already have `candidate_pool.jsonl` plus
both no-reference minus judges, use
`tools/build_short_candidate_wave_listen_pack.py` to produce a short A/B listen
pack without manually converting JSON. It selects only complete-pair minus rows
accepted by both the result judge and windowed judge, copies their existing
minus audio to short paths such as `audio/min/m001.flac`, writes blocked or
non-selected candidates to `blocked.jsonl`, and stores raw track ids,
candidate ids, route ids, source paths, judge rows, sha256, and duration in
`pack.json` / `selected.jsonl`. The route is packaging-only:
`gpu_inference_skipped=true`, no inference/training/model changes are run, and
`release_ready=false` / `can_publish_as_result=false` remain forced false. The
resulting `short_candidate_wave_listen_pack_manifest.v1` is accepted by the
feedback template, review handoff, and handoff validator routes.

For a minus hard-case that already has failing middle/warning windows, first
run `tools/build_minus_cmp.py` against the candidate-wave pack. Then rebuild the
candidate-wave pack with `--minus-warning-comparison-json`. This opt-in gate
fails closed when the same-window comparison has no improved warning window,
has any regressed warning window, has failed checks, or tries to set release or
publish flags. In that case `delivery_status=blocked`, `selected_count=0`, and
no `audio/min/*.flac` files are kept. This prevents a globally accepted minus
candidate from being handed off when it still fails the exact windows that
motivated the repair cycle.

If a selected minus is blocked only by pair phase/stereo consistency, use
`tools/run_no_reference_minus_phase_repair.py` as a diagnostic-only repair
candidate generator, then gate the variants with both
`tools/run_no_reference_result_judge.py` and
`tools/run_no_reference_windowed_judge.py`. Package only variants accepted by
both judges with `tools/build_short_minus_phase_repair_selection_pack.py`.
The packer copies the accepted audio to short paths such as
`audio/min/m001.flac`, stores raw candidate ids and source paths only in
`pack.json` / `selected.jsonl`, and keeps `release_ready=false` and
`can_publish_as_result=false`.

Use `tools/build_short_minus_quality_audit.py` on an aggregate short listen
pack when selected minus items need an explicit residual-vocal / muffling
audit before the final listen verdict. It audits only `stem_type=minus` rows,
reuses existing result/windowed judge JSONL when already recorded in source
provenance, or runs the existing no-reference minus judges when mix+vocal+minus
evidence is available. It writes short metadata names only:
`audit.json`, `rows.jsonl`, `rj.jsonl`, `wj.jsonl`, `cache.jsonl`,
`summary.md`, and `progress.jsonl`. The audit is diagnostic/listen-focus
evidence; it does not copy audio, repair audio, train, publish, relax thresholds, or mark
`release_ready`.

For fully automated corpus-scale acceptance, run
`tools/build_machine_stem_acceptance_gate.py` after aggregate validation and the
minus quality audit. This gate blocks `machine_pass_human_listen_pending` as a
final state: `final_result_allowed=true` is emitted only when the aggregate pack
is accepted, validation passed, source vocal/minus master evidence is pair
compatible, and every minus row passes the machine quality audit with no
blockers or warnings. A blocked gate writes `gate.json`, `blockers.jsonl`,
`summary.md`, and `progress.jsonl` so the next ARIS loop can target exact
machine-detected defects instead of asking for manual listening.

After the audit, consume it with `tools/build_short_listen_subset_pack.py`
using `--minus-audit-json`. In audit-gated mode the subset builder keeps
non-minus rows, keeps only `minus` rows whose `audit_decision=passed`, writes
excluded blocked/evidence-missing minus rows to `excluded.jsonl`, and fails
closed if a selected minus row has no matching audit row. This prevents a
blocked minus from staying in the next listen handoff as an accepted row.

If minus audit rows are blocked or passed with warnings such as residual vocal
watch, route them into `tools/build_short_listen_repair_queue.py` with
`--minus-audit-json`. This creates a machine audit repair/mining queue with
`queue_source=minus_audit_warning`; passed warning rows get
`verdict=machine_warning`, blocked rows get `verdict=machine_blocked` and
`priority=high`. It does not pretend a human rejected the audio, does not
repair audio, does not train, and does not mark release-ready.

When the full-pack machine gate is blocked, the same queue builder may be run
with `--machine-gate-json`. That path emits `queue_source=machine_full_pack_gate`
and `queue_status=ready_for_full_pack_loop`, while preserving the blocked stem's
pack/minus-audit provenance (`source_manifest`, `source_audio_path`, and nested
`source_row`). `tools/build_short_listen_repair_candidate_pack.py` consumes this
status only for machine full-pack gate queues. For minus repairs it can mine
source-matching stem packs, full-track switch packs, no-reference preview
manifests, and `judge_guided_minus_vocal_leakage_repair` candidate-only
manifests into a standard repair candidate pack. Those candidate packs remain
`listen_preview_only` repair evidence: they still require result judge,
windowed judge, replacement sweep, minus quality audit, and the final full-pack
machine gate before any listen/result delivery is allowed.

If the minus hard-case model-route expansion exhausts the registered route
space without producing an accepted minus candidate, run
`tools/build_full_pack_route_exhaustion_recovery_plan.py`. It reads the machine
gate, minus audit, minus route-expansion manifest, reference gap plan, and raw
reference handoff, then writes short metadata artifacts `plan.json`,
`actions.jsonl`, `progress.jsonl`, and `summary.md`. The plan is metadata-only:
it records whether ready reference material can be ingested, or whether the
next ARIS action must collect verified reference material / design a new minus
route or training candidate before rerunning gates. It never exports audio,
never marks a listen result accepted, and keeps `delivery_allowed=false`.
When a metadata loop already attempted reference intake for the same handoff
roots, pass the fresh intake manifest with `--reference-intake-json`; if the
candidate roots were consumed with `accepted_reference_rows=0`, the planner
does not recommend re-ingesting the same rejected/duplicate roots and instead
keeps the next action on verified reference collection / new route design.

Use `tools/build_short_listen_subset_pack.py` when a source pack contains more
accepted rows than the current listen handoff should include. It filters by
stem type and route intent, keeps provenance in the manifest, and writes short
paths such as `audio/sub/v001.flac`. This is useful for replacing a rejected
vocal item without carrying older rejected minus alternatives into the next
aggregate pack.

Use `tools/build_minus_mid_pack.py` when a minus item is rejected for middle-song
vocal bleed or muffling and the next repair cycle needs precise listening
evidence. It slices short diagnostic clips from selected minus candidates and
optional explicit baseline/rejected sources, writing compact paths such as
`audio/m001/w001.flac`, `audio/m002/w001.flac`, and `audio/old/w001.flac`.
The long source candidate ids, source paths, hashes, and judge provenance stay
inside `pack.json` / `items.jsonl`. The output is diagnostic-only:
`delivery_allowed=false`, `release_ready=false`,
`can_publish_as_result=false`, and `semantic_clean_minus` is not claimed.
This tool does not run inference, train, repair, publish, relax thresholds, or
mark release-ready.

Use `tools/build_mid_feedback.py` to convert explicit diagnostic clip verdicts
from a `build_minus_mid_pack.py` pack into the existing
`short_listen_feedback_manifest.v1` shape consumed by
`tools/build_short_listen_repair_queue.py`. This is only a bridge from
diagnostic listening evidence into repair/mining; it does not invent verdicts
for unreviewed candidates. For a user-reported rejected baseline, select only
that short source alias, for example `old`, and attach the exact issue tags.

Example post-feedback aggregate pattern:

- `vrep`: accepted replacement vocal subset pack;
- `mpr`: accepted minus phase-repair selection pack;
- `noe6`, `hh6`, `t026`: available 6-stem/6s packs;
- `mi`: music-intelligence stem listen pack if already accepted.

The aggregate pack is still a listen pack, not a release artifact. The goal
audit can reach `machine_pass_human_listen_pending`; it must not be marked
complete until the short-path files have a human listen verdict.

## Run

Example generated output paths below use short Windows-safe segments. Replace
`/app/lab_artifacts/src/...` inputs with the actual accepted source manifests
for the current run; keep newly generated output folders and filenames short.

```bash
docker compose run --rm --no-deps ml-worker python tools/build_no_reference_stem_listen_preview_manifest.py \
  --stem-manifest-json /app/lab_artifacts/testing_stuff_music_track2_nonvocal_job_matrix_20260612/nine_thou_grant_mohrman_superstars_remix_320/melody/htdemucs_6s_aggressive_melody_first/stem_manifest.json \
  --target-stem melody \
  --track-id nine_thou_grant_mohrman_superstars_remix_320 \
  --output-manifest-json /app/lab_artifacts/lp_ex/stem/manifest.json \
  --summary-md /app/lab_artifacts/lp_ex/stem/summary.md
```

```bash
docker compose run --rm --no-deps ml-worker python tools/build_stem_manifest_audio_pack.py \
  --stem-manifest-json /app/lab_artifacts/src/noe/stem_manifest.json \
  --output-dir /app/lab_artifacts/lp_ex/s6 \
  --track-id noe \
  --path-prefix-map /Users/andrejviluha/aimusic_local_artifacts=/app/artifacts \
  --stem vocal \
  --stem minus \
  --stem drums \
  --stem bass \
  --stem other \
  --stem guitar \
  --stem piano
```

```bash
docker compose run --rm --no-deps ml-worker python tools/build_aggregate_short_listen_pack.py \
  --source-manifest voc=/app/lab_artifacts/src/voc/pack.json \
  --source-manifest min=/app/lab_artifacts/src/min/pack.json \
  --source-manifest noe6=/app/lab_artifacts/src/noe6/pack.json \
  --output-dir /app/lab_artifacts/lp_v37_001 \
  --pack-id lp_v37_001
```

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_aggregate_short_listen_pack.py \
  --pack-manifest-json /app/lab_artifacts/lp_v37_001/pack.json \
  --output-dir /app/lab_artifacts/lp_v37_001/validation
```

```bash
docker compose run --rm --no-deps ml-worker python tools/audit_short_listen_goal_status.py \
  --pack-manifest-json /app/lab_artifacts/lp_v37_001/pack.json \
  --validation-json /app/lab_artifacts/lp_v37_001/validation/validation.json \
  --output-dir /app/lab_artifacts/lp_v37_001/goal_audit
```

```bash
docker compose run --rm --no-deps ml-worker python tools/build_minus_mid_pack.py \
  --candidate-pack-json /app/lab_artifacts/lp_v43_001/pack.json \
  --output-dir /app/lab_artifacts/lp_v43_001/mid002 \
  --source-alias mpr \
  --stem-type minus \
  --extra-source old=/app/lab_artifacts/lp_v40_001/audio/mpr/min/m001.flac \
  --extra-source-manifest old=/app/lab_artifacts/lp_v39_001/mpr_sel_001/pack.json \
  --window w001=48.0:12.0:middle_bleed \
  --window w002=69.22059:12.0:judge_probe \
  --window w003=138.441179:12.0:late_detail \
  --handoff-playlist-m3u /app/lab_artifacts/lp_v43_001/rp001/review_handoff/mid2.m3u
```

```bash
docker compose run --rm --no-deps ml-worker python tools/build_short_minus_quality_audit.py \
  --pack-manifest-json /app/lab_artifacts/lp_v45_001/pack.json \
  --output-dir /app/lab_artifacts/lp_v45_001/minqa001 \
  --window-seconds 12 \
  --max-windows-per-candidate 5
```

```bash
docker compose run --rm --no-deps ml-worker python tools/build_short_listen_subset_pack.py \
  --source-manifest-json /app/lab_artifacts/lp_v45_001/pack.json \
  --minus-audit-json /app/lab_artifacts/lp_v45_001/minqa001/audit.json \
  --output-dir /app/lab_artifacts/lp_v45_001/gsub001 \
  --alias-prefix g
```

```bash
docker compose run --rm --no-deps ml-worker python tools/build_short_listen_repair_queue.py \
  --minus-audit-json /app/lab_artifacts/lp_v45_001/minqa001/audit.json \
  --output-dir /app/lab_artifacts/lp_v45_001/gsub001/mwq001
```

```bash
docker compose run --rm --no-deps ml-worker python tools/build_short_listen_feedback_template.py \
  --pack-manifest-json /app/lab_artifacts/lp_v45_001/gsub001/pack.json \
  --output-dir /app/lab_artifacts/lp_v45_001/gsub001/tpl001
```

```bash
docker compose run --rm --no-deps ml-worker python tools/build_short_listen_review_handoff.py \
  --pack-manifest-json /app/lab_artifacts/lp_v45_001/gsub001/pack.json \
  --feedback-template-manifest-json /app/lab_artifacts/lp_v45_001/gsub001/tpl001/tpl.json \
  --output-dir /app/lab_artifacts/lp_v45_001/gsub001/rh001
```

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_short_listen_review_handoff.py \
  --handoff-manifest-json /app/lab_artifacts/lp_v45_001/gsub001/rh001/handoff.json \
  --output-dir /app/lab_artifacts/lp_v45_001/gsub001/rhv001
```

```bash
docker compose run --rm --no-deps ml-worker python tools/build_mid_feedback.py \
  --mid-pack-json /app/lab_artifacts/lp_v43_001/mid002/pack.json \
  --output-dir /app/lab_artifacts/lp_v43_001/mid002/fb001 \
  --select-source old \
  --verdict rejected \
  --issue-tag muffled_minus \
  --issue-tag residual_vocal_middle \
  --issue-tag user_reported_unacceptable \
  --note "User reported middle-section muffling and residual vocal in the rejected minus baseline."
```

```bash
docker compose run --rm --no-deps ml-worker python tools/build_short_listen_repair_queue.py \
  --feedback-manifest-json /app/lab_artifacts/lp_v43_001/mid002/fb001/feedback.json \
  --output-dir /app/lab_artifacts/lp_v43_001/mid002/rq001
```

```bash
docker compose run --rm --no-deps ml-worker python tools/build_short_listen_feedback_manifest.py \
  --pack-manifest-json /app/lab_artifacts/lp_v37_001/pack.json \
  --feedback-jsonl /app/lab_artifacts/lp_v37_001/feedback/feedback_items.jsonl \
  --output-dir /app/lab_artifacts/lp_v37_001/feedback
```

```bash
docker compose run --rm --no-deps ml-worker python tools/build_short_listen_feedback_template.py \
  --pack-manifest-json /app/lab_artifacts/lp_v37_001/pack.json \
  --output-dir /app/lab_artifacts/lp_v37_001/feedback_template
```

```bash
docker compose run --rm --no-deps ml-worker python tools/build_short_listen_review_handoff.py \
  --pack-manifest-json /app/lab_artifacts/lp_v37_001/pack.json \
  --feedback-template-manifest-json /app/lab_artifacts/lp_v37_001/feedback_template/tpl.json \
  --output-dir /app/lab_artifacts/lp_v37_001/review_handoff
```

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_short_listen_review_handoff.py \
  --handoff-manifest-json /app/lab_artifacts/lp_v37_001/review_handoff/handoff.json \
  --output-dir /app/lab_artifacts/lp_v37_001/review_handoff_validation
```

```bash
docker compose run --rm --no-deps ml-worker python tools/run_short_listen_review_packet.py \
  --pack-manifest-json /app/lab_artifacts/lp_v37_001/pack.json \
  --aggregate-validation-json /app/lab_artifacts/lp_v37_001/validation/validation.json \
  --output-dir /app/lab_artifacts/lp_v37_001/review_packet_001
```

```bash
docker compose run --rm --no-deps ml-worker python tools/run_short_listen_feedback_cycle.py \
  --pack-manifest-json /app/lab_artifacts/lp_v37_001/pack.json \
  --aggregate-validation-json /app/lab_artifacts/lp_v37_001/validation/validation.json \
  --feedback-jsonl /app/lab_artifacts/lp_v37_001/review_packet_001/feedback_template/items.jsonl \
  --output-dir /app/lab_artifacts/lp_v37_001/feedback_cycle_001
```

```bash
docker compose run --rm --no-deps ml-worker python tools/build_short_listen_repair_queue.py \
  --feedback-manifest-json /app/lab_artifacts/lp_v37_001/feedback/feedback.json \
  --output-dir /app/lab_artifacts/lp_v37_001/repair_queue
```

```bash
docker compose run --rm --no-deps ml-worker python tools/build_short_listen_repair_candidate_pack.py \
  --repair-queue-manifest-json /app/lab_artifacts/lp_v37_001/repair_queue/queue.json \
  --vocal-training-examples-jsonl /app/lab_artifacts/src/vtrain/examples.jsonl \
  --output-dir /app/lab_artifacts/lp_v37_001/repair_candidates
```

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_result_judge.py \
  --candidate-jsonl /app/lab_artifacts/lp_v37_001/repair_candidates/minus_judge_candidates.jsonl \
  --output-dir /app/lab_artifacts/lp_v37_001/repair_candidates/minus_result_judge \
  --score-mode minus
```

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_windowed_judge.py \
  --candidates-jsonl /app/lab_artifacts/lp_v37_001/repair_candidates/minus_judge_candidates.jsonl \
  --output-jsonl /app/lab_artifacts/lp_v37_001/repair_candidates/wj/wj.jsonl \
  --summary-md /app/lab_artifacts/lp_v37_001/repair_candidates/wj/summary.md \
  --cache-jsonl /app/lab_artifacts/lp_v37_001/repair_candidates/wj/cache.jsonl \
  --score-mode minus
```

```bash
docker compose run --rm --no-deps ml-worker python tools/build_short_listen_repair_selection_pack.py \
  --repair-candidate-pack-json /app/lab_artifacts/lp_v37_001/repair_candidates/pack.json \
  --minus-result-judge-jsonl /app/lab_artifacts/lp_v37_001/repair_candidates/minus_result_judge/no_reference_result_judge.jsonl \
  --minus-windowed-judge-jsonl /app/lab_artifacts/lp_v37_001/repair_candidates/wj/wj.jsonl \
  --output-dir /app/lab_artifacts/lp_v37_001/repair_selection \
  --max-vocal-candidates 1 \
  --max-minus-candidates 3
```

For a user-rejected middle-minus section, do not accept a whole replacement
candidate until it is compared against the rejected baseline at the same
windows. The comparison report is diagnostic only and never release-ready:

```bash
docker compose run --rm --no-deps ml-worker python tools/build_minus_cmp.py \
  --mid-pack-json /app/lab_artifacts/lp_v43_001/mid002/pack.json \
  --selection-pack-json /app/lab_artifacts/lp_v43_001/mid002/rs004/pack.json \
  --output-dir /app/lab_artifacts/lp_v43_001/mid002/cmp001 \
  --mix-path "/app/testing_flstudio_music/Chris Bulman & Sarju Jagai - No Escape.flac" \
  --vocal-path /app/lab_artifacts/autonomous_runs/arch_1a_fresh_music_manifest_001/candidates/jobs/local_registered__htdemucs_6s__both__shared/registered/vocals.flac \
  --baseline-alias old
```

If the comparison improves only some windows, build full-track switch inputs
instead of replacing the whole minus. The switch-input builder accepts both
`short_listen_repair_selection_pack_manifest.v1` and
`short_candidate_wave_listen_pack_manifest.v1`, so registered-model
candidate-wave packs can enter the same segment-switch path. Regressed windows
remain blocked for the alternate candidate and stay on the base candidate. If
`improved_window_count=0`, stop here and do not render a full-track preview:

```bash
docker compose run --rm --no-deps ml-worker python tools/build_minus_cmp_switch_inputs.py \
  --cmp-json /app/lab_artifacts/lp_v43_001/mid002/cmp001/cmp.json \
  --selection-pack-json /app/lab_artifacts/lp_v43_001/mid002/rs004/pack.json \
  --output-dir /app/lab_artifacts/lp_v43_001/mid002/sw001 \
  --track-id no_escape \
  --mix-path "/app/testing_flstudio_music/Chris Bulman & Sarju Jagai - No Escape.flac" \
  --vocal-path /app/lab_artifacts/autonomous_runs/arch_1a_fresh_music_manifest_001/candidates/jobs/local_registered__htdemucs_6s__both__shared/registered/vocals.flac \
  --base-candidate-id old
```

Render a full-duration preview-only switch candidate with short filenames:

```bash
docker compose run --rm --no-deps ml-worker python tools/render_no_reference_full_track_segment_switch_candidate.py \
  --candidate-pool-jsonl /app/lab_artifacts/lp_v43_001/mid002/sw001/pool.jsonl \
  --windowed-judge-jsonl /app/lab_artifacts/lp_v43_001/mid002/sw001/wj.jsonl \
  --output-dir /app/lab_artifacts/lp_v43_001/mid002/fts001 \
  --manifest-json /app/lab_artifacts/lp_v43_001/mid002/fts001/manifest.json \
  --summary-md /app/lab_artifacts/lp_v43_001/mid002/fts001/summary.md \
  --base-candidate-id old \
  --selection-decision-field cmp_selection_decision
```

The full-track switch manifest must preserve the original rejected/base
minus path as `source_minus_path` and `base_minus_path`. The downstream
repair-candidate pack rejects full-track switch manifests that only expose
the rendered preview minus, because replacement/sweep gates must compare a
new candidate against the original minus source rather than against the
already repaired artifact.

Promote the full-track switch to an accepted listen-pack source only after
post-render result judge, windowed judge, and targeted same-window comparison
all pass. This still does not mean release-ready:

```bash
docker compose run --rm --no-deps ml-worker python tools/build_fts_listen_pack.py \
  --switch-manifest-json /app/lab_artifacts/lp_v43_001/mid002/fts001/manifest.json \
  --result-judge-jsonl /app/lab_artifacts/lp_v43_001/mid002/fts001/rj001/no_reference_result_judge.jsonl \
  --windowed-judge-jsonl /app/lab_artifacts/lp_v43_001/mid002/fts001/wj001/wj.jsonl \
  --targeted-cmp-json /app/lab_artifacts/lp_v43_001/mid002/fts001/tcmp001/manifest.json \
  --output-dir /app/lab_artifacts/lp_v43_001/mid002/flp001
```

The improvement loop is an explicit idea funnel:

1. Capture the listen failure as feedback and repair queue rows.
2. Generate or collect candidates through Docker-only routes.
3. Compare candidates against the rejected baseline at the failing windows.
4. Keep only windows with objective proxy improvement.
5. Render a full-track preview only through segment-aware switching.
6. Run post-render result/windowed judges and targeted comparison.
7. Build the next listen pack only from accepted, short-name source manifests.
8. If any gate fails, add the failure to the next ARIS/brainstorm shortlist and
   do not publish the candidate as a result.

Use `tools/build_listen_idea_funnel.py` to make that shortlist reproducible.
It reads the current goal audit, aggregate pack, optional minus comparison,
targeted comparison, and optional feedback manifest, then writes short-name
`funnel.json`, `ideas.jsonl`, `summary.md`, and `progress.jsonl`. The funnel is
not a listen result, does not train, and keeps `release_ready=false`. When
complete 6-stem groups exist, the funnel must include both the short semantic
review-pack command and the follow-up semantic batch command so semantic quality
does not remain a manual hidden step. If semantic review completes with blocked
stems, the funnel must also include a route-gap planning command rather than
pretending the pack is ready. When minus rows exist, the funnel must include
the minus quality audit, the audit-gated subset command, the machine audit
repair queue command, the minus replacement sweep command, and concrete
current-audit ideas for blocked/warning minus rows. Blocked minus rows must carry exact `blocker_reasons` and
`blocked_audio_paths`, so they cannot be hidden behind a generic next step.

```bash
docker compose run --rm --no-deps ml-worker python tools/build_listen_idea_funnel.py \
  --goal-audit-json /app/lab_artifacts/lp_v45_001/rp001/goal_audit/goal_status_audit.json \
  --pack-manifest-json /app/lab_artifacts/lp_v45_001/pack.json \
  --minus-cmp-json /app/lab_artifacts/lp_v43_001/mid002/cmp001/cmp.json \
  --targeted-cmp-json /app/lab_artifacts/lp_v43_001/mid002/fts001/tcmp001/manifest.json \
  --output-dir /app/lab_artifacts/lp_v45_001/idea001
```

Use `tools/build_music_intelligence_stem_listen_pack.py` to materialize the
semantic stem-quality review queue for complete 6-stem groups. For an aggregate
pack, pass `--input-pack-json`, restrict to complete short aliases with
`--source-alias`, and require every stem type. The tool copies existing audio to
short review paths, writes `pack.json`, `selected.jsonl`, `review.json`,
`rows.jsonl`, `summary.md`, and `progress.jsonl`, and marks every row
`review_status=semantic_review_pending`. A successful queue uses
`review_queue_ready=true`, `delivery_status=semantic_review_pending`,
`delivery_allowed=false`, `listenable=false`, and
`presentation_status=semantic_review_queue_not_for_listen_pack`; it is a review
artifact, not semantic proof or a listen-pack result.

```bash
docker compose run --rm --no-deps ml-worker python tools/build_music_intelligence_stem_listen_pack.py \
  --input-pack-json /app/lab_artifacts/lp_v45_001/pack.json \
  --source-alias noe6 \
  --source-alias hh6 \
  --source-alias t026 \
  --required-stem-type vocal \
  --required-stem-type minus \
  --required-stem-type drums \
  --required-stem-type bass \
  --required-stem-type other \
  --required-stem-type guitar \
  --required-stem-type piano \
  --output-dir /app/lab_artifacts/lp_v45_001/sem001
```

Then run the existing semantic batch against the short `pack.json`. This does
not copy/export audio and does not run model inference; it writes short metadata
under `semj001`, including `review/review.json`. Unsupported stems remain
blocked rows instead of being treated as clean ground truth. Promotion into any
listen/result handoff still requires completed semantic review, minus audit when
minus rows exist, aggregate validation, and the final machine gate.

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_stem_semantic_batch.py \
  --input-pack-json /app/lab_artifacts/lp_v45_001/sem001/pack.json \
  --short-output-names \
  --output-root /app/lab_artifacts/lp_v45_001/semj001
```

Use `tools/build_semantic_stem_route_gap_plan.py` after semantic review when
blocked stems remain. It groups blockers by stem type, writes `gap.json`,
`gaps.jsonl`, `summary.md`, and `progress.jsonl`, and recommends the next route
family to add or run. It is planning-only: no audio inference, training,
publishing, threshold relaxation, or release-ready transition.

```bash
docker compose run --rm --no-deps ml-worker python tools/build_semantic_stem_route_gap_plan.py \
  --semantic-review-json /app/lab_artifacts/lp_v45_001/semj001/review/review.json \
  --output-dir /app/lab_artifacts/lp_v45_001/semgap001
```

Use `tools/build_listen_coverage_scorecard.py` after the idea funnel to make the
goal checkpoint explicit. It reads `pack.json`, `coverage.json`, the goal audit,
`funnel.json`, the semantic review artifact, and optionally the minus quality
audit, then writes short-name `score.json`, `groups.jsonl`, `summary.md`, and
`progress.jsonl`. The scorecard fails closed if the idea funnel is missing or
not `ready`, if semantic stem review is missing/pending for complete 6-stem
groups, if the minus audit reports blocked/evidence-missing minus rows, if any
source tries to set `release_ready`, or if safety flags drift. It still does
not publish a result: it records `scorecard_status`, stem coverage counts,
complete 6-stem groups, semantic quality gaps, minus-audit blockers, and the
remaining human listen blocker.

```bash
docker compose run --rm --no-deps ml-worker python tools/build_listen_coverage_scorecard.py \
  --pack-manifest-json /app/lab_artifacts/lp_v45_001/pack.json \
  --coverage-json /app/lab_artifacts/lp_v45_001/coverage.json \
  --goal-audit-json /app/lab_artifacts/lp_v45_001/rp001/goal_audit/goal_status_audit.json \
  --idea-funnel-json /app/lab_artifacts/lp_v45_001/idea001/funnel.json \
  --semantic-review-json /app/lab_artifacts/lp_v45_001/semj001/review/review.json \
  --minus-audit-json /app/lab_artifacts/lp_v45_001/minqa001/audit.json \
  --output-dir /app/lab_artifacts/lp_v45_001/cov001
```

`tools/run_short_listen_feedback_cycle.py` is only allowed to finish the
`human_accepted_no_repair_needed` or `repair_queue_ready` paths after the goal
audit has already proven all non-human machine evidence. Pass
`--machine-gate-json` from `tools/build_machine_stem_acceptance_gate.py`; that
gate must carry aggregate validation, semantic review evidence, and completed
minus audit evidence. If the gate is missing or any other non-human blocker
remains, the feedback cycle fails closed with
`goal_audit:machine_evidence_incomplete`, even when the human feedback itself is
rejected. This prevents rejected feedback from silently becoming a repair queue
while aggregate validation, semantic review, minus audit, or final machine-gate
evidence is absent.

## Validation

```bash
docker compose run --rm --no-deps ml-worker python -m pytest \
  /app/tests/test_build_no_reference_stem_listen_preview_manifest.py \
  /app/tests/test_build_stem_manifest_audio_pack.py \
  /app/tests/test_build_music_intelligence_stem_listen_pack.py \
  /app/tests/test_build_aggregate_short_listen_pack.py \
  /app/tests/test_validate_aggregate_short_listen_pack.py \
  /app/tests/test_audit_short_listen_goal_status.py \
  /app/tests/test_build_short_listen_feedback_template.py \
  /app/tests/test_build_short_listen_review_handoff.py \
  /app/tests/test_validate_short_listen_review_handoff.py \
  /app/tests/test_run_short_listen_review_packet.py \
  /app/tests/test_run_short_listen_feedback_cycle.py \
  /app/tests/test_build_short_listen_feedback_manifest.py \
  /app/tests/test_build_short_listen_repair_queue.py \
  /app/tests/test_build_short_listen_repair_candidate_pack.py \
  /app/tests/test_build_short_listen_repair_selection_pack.py \
  /app/tests/test_build_minus_cmp.py \
  /app/tests/test_build_minus_cmp_switch_inputs.py \
  /app/tests/test_render_no_reference_full_track_segment_switch_candidate.py \
  /app/tests/test_build_fts_listen_pack.py \
  /app/tests/test_run_no_reference_stem_semantic_batch.py \
  /app/tests/test_build_semantic_stem_route_gap_plan.py \
  /app/tests/test_build_listen_idea_funnel.py \
  /app/tests/test_build_listen_coverage_scorecard.py \
  /app/tests/test_build_short_listen_subset_pack.py \
  /app/tests/test_build_short_minus_phase_repair_selection_pack.py \
  /app/tests/test_artifact_status_contract.py \
  -q
```
