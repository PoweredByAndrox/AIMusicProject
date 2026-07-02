# Train-Ready Audio Model Decision

This contract is the gate for claiming a verified TRAIN result. It does not
promote a model, does not edit runtime gates, does not create listen artifacts,
and always keeps `release_ready=false`.

## Verified Corpus Gate

`tools/validate_verified_audio_training_corpus.py` validates
`generic_finetuning_audio_training_example.v1` rows before GPU training.

Required per row:

- `trainable=true`
- `permission_status` in `owned`, `licensed`, `cleared`, `project_owned`
- `human_verified=true` or `reference_verified=true`
- existing `source_audio_path`
- existing `target_audio_path`
- no `release_ready=true`
- no `delivery_allowed=true`

The validator also blocks track leakage across split labels. A track cannot
appear in both train and dev/test inside the same corpus view.

Outputs:

- `verified_audio_training_corpus_manifest.json`
- `verified_audio_training_corpus_rows.jsonl`
- `verified_audio_training_corpus_progress.jsonl`

The route is metadata-only:

- `training_started=false`
- `fine_tune_started=false`
- `models_changed=false`
- `model_registry_edited=false`
- `release_ready=false`

## Train Result Decision Gate

`tools/build_train_ready_audio_model_decision.py` reads a completed
`generic_audio_finetune_matrix_manifest.json`, recomputes the best config
success rate from the matrix rows, and writes a final train-result decision.

Default success rule:

```text
success_rate > 0.90
```

Exactly `0.90` is not enough. This intentionally matches the project goal of
greater-than-90% TRAIN success, not at-least-90%.

The gate fails closed if any of these are true:

- matrix validation did not pass
- best config is missing
- best row has failed checks
- holdout eval count is missing
- success rate is not strictly above threshold
- matrix or row says `release_ready=true`
- matrix or row says `delivery_allowed=true`
- matrix says registry/gates/thresholds changed
- leak-safe split manifest reports critical leakage
- holdout rows show a holdout track also used for checkpoint training
- holdout rows show CPU application for GPU-required evaluation
- holdout rows have `status` other than `passed`
- holdout rows carry row-level `failed_checks`

Outputs:

- `train_ready_audio_model_decision_manifest.json`
- `train_ready_audio_model_decision_summary.md`
- `train_ready_audio_model_decision_progress.jsonl`

For candidate-family blend adapter matrices, first write an explicit locked
split manifest with `tools/build_candidate_family_blend_locked_split.py`. The
builder records train and holdout track ids, requires complete source-family
groups, rejects pseudo-teacher or release-ready input rows, and writes
`critical_leakage_count=0` only when train and holdout tracks do not overlap.
Pass the manifest to this decision gate through
`--leak-safe-split-manifest-json`.

Allowed final decisions:

- `train_result_verified`
- `train_result_not_ready`

Even when `train_result_verified`, the artifact remains training evidence only:

- `promotion_allowed=false`
- `delivery_allowed=false`
- `release_ready=false`

To create a user-checkable listening package from a verified candidate-family
blend matrix, use `tools/build_candidate_family_blend_listen_pack.py`. It does
not rerender or repair audio. It copies only already-passed holdout candidate
audio into a listen pack, writes `delivery_status=accepted`,
`delivery_allowed=true`, `presentation_status=accepted_for_listen_pack`, and
keeps `release_ready=false`.

Listen-pack audio paths must use short Windows-safe aliases such as
`audio/v001.flac` or `audio/t001/vocal.flac`. Raw track names, full hashes,
config IDs, and stem IDs stay in the manifest provenance fields, not in file or
directory names.

Listen-pack metadata files use short names too:

- `pack.json`
- `selected.jsonl`
- `progress.jsonl`
- `summary.md` when a human summary is written
- `coverage.json` for aggregate packs

For full-length vocal listening, use
`tools/build_candidate_family_full_length_listen_pack.py`. It applies the
learned candidate-family blend weights to the full source candidate audio and
does not crop the output to a short target/reference clip. This is still a
listen-check artifact only.

For accepted runtime stem listening, use
`tools/build_music_intelligence_stem_listen_pack.py`. It copies only
`analysis_allowed=true`, `quality_status=accepted` Music Intelligence stems,
such as accepted `vocal` and `minus`, and keeps `release_ready=false`.

For full 6s stem-pack delivery, run
`tools/build_machine_stem_acceptance_gate.py` with
`--require-training-coverage`, `--training-plan-json`, and one or more
`--train-ready-decision-json` inputs. The gate requires every required stem in
the pack to have an existing stem audio file plus verified train-ready model
evidence; a manifest row or ready training job alone is not enough. Missing
files are blocked as `required_stem_audio_file_missing:<stem>` without reading or
decoding audio bytes. Blocked training jobs are reported per stem unless that
stem is already covered by a verified `train_result_verified` decision. The gate
remains fail-closed: a pack with missing stem files, missing stem model evidence,
failed minus audit, failed semantic review, or failed validation must keep
`final_result_allowed=false` and must not be presented as a listen/result
artifact.

Before closing the ARIS full-pack goal, run
`tools/audit_full_pack_goal_completion.py` with the latest machine gate, ARIS
cycle status, reference collection queue, verified-corpus manifest, and training
plan. The audit writes short metadata-only files (`audit.json`, `summary.md`,
`progress.jsonl`) and is not itself a listen artifact. It independently requires
the full 6s pack contract: `required_stems`, actual `stem_types`, mandatory
machine-gate names, and train-ready model evidence must cover `vocal`, `minus`,
`drums`, `bass`, `other`, `guitar`, and `piano`. A gate that proves only
`vocal`/`minus` or only ready jobs must keep `goal_complete=false`,
`result_delivery_allowed=false`, and `result_presentation_status=not_accepted`.

When that gate is blocked, feed its `gate.json` to
`tools/build_short_listen_repair_queue.py --machine-gate-json`. This produces a
machine-only next-action queue with `queue_status=ready_for_full_pack_loop` and
`delivery_allowed=false`. The queue does not repair audio and does not start
training; it converts gate blockers into explicit actions such as minus repair,
verified-reference corpus expansion, GPU teacher-mask training, holdout eval,
train-ready decision rebuild, and final full-pack gate rebuild. Stems already
covered by a verified train-ready decision are not requeued only because an
alternative training job was blocked.

Then feed that queue to `tools/build_full_pack_loop_iteration_plan.py` with the
current `--training-plan-json`. This writes short machine-only artifacts
(`plan.json`, `rows.jsonl`, `progress.jsonl`, `summary.md`) that say which
actions are executable now and which are blocked by missing verified references,
missing stem training jobs, or unresolved minus repair. It must keep
`delivery_allowed=false`, `training_started=false`, and `release_ready=false`;
it is an ARIS loop execution plan, not a listen pack or accepted result.

Feed the resulting `plan.json` to
`tools/build_generic_audio_training_recovery_queue.py` with
`--full-pack-loop-plan-json --short-output-names` to build a
corpus-expansion worklist for the whole 6s stem pack. Existing verified
reference candidates are annotated with the full-pack blocker that needs them;
missing stems get blocked `full_pack_loop_plan_missing_stem` rows with
`missing_verified_reference_candidate`. This queue is still metadata-only, must
write short files (`queue.jsonl`, `tpl.jsonl`, `rec.json`, `progress.jsonl`),
and must keep `training_started=false`, `delivery_allowed=false`, and
`release_ready=false`.

If the recovery queue contains permission-only verified reference gaps, use
`tools/build_project_owned_audio_permission_decisions.py --short-output-names`
and then `tools/ingest_generic_audio_training_recovery_decisions.py
--short-output-names`. These steps write short metadata outputs (`dec.jsonl`,
`perm.json`, `audio.jsonl`, `ingest.json`, `progress.jsonl`) and remain
dataset/recovery artifacts only; they must not start training, copy listen
audio, or mark any result accepted.

After recovery decisions and corpus validation, use
`tools/build_full_pack_reference_corpus_gap_plan.py` to convert the blocked
full-pack loop plan, training plan, and verified-corpus manifest into exact
reference-corpus requirements. It writes short artifacts (`gap.json`,
`rows.jsonl`, `progress.jsonl`, `summary.md`) and records per-stem missing
unique-track counts plus missing training jobs. This is the handoff for
collecting more verified reference stems; it must not read/copy audio, build a
dataset, start training, relax thresholds, or create listen artifacts.

Before importing candidate material, use
`tools/audit_full_pack_reference_material_discovery.py` when you need a
machine-readable proof of whether existing disk artifacts contain eligible
verified reference bundles for the remaining full 6s stem-pack gaps. The audit
uses the gap plan and current approved audio-training examples, scans one or
more `--candidate-root` locations for `ground_truth_manifest.json`, and writes
short files (`discovery.json`, `rows.jsonl`, `eligible.jsonl`,
`rejected.jsonl`, `progress.jsonl`, `summary.md`). It does not import rows,
read/decode/copy audio, start training, or create listen artifacts. Eligible
rows only mean the material is ready for the normal intake gate; rejected rows
carry machine-readable reasons such as duplicate track/stem, synthetic fixture
source, missing rights, missing audio paths, or unsafe manifest flags.

Run `tools/audit_full_pack_raw_reference_candidates.py` on the same candidate
roots when existing disk artifacts may contain raw audio folders without
`ground_truth_manifest.json`. It lists only path-level candidates such as a
detected full/source mix plus required stem-looking files, writes short files
(`raw.json`, `rows.jsonl`, `reject.jsonl`, `progress.jsonl`, `summary.md`), and
marks every row `triage_status=needs_verified_ground_truth_manifest`. This
audit is diagnostic-only: it must not read/decode/copy audio, write
`ground_truth_manifest.json`, import training rows, or create accepted/listen
results. Raw wav/mp3 folders become actionable collection evidence, not
verified references.
For short 6s listen packs, stem audio may live under paths such as
`audio/stm/t001/guitar.flac` while the source mix exists only in manifest
provenance. The raw audit may resolve that source mix from the nearest accepted
`pack.json` (`stem_manifest_audio_pack_manifest.v1` or
`aggregate_short_listen_pack_manifest.v1`) only when the pack rows match the
scanned stem files and the referenced `source_audio_path` exists. This removes
false `missing_source_mix_candidate` blockers for complete 6s stem packs, but
does not accept the pack: every row remains
`needs_verified_ground_truth_manifest`, with `delivery_allowed=false` and
`ground_truth_manifest_written=false`, until the normal provenance, rights,
waveform, and reconstruction gates accept it.
Because these rows came from listen-pack provenance rather than a clean
user/project FL Studio reference export, the verified-bundle bridge must fail
closed with
`source_mix_from_listen_pack_manifest_requires_clean_reference_bundle` instead
of turning generated 6s preview stems into training ground truth. They are
useful to prove that the whole 6s pack exists and to drive collection, not to
silently create pseudo-reference data.

Run `tools/build_full_pack_raw_reference_bundle_worklist.py` after raw
candidate triage to turn those diagnostic rows into short work items under
`refs/r001/`, `refs/r002/`, and so on. The worklist writes only metadata
templates (`work.json`, `items.jsonl`, `refs/<alias>/tpl.json`,
`progress.jsonl`, `summary.md`) with candidate source/stem paths and required
actions. It must still keep `accepted_reference_rows=0`,
`eligible_reference_material_rows=0`, `delivery_allowed=false`, and
`ground_truth_manifest_written=false`; the template becomes
`ground_truth_manifest.json` only after independent provenance, rights, and
waveform/reference verification.
When a formal collection handoff already exists, pass `--handoff-json` so the
worklist is aligned to the full-pack slots rather than only to whatever raw
folders happened to be discovered. In this mode `handoff.json` is the slot
source of truth: rows preserve `bundle_alias`, `source_collection_id`, `gap_id`,
`queue_item_id`, and `slot_index`; matching raw candidates become
`needs_verified_ground_truth_manifest`; required stems without a raw candidate
remain explicit `blocked_missing_raw_reference_candidate` rows. This prevents a
partial raw scan, for example only `vocal`/`minus` or a subset of 6s stems, from
silently shrinking the full pack.

When raw work items point at actual user/project FL Studio exports, use
`tools/build_full_pack_raw_reference_verified_bundles.py` to run that
verification bridge. The builder requires explicit `--allowed-source-root` and
`--rights-status`, decodes the candidate audio through
`build_flstudio_ground_truth_bundle.py`, writes short outputs (`bundles.json`,
`rows.jsonl`, `ready.jsonl`, `blocked.jsonl`, `progress.jsonl`, `summary.md`),
and moves only verified bundles to `refs/b001/`, `refs/b002/`, and so on.
Blocked rows stay diagnostic-only with machine-readable blockers such as
`source_not_under_allowed_root`, `full_mix_reconstruction_not_accepted`, or a
short `failure_detail` for audio import errors. The builder never starts
training, never creates listen artifacts, and keeps `delivery_allowed=false`;
its `candidate_roots` may be passed to the normal reference intake only when
`verified_bundle_count>0`. Slot-aware worklists may also contain
`blocked_missing_raw_reference_candidate` rows from formal handoff slots; the
verified-bundle bridge must preserve those rows in `blocked.jsonl` and
`blocked_by_stem` instead of failing the whole run or dropping the stem from the
pack. `run_full_pack_aris_metadata_loop.py` can pass a formal collection handoff
into that worklist with `--raw-reference-handoff-json`; it can run the verified
bundle bridge inline only when the matching raw-reference allowed roots and
rights status are supplied; otherwise it records
`raw_reference_verified_bundle_status=not_requested`.
When a raw candidate directory contains unrelated audio, the bridge may perform
a mixed-directory diagnostic import so it can record `input_scope_contract`,
classified stems, and reconstruction status in `diag/<alias>/gt_rej.json`.
Those rows must stay blocked with `input_scope_not_clean`, must not produce a
`candidate_root`, and must not write an accepted
`ground_truth_manifest.json` until a clean scoped export is provided.

When candidate `ground_truth_manifest.json` bundles are available, run
`tools/build_full_pack_reference_corpus_intake.py` with the gap plan, current
approved audio-training examples, and one or more `--candidate-root` paths. The
intake accepts only required stems from new `(track_id, stem_type)` pairs with
`ground_truth_contract.verified=true`, trainable rights, user-supplied/FL
source provenance, and existing source/target audio paths. Duplicate,
synthetic, unverified, missing-rights, or missing-audio rows are rejected with
machine-readable reasons. Outputs stay short (`intake.json`, `rows.jsonl`,
`audio.jsonl`, `progress.jsonl`, `summary.md`) and are not listen artifacts.

After `tools/build_full_pack_raw_reference_verified_bundles.py`, run
`tools/build_full_pack_raw_reference_bundle_recovery_queue.py` when blocked raw
bundle rows remain. It reads `bundles.json` plus `blocked.jsonl` and writes
short metadata-only recovery artifacts (`queue.json`, `rows.jsonl`,
`progress.jsonl`, `summary.md`). The queue classifies blockers into
machine-actionable intents:

- `collect_clean_reference_bundle` for listen-pack/source-root provenance
  blockers that need a clean project/user reference export;
- `repair_or_replace_reconstruction_failed_bundle` for failed full-mix
  reconstruction, mixed-scope, or bundle-build failures;
- `replace_missing_requested_stem_candidate` for missing raw candidates or
  `verified_bundle_missing_requested_stem:<stem>` blockers.

This queue must keep `accepted_reference_rows=0`,
`eligible_reference_material_rows=0`, `delivery_allowed=false`,
`release_ready=false`, `training_started=false`, and
`ground_truth_manifest_written=false`. It is only the next-action map for ARIS;
it must not import blocked rows, copy/decode audio, start training, or present
anything as a listen/result artifact.
If a blocked verified-bundle row already carries output fields such as
`candidate_root` or `ground_truth_manifest_json`, the queue must fail closed
instead of forwarding that row as actionable recovery input.
Run `tools/build_full_pack_raw_reference_recovery_action_plan.py` on that queue
to convert the blocked recovery rows into short, deterministic ARIS action
items (`plan.json`, `items.jsonl`, `progress.jsonl`, `summary.md`). The action
plan maps each row to the next gate (`collect_verified_clean_reference_bundle`,
`rerun_full_mix_reconstruction_audit`, `rerun_raw_reference_bundle_worklist`, or
`inspect_raw_reference_blockers`) while preserving source bundle aliases,
requested stems, blockers, and diagnostic paths. It must keep
`accepted_reference_rows=0`, `eligible_reference_material_rows=0`,
`delivery_allowed=false`, `training_started=false`, `audio_read=false`, and
`ground_truth_manifest_written=false`; it is not a command execution result and
cannot be presented as a listen pack.
Run `tools/execute_full_pack_raw_reference_recovery_action_plan.py` to execute
only the metadata/handoff part of that plan. It writes short outputs
(`exec.json`, `items.jsonl`, `blocked.jsonl`, `progress.jsonl`, `summary.md`,
and `tpl/a001.json`-style request templates) and marks each action as blocked
waiting for external clean reference material, a clean scoped export, or a
missing requested stem candidate. This executor must not read/decode/copy audio,
write audio placeholders, create `ground_truth_manifest.json`, import reference
rows, start training, or set any delivery/listen status.
Run `tools/build_full_pack_raw_reference_recovery_material_responses.py` before
that preflight when clean verified bundle roots are already available. The
builder consumes `exec.json`, scans only explicit `--candidate-root` locations
for existing `ground_truth_manifest.json` bundles, verifies requested stems,
rights, FL Studio/user provenance, and safe flags, then writes short
`rsp/a001.json`-style response metadata plus `responses.json`, `items.jsonl`,
`ready.jsonl`, `blocked.jsonl`, `progress.jsonl`, and `summary.md`. Missing or
unsafe material stays blocked and no response file is written for that action.
This step is not an import gate, does not accept ground truth, and must keep
`delivery_allowed=false`, `training_started=false`, `audio_read=false`,
`audio_copied=false`, and `ground_truth_manifest_written=false`.
Run `tools/build_full_pack_raw_reference_recovery_material_preflight.py` after
that executor when external clean material is supplied. The preflight consumes
`exec.json`, reads per-action response metadata from a short response root such
as `rsp/a001.json`, checks that declared files exist under explicit allowed
source roots, validates `ground_truth_contract.verified=true`, FL Studio/user
provenance, trainable rights, and path consistency, then writes short outputs
(`preflight.json`, `items.jsonl`, `ready.jsonl`, `blocked.jsonl`,
`progress.jsonl`, `summary.md`). Ready rows mean only
`ready_for_raw_reference_recovery_verification`; they are not accepted training
ground truth, not a listen pack, and must keep `delivery_allowed=false`,
`training_started=false`, `audio_read=false`, and
`ground_truth_manifest_written=false`.

Run `tools/build_full_pack_raw_reference_recovery_material_handoff.py` after
that preflight to expose only ready recovery material as candidate roots for
the normal reference corpus intake gate. The handoff revalidates the preflight
manifest, ready rows, candidate roots, `ground_truth_manifest.json`, verified
ground-truth contract, trainable rights, FL Studio/user provenance, requested
stems, and safe metadata flags, then writes short outputs (`handoff.json`,
`items.jsonl`, `ready.jsonl`, `blocked.jsonl`, `progress.jsonl`,
`summary.md`). It is still not an import gate and must keep
`accepted_reference_rows=0`, `delivery_allowed=false`, `training_started=false`,
`audio_read=false`, and `audio_copied=false`; accepted rows are imported only
when `tools/build_full_pack_reference_corpus_intake.py` accepts the same roots
under the current full-pack gap plan.

Use `tools/build_full_pack_aris_cycle_status.py` to reconcile the current
machine gate, repair queue, full-pack loop plan, reference gap plan, reference
intake, verified-corpus validation, and training plan into one machine-readable
cycle status. It writes `status.json`, `progress.jsonl`, and `summary.md`,
keeps `delivery_allowed=false`, `release_ready=false`, and
`training_started=false`, and emits a single `next_action` such as
`collect_verified_reference_corpus` or `run_gpu_training_and_holdout_eval`. This
is the ARIS loop checkpoint; it is not a listen/result artifact.

When the cycle status says `next_action=collect_verified_reference_corpus`, run
`tools/build_full_pack_reference_collection_queue.py` with the cycle status,
reference gap plan, and latest reference intake. It writes short metadata-only
artifacts (`queue.json`, `items.jsonl`, `tpl.json`, `progress.jsonl`,
`summary.md`) that expand the remaining full-pack stem gaps into concrete
collection slots. The template accepts only user-supplied FL Studio/full-mix
ground-truth bundles with `ground_truth_contract.verified=true`, trainable
rights, and existing `source_mix.flac` plus `stems/<stem>.flac` paths. This
queue must keep `delivery_allowed=false`, `training_started=false`,
`release_ready=false`, and `presentation_status=reference_collection_queue_not_for_listen_pack`.
When raw 6s candidate evidence was built by
`tools/build_full_pack_raw_reference_bundle_worklist.py`, pass it as
`--raw-reference-worklist-json`. The queue may copy only the safe
`raw_reference_evidence` metadata into matching collection slots. That evidence
must remain fail-closed (`accepted_as_ground_truth=false`,
`requires_clean_reference_bundle=true`, `delivery_allowed=false`) and is used
only to explain which raw/listen-pack candidate still needs a clean verified
reference bundle.

Then run `tools/build_full_pack_reference_bundle_handoff.py` with the latest
collection queue. It writes short handoff artifacts (`handoff.json`,
`items.jsonl`, `tpl.json`, `progress.jsonl`, `summary.md`) and assigns stable
short bundle aliases under `refs/<alias>/` for every remaining full-pack stem
gap. If a collection item contains `raw_reference_evidence`, the handoff must
preserve that evidence unchanged as planning metadata so the next ARIS pass can
see the 6s candidate/blocker context without treating it as verified training
ground truth.

Run `tools/build_full_pack_reference_bundle_scaffold.py` with that handoff when
you need concrete local bundle directories for the next ARIS loop pass. The
scaffold writes only short directories and metadata templates such as
`refs/r001/tpl.json`; it deliberately does not write
`ground_truth_manifest.json`, `source_mix.flac`, stem audio placeholders, or any
accepted/listen result. The next preflight must still block until real verified
reference files are present.

Before feeding handoff bundles back into ARIS intake, run
`tools/build_full_pack_reference_bundle_preflight.py` with the latest
`handoff.json` and the handoff root. It writes short metadata-only artifacts
(`preflight.json`, `items.jsonl`, `ready.jsonl`, `blocked.jsonl`,
`progress.jsonl`, `summary.md`) and checks every `refs/<alias>/` bundle for
`source_mix.flac`, `stems/<stem>.flac`, `ground_truth_manifest.json`,
`ground_truth_contract.verified=true`, FL Studio/user-supplied provenance,
trainable rights, and path consistency. The preflight output lists only ready
candidate roots for the next intake/metadata loop and keeps blocked bundles as
machine-readable blockers. It must not read audio bytes, copy/decode audio,
start training, or set any delivery/listen status.

Feed that `preflight.json` to `tools/build_full_pack_aris_next_run_plan.py`
before launching the next ARIS loop. It writes short metadata-only artifacts
(`next.json`, `args.jsonl`, `blocked.jsonl`, `progress.jsonl`, `summary.md`).
If preflight has verified ready bundle roots, the planner emits the next
`run_full_pack_aris_metadata_loop.py` argument list with repeated
`--candidate-root` entries and the next short output directory. If recovery
material handoff roots are ready, pass
`--raw-reference-recovery-material-handoff-json`; the planner carries those
roots into the repeated `--candidate-root` list and also repeats them as
`--raw-reference-recovery-material-candidate-root`, so the next loop preserves
both normal reference intake and raw-recovery material routing context instead
of a manual transfer. Ordinary empty reference-bundle handoff checks
(`handoff_status_not_ready`,
`handoff_failed_checks_not_empty`, `handoff_empty`) are nonfatal only when the
recovery-material handoff itself is safe and ready; unsafe or internally
inconsistent preflight/handoff manifests still fail closed with
`next_run_status=blocked`. If no verified roots are ready, it keeps
`next_run_status=blocked_waiting_for_reference_bundles` and records blocker
counts per stem. The planner never reads audio bytes, starts training, or
creates listen/result artifacts.

After each reference intake attempt, refresh the training corpus view with
`tools/merge_generic_audio_training_examples.py`, then rerun
`tools/validate_verified_audio_training_corpus.py` and
`tools/plan_generic_audio_finetune_jobs.py`. For new ARIS artifacts, pass
`--short-output-names` to those tools so the refreshed metadata uses short
files such as `merge.json`, `rows.jsonl`, `val.json`, `plan.json`, and
`progress.jsonl`. A refresh with zero accepted reference rows must preserve the
closed state: validation remains failed on insufficient unique tracks, training
jobs remain blocked, and no listen/result artifact may be created.

To run that refresh as one repeatable ARIS step, use
`tools/run_full_pack_aris_metadata_loop.py`. It chains reference intake,
short-name recovery queue generation, project-owned permission attestation,
recovery decision ingest, combined intake refresh, merge, verified-corpus
validation, training-job planning, cycle-status refresh, reference-collection
queue rebuild with raw-reference worklist evidence, reference bundle handoff,
reference bundle preflight, raw-reference recovery material response building,
raw-reference recovery material preflight, raw-reference recovery material
handoff, next-run planning, and completion audit into short outputs
(`loop.json`, `summary.md`,
`progress.jsonl`, plus subdirectories such as `int`, `rec`, `perm`, `ing`,
`cint`, `merge`, `val`, `tr`, `cycle`, `collect`, `hand`, `rrsp`, `rpre`,
`rhand`, `pre`, `next`, and `goal`).
The runner never decodes/copies listen audio, starts training, edits model
registries, or creates listen artifacts. If recovery ingest approves rows, it
may read target files only to compute sha256 for dataset provenance, and records
that in safety/progress metadata. Recovered rows are re-filtered before the
combined intake: duplicate existing `track_id` + `stem_type` pairs stay as
forensic recovery evidence and must not reduce `remaining_by_stem`, enter the
merged training corpus, or close reference gaps. Its CLI exits nonzero until the
completion audit is machine accepted, so a refreshed blocked state cannot be
mistaken for a result.

To combine separately accepted listen packs into one aggregate validation
staging folder, use `tools/build_aggregate_short_listen_pack.py`. It accepts source manifests as
`--source-manifest ALIAS=PATH`, validates each alias/path segment against the
short-name policy, copies audio without modification, writes `pack.json` and
`coverage.json`, and fails closed on release-ready, publishable, threshold
relaxed, registry-edited, missing-audio, or long-name sources.
Successful aggregate packs are not accepted listen/result artifacts. They must
write `delivery_status=aggregate_validation_pending`,
`delivery_allowed=false`,
`presentation_status=aggregate_validation_pending_not_for_listen_pack`, and
`aggregate_validation_ready=true` until validation, semantic review, minus
audit, and `tools/build_machine_stem_acceptance_gate.py` promote them.
When the aggregate pack is intended to represent the full 6s stem pack, pass
`--require-full-pack-stems` to the builder, validator, and review packet runner.
That contract requires `vocal`, `minus`, `drums`, `bass`, `other`, `guitar`,
and `piano`; a `vocal`/`minus` subset remains a diagnostic preview only and must
stay blocked for full-pack delivery/listen presentation.
If the source is a registered stem manifest, start the contract at
`tools/build_stem_manifest_audio_pack.py --require-full-pack-stems`; the emitted
`pack.json.required_stems` is then enforced by the aggregate validator and
review packet runner even when no later CLI flag is supplied.

Before calling any aggregate pack usable for user listening, run
`tools/validate_aggregate_short_listen_pack.py` and
`tools/audit_short_listen_goal_status.py`. The validator recomputes copied audio
sha256/metadata and checks promised metadata artifacts. The audit records which
goal requirements are machine-proved and keeps the goal open when the human
listen verdict is missing. It also records that 6s stem coverage is not semantic
ground truth unless separately verified.
Human review materialization starts only after
`tools/build_machine_stem_acceptance_gate.py` emits an accepted `gate.json` for
the same staged aggregate pack. The feedback template, review handoff,
handoff validator, and feedback manifest reject stale accepted aggregate packs
and non-aggregate subset/repair/candidate listen-file packs.

For multiple holdout folds, use
`tools/aggregate_train_ready_audio_model_decisions.py`. It combines child
decision manifests and applies the same strict success-rate rule to the total
passed/evaluated holdout count. Child release, delivery, promotion, or failed
checks block the aggregate decision.

## Docker Commands

```bash
python /app/tools/validate_verified_audio_training_corpus.py \
  --audio-training-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/merged_audio_training_v1/merged_audio_training_examples.jsonl \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/verified_audio_training_corpus_v1 \
  --min-unique-tracks 3
```

```bash
python /app/tools/build_train_ready_audio_model_decision.py \
  --matrix-manifest-json /app/lab_artifacts/generic_finetuning_playbook_20260613/shared_vocal_adapter_matrix_v1/generic_audio_finetune_matrix_manifest.json \
  --holdout-rows-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/shared_vocal_adapter_matrix_v1/configs/001_frame_balanced_smoke/holdout/generic_audio_finetune_holdout_rows.jsonl \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/train_ready_decision_v1 \
  --min-success-rate 0.90
```

```bash
python /app/tools/build_candidate_family_blend_locked_split.py \
  --audio-training-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/multi_candidate_conditioned_vocal_training_view_v13_strong_scnet_melband_all_verified_no_melody/candidate_conditioned_audio_training_examples.jsonl \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/candidate_family_blend_locked_split_v33_001 \
  --stem vocal \
  --family scnet \
  --family melband \
  --holdout-track-id nothing-can-wait \
  --holdout-track-id 3404a02f61d017a1ea0a3776f7db6f3abf7cd1969133e9bc87e7f879f6f71273
```

```bash
python /app/tools/build_candidate_family_blend_listen_pack.py \
  --train-ready-decision-manifest-json /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/train_ready_decision_vocal_candidate_family_blend_v33_locked_multi_holdout_001/train_ready_audio_model_decision_manifest.json \
  --holdout-rows-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/vocal_candidate_family_blend_matrix_v33_locked_multi_holdout_001/configs/001_v32_scnet_melband_blend_step005/holdout/candidate_family_blend_holdout_rows.jsonl \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/v35_holdout \
  --stem vocal
```

```bash
python /app/tools/build_candidate_family_full_length_listen_pack.py \
  --audio-training-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/multi_candidate_conditioned_vocal_training_view_v13_strong_scnet_melband_all_verified_no_melody/candidate_conditioned_audio_training_examples.jsonl \
  --adapter-json /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/vocal_candidate_family_blend_matrix_v33_locked_multi_holdout_001/configs/001_v32_scnet_melband_blend_step005/train/candidate_family_blend_adapter.json \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/v35_voc_full \
  --stem vocal
```

```bash
python /app/tools/build_music_intelligence_stem_listen_pack.py \
  --music-intelligence-manifest /app/lab_artifacts/autonomous_runs/arch_1a_fresh_music_manifest_001/music_intelligence_input_manifest.json \
  --music-intelligence-manifest /app/lab_artifacts/autonomous_runs/stem_3_household_goods_conditional_route_001/music_intelligence_input_manifest.json \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/v35_stems \
  --required-stem-type vocal \
  --required-stem-type minus
```

Run these commands inside Docker only.
