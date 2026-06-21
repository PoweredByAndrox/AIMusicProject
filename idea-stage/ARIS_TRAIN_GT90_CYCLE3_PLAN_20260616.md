# ARIS Train >90 Cycle 3 Plan

## Source-of-truth checkpoint

Cycle 2 and the guarded Cycle 3 preflight confirm that the current shared
teacher-mask adapter is not train-ready:

- aggregate LOTO decision:
  `/app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/aggregate_train_ready_decision_vocal_loto_v1/aggregate_train_ready_audio_model_decision_manifest.json`
- aggregate decision: `train_result_not_ready`
- aggregate success rate: `0.0`
- release_ready: `false`
- thresholds_relaxed: `false`

The high-KNN experiment is now observable and explicitly rejected instead of
running as a blind long job:

- guarded holdout:
  `/app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/vocal_shared_adapter_matrix_v3_high_knn_holdout_ncw_guard_check/generic_audio_finetune_holdout_manifest.json`
- failed check: `source_feature_memory_cost_guard`
- estimated distance ops: `12269819334240`
- max distance ops: `50000000000`
- candidate audio created: `false`

## Brainstormed options

1. **Rerun larger high-KNN with a longer timeout.**
   - Decision: reject.
   - Reason: it repeats an unbounded `torch.cdist` route and violates the
     observability/accelerator rule.

2. **Lower SDR thresholds or accept improvement-only rows.**
   - Decision: reject.
   - Reason: threshold relaxation is explicitly forbidden and would fake the
     `>90%` TRAIN result.

3. **Bounded source-feature memory: deterministic projection plus memory-frame cap.**
   - Decision: implement next.
   - Reason: it keeps the existing teacher-mask adapter family, records the
     route in checkpoint config, bounds application cost before holdout, and can
     be evaluated by the existing train-ready decision contract.

4. **Base-separator-conditioned correction adapter.**
   - Decision: defer.
   - Reason: likely stronger, but wider blast radius because it needs candidate
     generation as a training input. Use if bounded memory still fails.

## Cycle 3 implementation contract

Add two explicit config fields to the existing teacher-mask adapter route:

- `source_feature_memory_projection_dim`
- `source_feature_memory_max_frames`

Rules:

- no fallback route;
- no threshold relaxation;
- no release/listen artifact;
- no model registry edit;
- deterministic projection from the training seed;
- memory row cap stored in checkpoint metadata and manifest config;
- holdout application cost remains bounded by
  `max_source_feature_memory_distance_ops`;
- if the bounded route fails objective holdout, it remains a failed training
  artifact.

## Validation plan

1. Unit tests for projection/cap metadata and application guard.
2. Docker regression for teacher-mask, shared adapter, holdout, and matrix tests.
3. Run a small bounded projected-memory vocal matrix on the verified corpus.
4. Build train-ready decision only from passed matrix artifacts.
5. If success rate is not strictly above `0.90`, rerun ARIS brainstorm and
   move to the base-separator-conditioned correction adapter.
