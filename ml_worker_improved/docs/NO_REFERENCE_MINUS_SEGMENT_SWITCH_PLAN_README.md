# No-Reference Minus Segment Switch Plan

`tools/plan_no_reference_minus_segment_switch.py` builds a metadata-only plan
for `render_no_reference_minus_segment_switch_candidate.py` from existing
instrumental self-judge manifests and, optionally, the minus score matrix.

The planner does not render audio, copy audio, run inference, train, add
models, add cleanup/debleed/denoise routes, relax thresholds, or mark anything
release-ready. It is a matrix-driven routing artifact.

It selects the base candidate by:

1. explicit `--base-candidate-id`, when provided;
2. `--candidate-score-matrix-json`, when a matrix row maps to an available
   self-judge candidate;
3. the existing self-judge heuristic from the minus delivery pipeline.

The plan is `ready_for_render` only when every blocker segment on the selected
base has a safe alternate segment with the same window and all Docker-visible
source, vocal, and minus paths exist. Otherwise it records unresolved segments
or missing audio inputs and leaves `render_allowed=false`.

Output remains diagnostic:

- `delivery_status=blocked`
- `delivery_allowed=false`
- `presentation_status=diagnostic_plan_not_for_listen_pack`
- `release_status=not_release_ready`
- `release_ready=false`
- `gpu_inference_skipped=true`

Run:

```bash
docker compose run --rm --no-deps ml-worker python tools/plan_no_reference_minus_segment_switch.py \
  --self-judge-manifest /app/lab_artifacts/.../candidate_a/instrumental_self_judge_manifest.json \
  --self-judge-manifest /app/lab_artifacts/.../candidate_b/instrumental_self_judge_manifest.json \
  --candidate-score-matrix-json /app/lab_artifacts/.../no_reference_minus_candidate_score_matrix_manifest.json \
  --render-output-dir /app/lab_artifacts/.../minus_segment_switch_candidate \
  --output-json /app/lab_artifacts/.../minus_segment_switch_plan.json \
  --summary-md /app/lab_artifacts/.../minus_segment_switch_plan_summary.md
```

When `render_allowed=true`, `recommended_render_args` contains the exact
metadata-only next command arguments for the render step. The rendered candidate
still remains blocked until post-render minus self-judge, result judge,
selector, and artifact status contract accept it.
