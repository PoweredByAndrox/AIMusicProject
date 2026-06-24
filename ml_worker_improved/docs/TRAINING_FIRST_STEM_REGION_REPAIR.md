# Training-First Stem Region Repair

This contract moves repeated stem defects into training surfaces instead of
treating blocked candidates as the product.

## Runtime Contract

- `run_fully_automatic_stem_search.py` still blocks unsafe delivery.
- Every blocked run writes a root-cause/mining queue under
  `training/root_cause/`.
- Queue rows are evidence only: `delivery_allowed=false` and
  `presentation_status=training_mining_only`.
- Repeated blockers are classified as one of:
  `model_gap`, `routing_gap`, `repair_gap`, `infra_gap`, or `data_gap`.

## Repair Model

`tools/stem_region_repair_model.py` defines `StemRegionRepairNet`, a small
stereo Conv1d patch model.

Input channels:

- bad stem left/right;
- mix left/right;
- localized region mask;
- issue-type scalar.

Output:

- stereo waveform delta, multiplied by the region mask.

This means clean regions outside the defect mask are not edited by the apply
stage.

## Ground Truth Rules

- `synthetic_clean`, `reference_verified`, and `human_verified` may be ground
  truth.
- `generated_separator`, `pseudo_teacher`, and `model_generated` cannot be
  ground truth and cannot drive promotion.
- Blocked candidates are mining evidence only until a verified or synthetic
  target is built.

## CLIs

- `build_training_first_root_cause_queue.py`: converts blocked run candidates
  into root-cause/mining rows.
- `build_stem_region_hardcase_pack.py`: copies selected queue rows into a
  short-path hard-case pack with bad audio, score evidence, and explicit
  `data_gap` until a verified target exists.
- `build_stem_region_synthetic_pretrain_dataset.py`: converts hard-case issue
  metadata into synthetic clean/dirty repair rows for pretraining. It does not
  self-label hard-case bad audio as target.
- `build_stem_region_repair_dataset.py`: builds synthetic repair rows from a
  clean/reference stem by injecting a localized defect.
- `train_stem_region_repair_model.py`: trains one focused patch checkpoint for
  one defect family.
- `apply_stem_region_repair.py`: writes a repaired diagnostic patch and manifest
  with `delivery_allowed=false`.
- `apply_stem_region_repair_pack.py`: applies one issue-matched repair checkpoint
  to hard-case rows and writes before/after diagnostics without delivery.

## Current V1 Scope

V1 is deliberately small:

- one focused row per checkpoint;
- localized 1-3 second region repair;
- vocal-leak and stereo-collapse synthetic defects;
- no delivery promotion from the repair apply stage.

Delivery promotion needs a separate locked validation contract.
