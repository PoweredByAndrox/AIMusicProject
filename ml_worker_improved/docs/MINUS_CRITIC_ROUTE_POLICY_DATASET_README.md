# Minus Critic Route Policy Dataset

`build_minus_critic_route_policy_dataset.py` converts validated `minus_critic`
score rows into `stem_route_policy_row.v1` training evidence.

This is a shadow-only bridge. It does not train separator weights, does not
change runtime gates, and cannot publish listen artifacts. Its purpose is to
let the route-policy layer learn that a minus route pattern associated with
failed exact-reference windows should be quarantined or ranked lower in later
search runs.

Contract:

- input: `scored_minus_critic_windows.jsonl` from a validated target audio error
  critic run;
- failed minus labels (`true_label = 0`) map to `quarantine_route`;
- passing minus labels (`true_label = 1`) map to `block_unknown`, not
  `prioritize_route`, because prioritization still requires counterfactual
  replay evidence;
- generated policy rows remain `shadow_only`, `delivery_allowed = false`,
  `usable_for_promotion = false`, and `runtime_gates_changed = false`;
- source row ids and candidate ids are hashed into short generated ids so raw
  track titles do not become artifact identifiers.

Typical flow:

```bash
python -m tools.build_minus_critic_route_policy_dataset \
  --scores-jsonl artifacts/loop/it106/critics/r002/scores/scored_minus_critic_windows.jsonl \
  --out artifacts/loop/it107/policy/r001/dataset

python -m tools.train_stem_route_policy_model \
  --rows-jsonl artifacts/loop/it107/policy/r001/dataset/rows.jsonl \
  --out artifacts/loop/it107/policy/r001/model

python -m tools.validate_stem_route_policy_checkpoint \
  --checkpoint artifacts/loop/it107/policy/r001/model/model.pt \
  --rows-jsonl artifacts/loop/it107/policy/r001/dataset/rows.jsonl \
  --out artifacts/loop/it107/policy/r001/validation
```

The resulting checkpoint is still a `stem_route_policy_checkpoint.v1` shadow
model. It may be used by `run_fully_automatic_stem_search` only as policy
evidence for shadow/rank routing, not as an accepted delivery gate.
