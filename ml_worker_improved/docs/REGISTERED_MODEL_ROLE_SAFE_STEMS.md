# Registered Model Role-Safe Stems

Registered model artifacts may use legacy stem filenames from the underlying
separator, such as `instrumental.flac` for a user-facing minus target or
`vocals.flac` for a residual companion. New registered-model manifests must not
ask downstream tools or listeners to infer the requested role from the filename
alone.

When `tools/run_registered_model_candidate.py` materializes a requested target,
the manifest records role-safe metadata:

- `target_stem_artifact.canonical_target_stem`
- `target_stem_artifact.user_facing_role`
- `target_stem_artifact.manifest_stem_name`
- `target_stem_artifact.manifest_stem_alias_of` when the source stem name is a
  legacy alias such as `instrumental`
- `stems.<target>.role=target`
- `stems.<target>.canonical_stem`
- `stems.<target>.user_facing_role`

This is metadata-only. It does not rename old artifact files and does not turn a
diagnostic candidate into a listen result. Quality gates, vocal-leak checks,
detail/damage audits, and delivery flags still decide whether a minus can be
published.
