# Registered Model Role-Safe Stems

Registered model artifacts may use legacy stem filenames from the underlying
separator, such as `instrumental.flac` for a user-facing minus target or
`vocals.flac` for a residual companion. New registered-model manifests must not
ask downstream tools or listeners to infer the requested role from the filename
alone.

When `tools/run_registered_model_candidate.py` materializes a requested target,
the manifest records role-safe metadata:

- `target_stem`
- `target_product_stem`
- `target_audio_path`
- `target_product_audio_path`
- `target_model_output_audio_path`
- `target_stem_artifact.canonical_stem_type`
- `target_stem_artifact.user_facing_stem_role`
- `target_stem_artifact.manifest_stem_name`
- `target_stem_artifact.product_audio_path`
- `target_stem_artifact.model_output_audio_path`
- `stems.instrumental.role=target` for the canonical model output
- `product_aliases.minus.source_stem=instrumental` when the underlying
  separator emits an instrumental stem for the listener-facing minus product
- `product_aliases.minus.product_audio_path`
- `product_aliases.minus.model_output_audio_path`
- `target_product_audio_materialized_in_run=false` for metadata-only product
  aliases

For instrumental checkpoints, new generated low-level model `run/` directories
materialize only the native model stem:

- `instrumental.flac` - canonical model output file.

They must not also materialize `run/minus.flac` as a hardlink/copy of
`instrumental.flac`. The product role `minus` is represented as metadata in
`product_aliases.minus` and may be materialized only at a product boundary such
as a candidate `stems/minus.flac`, listen pack, or explicit product-stage repair
with its own manifest provenance. If a legacy source manifest contains
`stems.minus.role=product_alias`, consumers must verify
`alias_of=instrumental` and prefer the canonical `instrumental` model-output
path; aliases that point to `vocals` or any other source are invalid.

This is naming/provenance-only. It does not rename old artifact files, and a
metadata or product-boundary alias does not change audio bytes or turn a
diagnostic candidate into a listen result. Quality gates, vocal-leak checks,
detail/damage audits, and delivery flags still decide whether a minus can be
published.
