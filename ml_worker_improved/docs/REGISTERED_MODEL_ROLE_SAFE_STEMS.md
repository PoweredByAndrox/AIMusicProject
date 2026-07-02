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
- `stems.minus.role=product_alias` and `stems.minus.alias_of=instrumental`
  when the underlying separator emits an instrumental stem for the
  listener-facing minus product
- `stems.minus.product_audio_path`
- `stems.minus.model_output_audio_path`

For instrumental checkpoints, new generated run directories materialize both:

- `instrumental.flac` - canonical model output file;
- `minus.flac` - product-role alias used by downstream listen/pack/judge paths.

The two files are expected to have identical audio bytes unless a later
explicit product-stage repair creates a separate minus candidate with its own
manifest provenance.

This is naming/provenance-only. It does not rename old artifact files, and the
product alias file does not change audio bytes or turn a diagnostic candidate
into a listen result. Quality gates, vocal-leak checks, detail/damage audits,
and delivery flags still decide whether a minus can be published.
