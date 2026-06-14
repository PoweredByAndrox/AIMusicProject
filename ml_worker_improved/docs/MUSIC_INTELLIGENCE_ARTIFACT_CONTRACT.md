# Music Intelligence Artifact Contract

Phase ARCH-1 prepares the autonomous no-reference runtime for a future music
intelligence layer. It does not run melody transcription, recommendation
indexing, separator training, model downloads, threshold relaxation, cleanup,
debleed, denoise, publishing, or `release_ready`.

## Runtime Artifact

Every autonomous runtime run writes:

```text
music_intelligence_input_manifest.json
```

The file is an input contract for future melody, harmony, rhythm, motif, phrase,
and recommendation analysis. Dirty or blocked stems are included for provenance,
but they are never treated as ground truth.

Top-level fields:

```json
{
  "schema_version": 1,
  "run_id": "song_001",
  "track_id": "song",
  "run_dir": "/app/lab_artifacts/autonomous_runs/song_001",
  "runtime_status": "preview_ready",
  "analysis_ready_stems_count": 2,
  "blocked_stems_count": 0,
  "future_analysis_supported": false,
  "melody_transcription_ran": false,
  "recommendation_index_updated": false,
  "release_ready": false,
  "stems": []
}
```

Per-stem fields:

```json
{
  "track_id": "song",
  "run_id": "song_001",
  "candidate_id": "candidate-a",
  "stem_id": "song_001__candidate-a__vocal__hash",
  "stem_type": "vocal",
  "instrument_hint": "vocal",
  "audio_path": "/app/path/vocals.flac",
  "normalized_audio_path": "/app/path/vocals.flac",
  "sample_rate": 44100,
  "channels": 2,
  "duration_sec": 180.0,
  "quality_status": "accepted",
  "blockers": [],
  "analysis_allowed": true,
  "analysis_block_reason": "",
  "source_separator_config": "htdemucs_ft",
  "source_candidate_family": "local_registered",
  "pair_compatibility_status": "pair_compatible",
  "release_ready": false
}
```

Allowed `stem_type` values:

- `vocal`
- `minus`
- `instrumental`
- `piano`
- `bass`
- `drums`
- `other`
- `unknown`

Eligibility rules:

- Only safe accepted stems can set `analysis_allowed=true`.
- Blocked stems stay in the manifest with `analysis_allowed=false`.
- Any `release_ready=true` source flag blocks analysis and is rewritten to
  `release_ready=false`.
- Runtime statuses other than `preview_ready` or `accepted_for_listen_pack`
  block analysis.
- Dirty stems are provenance inputs only, not training targets or ground truth.

## Run Manifest Hook

`run_manifest.json` contains:

```json
{
  "music_intelligence": {
    "manifest_written": true,
    "manifest_path": "/app/lab_artifacts/autonomous_runs/song_001/music_intelligence_input_manifest.json",
    "analysis_ready_stems_count": 2,
    "blocked_stems_count": 0,
    "future_analysis_supported": false,
    "melody_transcription_ran": false,
    "recommendation_index_updated": false,
    "release_ready": false
  }
}
```

`runtime_summary.md` mirrors these counts and false future hooks.

## Future Schema Stubs

These files are documented for later phases only. ARCH-1 does not create them
from audio and does not run transcription.

### stem_note_events.jsonl

Fields:

- `track_id`
- `stem_id`
- `instrument_role`
- `note_name`
- `midi_pitch`
- `start_sec`
- `end_sec`
- `duration_sec`
- `velocity_proxy`
- `confidence`
- `source_transcription_route`

### stem_pitch_contours.jsonl

Fields:

- `track_id`
- `stem_id`
- `instrument_role`
- `start_sec`
- `end_sec`
- `hop_sec`
- `pitch_hz`
- `voicing_confidence`
- `source_transcription_route`

### stem_melody_motifs.jsonl

Fields:

- `motif_id`
- `track_id`
- `stem_id`
- `instrument_role`
- `interval_pattern`
- `rhythm_pattern`
- `scale_degree_pattern`
- `contour`
- `start_sec`
- `end_sec`
- `repeat_count`
- `confidence`

### stem_chord_events.jsonl

Fields:

- `track_id`
- `stem_id`
- `instrument_role`
- `chord_name`
- `root_midi_pitch`
- `start_sec`
- `end_sec`
- `confidence`
- `source_transcription_route`

### stem_rhythm_features.jsonl

Fields:

- `track_id`
- `stem_id`
- `instrument_role`
- `tempo_bpm`
- `meter`
- `onset_times_sec`
- `syncopation_score`
- `confidence`

### stem_instrument_roles.jsonl

Fields:

- `track_id`
- `stem_id`
- `instrument_role`
- `instrument_hint`
- `role_confidence`
- `active_segments`

### track_music_fingerprint.json

Fields:

- `track_id`
- `primary_melody_stem_id`
- `phrase_structure`
- `motif_ids`
- `analysis_quality_score`
- `release_ready=false`

### track_recommendation_features.json

Fields:

- `track_id`
- `global_melody_embedding`
- `vocal_melody_embedding`
- `piano_melody_embedding`
- `orchestral_melody_embedding`
- `bassline_embedding`
- `chord_progression_embedding`
- `rhythm_embedding`
- `motif_embedding_ids`
- `analysis_quality_score`
- `release_ready=false`

## Commands

Build a manifest for an existing run:

```bash
docker compose run --rm --no-deps ml-worker python tools/build_music_intelligence_input_manifest.py \
  --run-dir /app/lab_artifacts/autonomous_runs/song_001
```

Validate a manifest and future schema examples:

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_music_analysis_ready_stems.py \
  --manifest /app/lab_artifacts/autonomous_runs/song_001/music_intelligence_input_manifest.json
```
