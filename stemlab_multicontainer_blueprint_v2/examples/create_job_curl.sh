curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "track_name": "Demo Vocals Stem",
    "input_path": "/app/data/inputs/demo_vocals.wav",
    "preset_name": "lyrics_sync_from_vocals",
    "notes": "real local lyrics sync from curl",
    "enable_lyrics": true,
    "lyrics_language": "en",
    "lyrics_reference_path": "/app/data/inputs/demo_lyrics.txt"
  }'
