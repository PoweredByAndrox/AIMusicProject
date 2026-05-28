curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "track_name": "Demo Track",
    "input_path": "/app/data/inputs/demo.wav",
    "preset_name": "fine_grained_14stem",
    "notes": "dry-run from curl"
  }'
