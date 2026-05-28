from __future__ import annotations

import json
import sys
from pathlib import Path

from app.main import align_reference_with_stable, transcribe_with_whisperx


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "missing_mode"}))
        return 1

    mode = sys.argv[1].strip().lower()
    payload = json.loads(sys.stdin.read() or "{}")

    if mode == "transcribe":
        result, metadata, command_preview = transcribe_with_whisperx(
            Path(payload["audio_path"]),
            payload.get("language"),
        )
    elif mode == "align":
        result, metadata, command_preview = align_reference_with_stable(
            Path(payload["audio_path"]),
            Path(payload["reference_path"]),
            payload.get("language"),
        )
    else:
        print(json.dumps({"error": f"unsupported_mode:{mode}"}))
        return 1

    print(
        json.dumps(
            {
                "payload": result,
                "metadata": metadata,
                "command_preview": command_preview,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
