from __future__ import annotations

import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import tools.run_fully_automatic_stem_search as stem_search
import tools.resident_separator_executor as resident_executor
import tools.serve_persistent_separator as separator_server
from tools.stem_forensic_crosscheck import score_cross_stem_forensic_matrix
from tools.persistent_separator_backend import (
    MODE_PERSISTENT_SERVER,
    PersistentSeparatorClient,
    PersistentSeparatorRejected,
    PersistentSeparatorUnavailable,
    SeparatorBackendConfig,
    separator_cache_key,
)
from tools.run_fully_automatic_stem_search import (
    _candidate_contamination_checks,
    _run_target_specific_commands,
    apply_minus_vocal_debleed,
    execute_recipe_candidate,
)
from tools.resident_separator_executor import (
    RESIDENT_FIXTURE_ROUTE_ID,
    AudioSeparatorResidentExecutor,
    MelBandResidentExecutor,
    resident_executor_health,
    reset_resident_separator_executors_for_tests,
)
from tools.stem_contamination_common import STEMS


SAMPLE_RATE = 16_000


def _write_stem(path: Path, frames: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(frames, dtype=np.float32), SAMPLE_RATE)


def test_separator_cache_key_is_windows_safe_and_hides_raw_track_path() -> None:
    key = separator_cache_key(
        route_id="audio_separator_bs_roformer_vocals",
        source_audio_path="D:/AIMusicProject/testing_stuff_music/ytmp3free.cc_karak-yani-rainbow-youtubemp3free.org.mp3",
        command=["python", "tools/export_audio_separator_vocal_full.py", "{window_mix}"],
        stems=("vocals",),
        params={"device": "cuda"},
    )

    assert key.startswith("sep_")
    assert len(key) <= 32
    assert "karak" not in key.lower()
    assert "rainbow" not in key.lower()
    assert "ytmp3" not in key.lower()
    assert "/" not in key
    assert "\\" not in key


def test_persistent_separator_config_forbids_cpu_fallback_when_gpu_required() -> None:
    with pytest.raises(ValueError, match="cpu_fallback_forbidden"):
        SeparatorBackendConfig(
            mode=MODE_PERSISTENT_SERVER,
            server_url="http://127.0.0.1:9",
            require_gpu=True,
            device="cpu",
        ).validated()


def test_persistent_separator_config_allows_mps_only_for_persistent_backend() -> None:
    config = SeparatorBackendConfig(
        mode=MODE_PERSISTENT_SERVER,
        server_url="http://127.0.0.1:9",
        require_gpu=True,
        device="mps",
    ).validated()

    assert config.device == "mps"
    with pytest.raises(ValueError, match="mps_requires_persistent"):
        SeparatorBackendConfig(mode="subprocess", require_gpu=True, device="mps").validated()


def test_mps_persistent_backend_preflight_delegates_without_nvidia_smi(monkeypatch) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("nvidia-smi should not be called for mps persistent backend")

    monkeypatch.setattr(stem_search.subprocess, "run", fail_if_called)
    config = SeparatorBackendConfig(
        mode=MODE_PERSISTENT_SERVER,
        server_url="http://127.0.0.1:8793",
        require_gpu=True,
        device="mps",
    ).validated()

    preflight = stem_search._collect_separator_accelerator_preflight(
        require_gpu=True,
        min_free_mib=4096,
        separator_backend_config=config,
    )

    assert preflight["status"] == "delegated_to_persistent_backend"
    assert preflight["accelerator"] == "mps"


def test_persistent_mps_minus_pair_recipe_uses_dedicated_minus_route_without_stem_sum() -> None:
    recipe = stem_search.build_persistent_mps_minus_pair_recipe(
        contamination_model_manifest="artifacts/scd/m001/manifest.json",
    )

    assert recipe["available"] is True
    assert recipe["profile"] == stem_search.PERSISTENT_MPS_MINUS_PAIR_PROFILE
    assert [candidate["candidate_id"] for candidate in recipe["candidates"]] == ["c001", "c002", "c003", "c004"]
    candidate = recipe["candidates"][0]
    assert candidate["diagnostic_only"] is True
    assert candidate["required_stems"] == ["vocals", "minus"]
    assert candidate["minus_source"] == "minus_lane"
    assert candidate["minus_rebuilt_from_non_vocal_stems"] is False
    routes = candidate["target_specific_commands"]
    assert [route["role"] for route in routes] == ["vocals", "minus"]
    assert {route["route_id"] for route in routes} == {stem_search.PERSISTENT_MPS_MINUS_PAIR_ROUTE_ID}
    assert routes[1]["output_map"] == {"minus": "instrumental"}
    assert all(route["server_stems"] == ["vocals", "instrumental"] for route in routes)
    assert routes[0]["shared_separator_cache_id"] == routes[1]["shared_separator_cache_id"]
    checks, hints_used = _candidate_contamination_checks(
        {
            **candidate,
            "contamination_model_manifest": recipe["contamination_model_manifest"],
        }
    )
    assert hints_used is True
    assert candidate["contamination_check_mode"] == "explicit_only"
    assert candidate["use_default_contamination_model_manifest"] is False
    assert checks == [{"tested_stem": "minus", "foreign_source": "vocals"}]
    assert candidate["cross_stem_forensic_gate"]["pairs"] == [{"tested_stem": "minus", "source_stem": "vocals"}]
    assert recipe["candidates"][1]["vocal_projection_debleed"]["target_stems"] == ["minus"]
    assert recipe["candidates"][2]["post_minus_vocal_debleed"]["target_stems"] == ["minus"]
    assert recipe["candidates"][3]["post_minus_vocal_debleed"]["strength"] == 1.0


def test_minus_vocal_debleed_can_target_dedicated_minus_lane(tmp_path: Path) -> None:
    frames = 512
    t = np.arange(frames, dtype=np.float32) / float(SAMPLE_RATE)
    vocal_band = (0.05 * np.sin(2.0 * np.pi * 900.0 * t)).astype(np.float32)
    low_band = (0.10 * np.sin(2.0 * np.pi * 110.0 * t)).astype(np.float32)
    paths = {
        "vocals": str(tmp_path / "vocals.flac"),
        "minus": str(tmp_path / "minus.flac"),
    }
    sf.write(paths["vocals"], np.zeros(frames, dtype=np.float32), SAMPLE_RATE)
    sf.write(paths["minus"], low_band + vocal_band, SAMPLE_RATE)

    result = apply_minus_vocal_debleed(
        stem_paths=paths,
        config={
            "target_stems": ["minus"],
            "strength": 1.0,
            "low_hz": 700.0,
            "high_hz": 1200.0,
            "passes": 1,
        },
    )

    cleaned_minus, _ = sf.read(paths["minus"], dtype="float32", always_2d=False)
    repaired_vocals, _ = sf.read(paths["vocals"], dtype="float32", always_2d=False)
    assert result["applied"] is True
    assert result["target_stems"] == ["minus"]
    assert result["per_stem"]["minus"]["moved_rms"] > 0.0
    assert result["per_stem"]["minus"]["removed_band_ratio"] > 0.9
    assert result["per_stem"]["minus"]["output_band_rms"] < result["per_stem"]["minus"]["input_band_rms"] * 0.05
    assert float(np.sqrt(np.mean(np.square(cleaned_minus.astype(np.float64))))) < float(
        np.sqrt(np.mean(np.square((low_band + vocal_band).astype(np.float64))))
    )
    assert float(np.sqrt(np.mean(np.square(repaired_vocals.astype(np.float64))))) > 0.0


def test_explicit_forensic_pairs_fail_closed_when_empty(tmp_path: Path) -> None:
    result = score_cross_stem_forensic_matrix(
        stem_paths={},
        output_dir=tmp_path / "forensic",
        config={"enabled": True, "pairs": []},
    )

    assert result["check_mode"] == "explicit_pairs"
    assert result["verdict"] == "blocked_unknown"
    assert result["blocker_reasons"] == ["missing_forensic_pairs"]


def test_persistent_mps_minus_pair_run_bypasses_demucs_availability_and_uses_shared_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mix = tmp_path / "mix.flac"
    _write_stem(mix, frames=128)
    calls: list[dict[str, object]] = []

    def fail_if_cuda_preflight_is_called(*_args, **_kwargs):
        raise AssertionError("cuda/nvidia-smi preflight should not be called for persistent mps route")

    class FakePersistentSeparatorClient:
        def __init__(self, server_url: str, *, request_timeout_sec: float = 30.0) -> None:
            self.server_url = server_url
            self.request_timeout_sec = request_timeout_sec

        def separate(self, **kwargs: object) -> dict[str, object]:
            calls.append(dict(kwargs))
            source_audio, sample_rate = sf.read(
                Path(str(kwargs["source_audio_path"])),
                dtype="float32",
                always_2d=False,
            )
            out_stems = Path(str(kwargs["output_dir"])) / "stems"
            out_stems.mkdir(parents=True, exist_ok=True)
            sf.write(out_stems / "vocals.flac", np.zeros_like(source_audio, dtype=np.float32), sample_rate)
            sf.write(out_stems / "instrumental.flac", source_audio, sample_rate)
            return {
                "status": "ok",
                "stems_dir": str(out_stems),
                "cache_hit": False,
                "actual_device": "mps",
            }

    monkeypatch.setattr(stem_search, "_collect_cuda_preflight", fail_if_cuda_preflight_is_called)
    monkeypatch.setattr(stem_search, "PersistentSeparatorClient", FakePersistentSeparatorClient)

    result = stem_search.run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "run",
        window_sec=1.0,
        max_candidates=1,
        require_gpu=True,
        no_artifact_rag=True,
        demucs_available=False,
        separator_backend=MODE_PERSISTENT_SERVER,
        separator_server_url="http://127.0.0.1:8793",
        separator_device="mps",
    )

    assert result["gpu_preflight"]["status"] == "delegated_to_persistent_backend"
    assert result["verdict"] != "blocked_no_available_separator"
    assert result.get("unavailable_reason") != "missing_demucs_cli"
    assert result["candidate_count"] == 1
    assert len(calls) == 1
    assert calls[0]["device"] == "mps"
    assert calls[0]["stems"] == ["vocals", "instrumental"]
    candidate = result["candidates"][0]
    assert candidate["required_stems"] == ["vocals", "minus"]
    assert candidate["stem_paths"]["minus"].endswith("minus.flac")
    assert candidate["minus_rebuilt_from_non_vocal_stems"] is False
    assert candidate["delivery_allowed"] is False
    assert candidate["candidate_untrusted"] is True
    assert candidate["contamination_score"]["check_mode"] == "explicit_only"
    assert candidate["contamination_score"]["required_check_count"] == 1
    assert candidate["cross_stem_forensic_score"]["check_mode"] == "explicit_pairs"
    assert candidate["cross_stem_forensic_score"]["verdict"] == "auto_accept"
    assert not any(
        str(reason).startswith("missing_forensic_")
        for reason in candidate["cross_stem_forensic_score"]["blocker_reasons"]
    )
    routes = candidate["target_specific_composition"]["routes"]
    assert routes[1]["separator_backend"]["cache_hit"] is True


def test_persistent_separator_client_reports_unavailable() -> None:
    client = PersistentSeparatorClient("http://127.0.0.1:9", request_timeout_sec=0.1)

    with pytest.raises(PersistentSeparatorUnavailable):
        client.separate(
            route_id="missing",
            source_audio_path="x.wav",
            output_dir="out",
            device="cuda",
            stems=("vocals",),
            batch_id="b001",
            command=["missing"],
            cache_key="sep_missing",
            timeout_sec=1,
            require_gpu=True,
        )


def test_persistent_separator_client_reports_server_rejection() -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
            response = json.dumps({"status": "blocked", "reason": "missing_required_stems"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = PersistentSeparatorClient(f"http://127.0.0.1:{server.server_port}", request_timeout_sec=1.0)
        with pytest.raises(PersistentSeparatorRejected, match="missing_required_stems"):
            client.separate(
                route_id="r001",
                source_audio_path="x.wav",
                output_dir="out",
                device="cuda",
                stems=("vocals",),
                batch_id="b001",
                command=["fixture"],
                cache_key="sep_reject",
                timeout_sec=1,
                require_gpu=True,
            )
    finally:
        server.shutdown()
        thread.join(timeout=2.0)


def test_persistent_backend_auto_request_timeout_exceeds_route_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, float] = {}

    class FakeClient:
        def __init__(self, server_url: str, *, request_timeout_sec: float) -> None:
            captured["request_timeout_sec"] = request_timeout_sec
            captured["server_url_set"] = 1.0 if server_url == "http://separator-server:8791" else 0.0

        def separate(self, **kwargs: object) -> dict[str, str]:
            stems_dir = Path(str(kwargs["output_dir"])) / "stems"
            _write_stem(stems_dir / "vocals.flac")
            return {"status": "ok", "stems_dir": str(stems_dir)}

    monkeypatch.setattr(stem_search, "PersistentSeparatorClient", FakeClient)
    config = SeparatorBackendConfig(
        mode=MODE_PERSISTENT_SERVER,
        server_url="http://separator-server:8791",
        require_gpu=True,
        device="cuda",
    ).validated()

    result = _run_target_specific_commands(
        candidate={
            "candidate_id": "c001",
            "target_specific_commands": [
                {
                    "role": "vocals",
                    "lane": "vocal_lane",
                    "route_id": "mdx23c_vocals",
                    "command_template": ["fixture-vocal", "{window_mix}", "{output_dir}"],
                }
            ],
        },
        window={
            "window_id": "w000",
            "mix_path": str(tmp_path / "mix.flac"),
            "sample_rate": SAMPLE_RATE,
            "sample_count": 64,
            "channel_count": 1,
        },
        candidate_dir=tmp_path / "c001",
        stems_dir=tmp_path / "c001" / "stems",
        command_runner=lambda _command, _timeout: None,
        timeout_sec=42,
        route_cache={},
        separator_backend_config=config,
    )

    assert result["blocker_reasons"] == []
    assert captured["server_url_set"] == 1.0
    assert captured["request_timeout_sec"] == 72.0


def test_persistent_backend_requests_mapped_source_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, _server_url: str, *, request_timeout_sec: float) -> None:
            captured["request_timeout_sec"] = request_timeout_sec

        def separate(self, **kwargs: object) -> dict[str, str]:
            captured["stems"] = list(kwargs["stems"])  # type: ignore[arg-type]
            stems_dir = Path(str(kwargs["output_dir"])) / "stems"
            _write_stem(stems_dir / "instrumental.flac")
            return {"status": "ok", "stems_dir": str(stems_dir)}

    monkeypatch.setattr(stem_search, "PersistentSeparatorClient", FakeClient)
    config = SeparatorBackendConfig(
        mode=MODE_PERSISTENT_SERVER,
        server_url="http://separator-server:8791",
        require_gpu=True,
        device="cuda",
    ).validated()

    result = _run_target_specific_commands(
        candidate={
            "candidate_id": "c001",
            "target_specific_commands": [
                {
                    "role": "minus",
                    "lane": "minus_lane",
                    "route_id": "becruily_instrumental_vocals",
                    "output_map": {"minus": "instrumental"},
                    "command_template": ["fixture-minus", "{window_mix}", "{output_dir}"],
                }
            ],
        },
        window={
            "window_id": "w000",
            "mix_path": str(tmp_path / "mix.flac"),
            "sample_rate": SAMPLE_RATE,
            "sample_count": 64,
            "channel_count": 1,
        },
        candidate_dir=tmp_path / "c001",
        stems_dir=tmp_path / "c001" / "stems",
        command_runner=lambda _command, _timeout: None,
        timeout_sec=42,
        route_cache={},
        separator_backend_config=config,
    )

    assert result["blocker_reasons"] == []
    assert captured["stems"] == ["instrumental"]
    assert (tmp_path / "c001" / "stems" / "minus.flac").exists()


def test_persistent_backend_passes_prefer_resident_and_caches_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, _server_url: str, *, request_timeout_sec: float) -> None:
            captured["request_timeout_sec"] = request_timeout_sec

        def separate(self, **kwargs: object) -> dict[str, object]:
            captured["prefer_resident"] = kwargs["prefer_resident"]
            stems_dir = Path(str(kwargs["output_dir"])) / "stems"
            _write_stem(stems_dir / "vocals.flac")
            return {
                "status": "ok",
                "stems_dir": str(stems_dir),
                "resident_backend_used": True,
                "resident_executor": {"executor_id": "audio_separator_bs_roformer_vocals"},
                "resident_benchmark": {"model_load_count": 1, "worker_request_count": 1},
            }

    monkeypatch.setattr(stem_search, "PersistentSeparatorClient", FakeClient)
    config = SeparatorBackendConfig(
        mode=MODE_PERSISTENT_SERVER,
        server_url="http://separator-server:8791",
        require_gpu=True,
        device="cuda",
    ).validated()
    route_cache: dict[str, dict[str, object]] = {}
    window = {
        "window_id": "w000",
        "mix_path": str(tmp_path / "mix.flac"),
        "sample_rate": SAMPLE_RATE,
        "sample_count": 64,
        "channel_count": 1,
    }
    candidate = {
        "candidate_id": "c001",
        "target_specific_commands": [
            {
                "role": "vocals",
                "lane": "vocal_lane",
                "route_id": "audio_separator_bs_roformer_vocals",
                "prefer_resident": True,
                "command_template": [
                    "python",
                    "/app/tools/export_audio_separator_vocal_full.py",
                    "{window_mix}",
                    "{output_dir}",
                ],
            }
        ],
    }

    first = _run_target_specific_commands(
        candidate=candidate,
        window=window,
        candidate_dir=tmp_path / "c001",
        stems_dir=tmp_path / "c001" / "stems",
        command_runner=lambda _command, _timeout: None,
        timeout_sec=42,
        route_cache=route_cache,
        separator_backend_config=config,
    )
    second = _run_target_specific_commands(
        candidate={**candidate, "candidate_id": "c002"},
        window=window,
        candidate_dir=tmp_path / "c002",
        stems_dir=tmp_path / "c002" / "stems",
        command_runner=lambda _command, _timeout: None,
        timeout_sec=42,
        route_cache=route_cache,
        separator_backend_config=config,
    )

    assert captured["prefer_resident"] is True
    assert first["routes"][0]["separator_backend"]["prefer_resident"] is True
    assert first["routes"][0]["separator_backend"]["resident_backend_used"] is True
    assert first["routes"][0]["separator_backend"]["resident_executor"]["executor_id"] == "audio_separator_bs_roformer_vocals"
    assert second["routes"][0]["reused_route"] is True
    assert second["routes"][0]["separator_backend"]["cache_hit"] is True
    assert second["routes"][0]["separator_backend"]["resident_backend_used"] is True
    assert second["routes"][0]["separator_backend"]["resident_benchmark"]["model_load_count"] == 1


def test_persistent_server_route_reuse_uses_cache_not_second_request(tmp_path: Path) -> None:
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(body.decode("utf-8"))
            requests.append(payload)
            stems_dir = Path(payload["output_dir"]) / "stems"
            for stem in payload["stems"]:
                _write_stem(stems_dir / f"{stem}.flac")
            manifest_path = Path(payload["output_dir"]) / "manifest.json"
            manifest_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            response = json.dumps({"status": "ok", "stems_dir": str(stems_dir), "manifest_path": str(manifest_path)}).encode(
                "utf-8"
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = SeparatorBackendConfig(
            mode=MODE_PERSISTENT_SERVER,
            server_url=f"http://127.0.0.1:{server.server_port}",
            require_gpu=True,
            device="cuda",
        ).validated()
        route_cache: dict[str, dict[str, str]] = {}
        window = {
            "window_id": "w000",
            "mix_path": str(tmp_path / "mix.flac"),
            "sample_rate": SAMPLE_RATE,
            "sample_count": 64,
            "channel_count": 1,
        }
        candidate = {
            "candidate_id": "c001",
            "target_specific_commands": [
                {
                    "role": "six_stems",
                    "lane": "stem_lane",
                    "route_id": "htdemucs_6s",
                    "command_template": ["fixture-route", "{window_mix}", "{output_dir}"],
                }
            ],
        }

        first = _run_target_specific_commands(
            candidate=candidate,
            window=window,
            candidate_dir=tmp_path / "c001",
            stems_dir=tmp_path / "c001" / "stems",
            command_runner=lambda _command, _timeout: None,
            timeout_sec=30,
            route_cache=route_cache,
            separator_backend_config=config,
        )
        second = _run_target_specific_commands(
            candidate={**candidate, "candidate_id": "c002"},
            window=window,
            candidate_dir=tmp_path / "c002",
            stems_dir=tmp_path / "c002" / "stems",
            command_runner=lambda _command, _timeout: None,
            timeout_sec=30,
            route_cache=route_cache,
            separator_backend_config=config,
        )
    finally:
        server.shutdown()
        thread.join(timeout=2.0)

    assert len(requests) == 1
    assert first["routes"][0]["separator_backend"]["mode"] == MODE_PERSISTENT_SERVER
    assert first["routes"][0]["separator_backend"]["cache_hit"] is False
    assert second["routes"][0]["reused_route"] is True
    assert second["routes"][0]["separator_backend"]["cache_hit"] is True
    assert all((tmp_path / "c002" / "stems" / f"{stem}.flac").exists() for stem in STEMS)


def test_persistent_server_cached_failure_prevents_duplicate_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    separator_server._ROUTE_CACHE.clear()
    monkeypatch.setattr(separator_server, "GLOBAL_PROGRESS_PATH", tmp_path / "global_progress.jsonl")
    marker = tmp_path / "reran.txt"
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).write_text('ran', encoding='utf-8'); import time; time.sleep(2)",
    ]
    payload = {
        "route_id": "slow_route",
        "batch_id": "w000:c001:slow_route",
        "cache_key": "sep_timeout_fixture",
        "output_dir": str(tmp_path / "first"),
        "stems": ["vocals"],
        "command": command,
        "timeout_sec": 1,
        "require_gpu": False,
        "device": "cpu",
    }

    status, response = separator_server._handle_separate(payload)
    assert status == 500
    assert response["reason"] == "separator_command_timeout"
    assert marker.exists()

    marker.unlink()
    second_payload = {**payload, "batch_id": "w000:c002:slow_route", "output_dir": str(tmp_path / "second")}
    started = time.monotonic()
    second_status, second_response = separator_server._handle_separate(second_payload)

    assert time.monotonic() - started < 0.5
    assert second_status == 500
    assert second_response["cache_hit"] is True
    assert second_response["reason"] == "cached_separator_command_timeout"
    assert not marker.exists()


def test_resident_executor_supported_route_runs_without_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    separator_server._ROUTE_CACHE.clear()
    reset_resident_separator_executors_for_tests()
    monkeypatch.setattr(separator_server, "GLOBAL_PROGRESS_PATH", tmp_path / "global_progress.jsonl")

    def fail_popen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("resident route must not spawn subprocess")

    monkeypatch.setattr(separator_server.subprocess, "Popen", fail_popen)
    mix_path = tmp_path / "mix.flac"
    _write_stem(mix_path)
    payload = {
        "route_id": RESIDENT_FIXTURE_ROUTE_ID,
        "batch_id": "w000:c001:resident",
        "cache_key": "sep_resident_fixture",
        "source_audio_path": str(mix_path),
        "output_dir": str(tmp_path / "resident"),
        "stems": ["vocals", "drums"],
        "command": ["resident-fixture"],
        "timeout_sec": 30,
        "require_gpu": False,
        "device": "cpu",
    }

    status, response = separator_server._handle_separate(payload)

    assert status == 200
    assert response["status"] == "ok"
    assert response["resident_backend_used"] is True
    assert response["resident_executor"]["executor_id"] == RESIDENT_FIXTURE_ROUTE_ID
    assert response["resident_executor"]["model_load_count"] == 1
    assert response["delivery_allowed"] is False
    assert (tmp_path / "resident" / "stems" / "vocals.flac").is_file()
    assert (tmp_path / "resident" / "stems" / "drums.flac").is_file()
    progress = (tmp_path / "resident" / "server_progress.jsonl").read_text(encoding="utf-8")
    assert "resident_executor_start" in progress
    assert "resident_executor_end" in progress

    health = resident_executor_health()
    assert health["loaded_routes"][RESIDENT_FIXTURE_ROUTE_ID]["model_load_count"] == 1


def test_resident_fixture_blocks_when_gpu_inference_is_required(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    separator_server._ROUTE_CACHE.clear()
    reset_resident_separator_executors_for_tests()
    monkeypatch.setattr(separator_server, "GLOBAL_PROGRESS_PATH", tmp_path / "global_progress.jsonl")
    mix_path = tmp_path / "mix.flac"
    _write_stem(mix_path)
    payload = {
        "route_id": RESIDENT_FIXTURE_ROUTE_ID,
        "batch_id": "w000:c001:resident",
        "cache_key": "sep_resident_gpu_fixture",
        "source_audio_path": str(mix_path),
        "output_dir": str(tmp_path / "resident_gpu"),
        "stems": ["vocals"],
        "command": ["resident-fixture"],
        "timeout_sec": 30,
        "require_gpu": True,
        "device": "cuda",
    }

    status, response = separator_server._handle_separate(payload)

    assert status == 500
    assert response["status"] == "blocked"
    assert response["reason"] == "resident_fixture_gpu_inference_unavailable"
    assert response["resident_backend_used"] is True
    assert response["gpu_inference_skipped"] is True
    assert response["delivery_allowed"] is False


def test_unsupported_route_uses_subprocess_and_marks_resident_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    separator_server._ROUTE_CACHE.clear()
    reset_resident_separator_executors_for_tests()
    monkeypatch.setattr(separator_server, "GLOBAL_PROGRESS_PATH", tmp_path / "global_progress.jsonl")
    stems_dir = tmp_path / "subprocess" / "stems"
    command = [
        sys.executable,
        "-c",
        (
            "import sys; from pathlib import Path; "
            "d=Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True); "
            "(d/'vocals.flac').write_bytes(b'fixture')"
        ),
        str(stems_dir),
    ]
    payload = {
        "route_id": "subprocess_only_fixture",
        "batch_id": "w000:c001:subprocess",
        "cache_key": "sep_subprocess_fixture",
        "source_audio_path": str(tmp_path / "mix.flac"),
        "output_dir": str(tmp_path / "subprocess"),
        "stems": ["vocals"],
        "command": command,
        "timeout_sec": 5,
        "require_gpu": False,
        "device": "cpu",
    }

    status, response = separator_server._handle_separate(payload)

    assert status == 200
    assert response["status"] == "ok"
    assert response["resident_backend_used"] is False
    assert response["resident_capable"] is False
    assert response["delivery_allowed"] is False


def test_audio_separator_resident_route_is_opt_in_and_uses_worker_not_route_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    separator_server._ROUTE_CACHE.clear()
    reset_resident_separator_executors_for_tests()
    monkeypatch.setattr(separator_server, "GLOBAL_PROGRESS_PATH", tmp_path / "global_progress.jsonl")

    def fail_popen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("resident audio separator route must not spawn the route subprocess")

    def fake_worker_request(self: AudioSeparatorResidentExecutor, request: dict[str, object]) -> dict[str, object]:
        stems_dir = Path(str(request["output_dir"])) / "stems"
        _write_stem(stems_dir / "vocals.flac")
        _write_stem(stems_dir / "instrumental.flac")
        return {
            "status": "ok",
            "stems_dir": str(stems_dir),
            "output_files": [str(stems_dir / "vocals.flac"), str(stems_dir / "instrumental.flac")],
            "runtime_contract": {"status": "ready", "cuda_provider_available": True, "blockers": []},
        }

    monkeypatch.setattr(separator_server.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(AudioSeparatorResidentExecutor, "_run_worker_request", fake_worker_request)
    mix_path = tmp_path / "mix.flac"
    _write_stem(mix_path)
    payload = {
        "route_id": "audio_separator_bs_roformer_vocals",
        "batch_id": "w000:c001:audio-separator",
        "cache_key": "sep_audio_separator_resident",
        "source_audio_path": str(mix_path),
        "output_dir": str(tmp_path / "audio_separator"),
        "stems": ["vocals", "instrumental"],
        "command": [
            "python",
            "/app/tools/export_audio_separator_vocal_full.py",
            str(mix_path),
            str(tmp_path / "audio_separator"),
            "--model-filename",
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "--model-file-dir",
            "/app/models/audio_separator",
            "--audio-separator-python",
            "/opt/audio-separator-venv/bin/python",
            "--require-cuda-provider",
        ],
        "timeout_sec": 30,
        "require_gpu": True,
        "device": "cuda",
        "prefer_resident": True,
    }

    status, response = separator_server._handle_separate(payload)

    assert status == 200
    assert response["status"] == "ok"
    assert response["resident_backend_used"] is True
    assert response["resident_executor"]["executor_id"] == "audio_separator_bs_roformer_vocals"
    assert response["resident_executor"]["adapter_kind"] == "audio_separator_worker"
    assert response["resident_executor"]["model_load_count"] == 1
    assert response["gpu_inference_skipped"] is False
    assert response["delivery_allowed"] is False
    assert (tmp_path / "audio_separator" / "stems" / "vocals.flac").is_file()
    assert (tmp_path / "audio_separator" / "stems" / "instrumental.flac").is_file()


def test_melband_resident_route_is_opt_in_and_uses_worker_not_route_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    separator_server._ROUTE_CACHE.clear()
    reset_resident_separator_executors_for_tests()
    monkeypatch.setattr(separator_server, "GLOBAL_PROGRESS_PATH", tmp_path / "global_progress.jsonl")

    def fail_popen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("resident MelBand route must not spawn the route subprocess")

    def fake_worker_request(self: MelBandResidentExecutor, request: dict[str, object]) -> dict[str, object]:
        stems_dir = Path(str(request["output_dir"])) / "stems"
        _write_stem(stems_dir / "vocals.flac")
        _write_stem(stems_dir / "instrumental.flac")
        _write_stem(Path(str(request["output_dir"])) / "raw_ref.flac")
        return {
            "status": "ok",
            "stems_dir": str(stems_dir),
            "output_files": [str(stems_dir / "vocals.flac"), str(stems_dir / "instrumental.flac")],
            "runtime_contract": {"status": "ready", "cuda_available": True, "blockers": []},
            "model_load_count": 1,
            "worker_request_count": 1,
            "persistent_worker_process_used": True,
            "per_request_external_runtime_subprocess_used": False,
        }

    monkeypatch.setattr(separator_server.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(MelBandResidentExecutor, "_run_worker_request", fake_worker_request)
    mix_path = tmp_path / "mix.flac"
    _write_stem(mix_path)
    payload = {
        "route_id": "becruily_instrumental_vocals",
        "batch_id": "w000:c001:becruily",
        "cache_key": "sep_becruily_resident",
        "source_audio_path": str(mix_path),
        "output_dir": str(tmp_path / "becruily"),
        "stems": ["vocals", "instrumental"],
        "command": [
            "python",
            "tools/export_melband_vocal_full.py",
            str(mix_path),
            str(tmp_path / "becruily" / "stems"),
            "--model-path",
            "/app/vendor/becruily_instrumental/mel_band_roformer_instrumental_becruily.ckpt",
            "--config-path",
            "/app/vendor/becruily_instrumental/config_instrumental_becruily.yaml",
            "--manifest-model-id",
            "becruily_instrumental_vocals",
        ],
        "timeout_sec": 30,
        "require_gpu": True,
        "device": "cuda",
        "prefer_resident": True,
    }

    status, response = separator_server._handle_separate(payload)

    assert status == 200
    assert response["status"] == "ok"
    assert response["resident_backend_used"] is True
    assert response["resident_executor"]["executor_id"] == "becruily_instrumental_vocals"
    assert response["resident_executor"]["adapter_kind"] == "melband_roformer_worker"
    assert response["resident_executor"]["model_load_count"] == 1
    assert response["gpu_inference_skipped"] is False
    assert response["delivery_allowed"] is False
    assert (tmp_path / "becruily" / "stems" / "vocals.flac").is_file()
    assert (tmp_path / "becruily" / "stems" / "instrumental.flac").is_file()


def test_audio_separator_resident_worker_reuses_process_and_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reset_resident_separator_executors_for_tests()
    launches: list[list[str]] = []
    requests: list[dict[str, object]] = []

    class FakeStdout:
        def __init__(self) -> None:
            self.lines: queue.Queue[str] = queue.Queue()

        def readline(self) -> str:
            try:
                return self.lines.get(timeout=2.0)
            except queue.Empty:
                return ""

    class FakeStdin:
        def __init__(self, stdout: FakeStdout) -> None:
            self.stdout = stdout
            self.model_loaded = False
            self.request_count = 0

        def write(self, line: str) -> int:
            payload = json.loads(line)
            requests.append(payload)
            if payload.get("action") == "shutdown":
                self.stdout.lines.put("")
                return len(line)
            self.request_count += 1
            self.model_loaded = True
            model_output_dir = Path(payload["model_output_dir"])
            _write_stem(model_output_dir / "vocals.flac")
            _write_stem(model_output_dir / "instrumental.flac")
            response = {
                "request_id": payload["request_id"],
                "status": "ok",
                "output_files": ["vocals.flac", "instrumental.flac"],
                "model_load_count": 1,
                "worker_request_count": self.request_count,
                "available_providers": ["CUDAExecutionProvider"],
                "persistent_worker_process_used": True,
            }
            self.stdout.lines.put(resident_executor.AUDIO_SEPARATOR_WORKER_RESPONSE_PREFIX + json.dumps(response) + "\n")
            return len(line)

        def flush(self) -> None:
            return

    class FakeProcess:
        def __init__(self, command: list[str], **_kwargs: object) -> None:
            launches.append(command)
            self.stdout = FakeStdout()
            self.stdin = FakeStdin(self.stdout)
            self.pid = 12345
            self.returncode = None

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(resident_executor.subprocess, "Popen", FakeProcess)
    source = tmp_path / "mix.flac"
    _write_stem(source)
    command = [
        "python",
        "/app/tools/export_audio_separator_vocal_full.py",
        str(source),
        str(tmp_path / "out"),
        "--model-filename",
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "--model-file-dir",
        "/app/models/audio_separator",
        "--audio-separator-python",
        "/opt/audio-separator-venv/bin/python",
        "--require-cuda-provider",
    ]
    executor = AudioSeparatorResidentExecutor("audio_separator_bs_roformer_vocals")

    first = executor.separate(
        source_audio_path=source,
        output_dir=tmp_path / "first",
        stems=("vocals", "instrumental"),
        device="cuda",
        require_gpu=True,
        command=command,
    )
    second = executor.separate(
        source_audio_path=source,
        output_dir=tmp_path / "second",
        stems=("vocals", "instrumental"),
        device="cuda",
        require_gpu=True,
        command=command,
    )

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert len(launches) == 1
    assert len(requests) == 2
    assert second["resident_benchmark"]["model_load_count"] == 1
    assert second["resident_benchmark"]["worker_request_count"] == 2
    assert second["worker_response"]["persistent_worker_process_used"] is True
    assert second["worker_response"]["model_load_count"] == 1


def test_audio_separator_server_passes_payload_timeout_to_resident_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    separator_server._ROUTE_CACHE.clear()
    reset_resident_separator_executors_for_tests()
    monkeypatch.setattr(separator_server, "GLOBAL_PROGRESS_PATH", tmp_path / "global_progress.jsonl")
    captured: dict[str, object] = {}

    def fake_worker_request(self: AudioSeparatorResidentExecutor, request: dict[str, object]) -> dict[str, object]:
        captured["timeout_sec"] = request["timeout_sec"]
        stems_dir = Path(str(request["output_dir"])) / "stems"
        _write_stem(stems_dir / "vocals.flac")
        _write_stem(stems_dir / "instrumental.flac")
        return {
            "status": "ok",
            "output_files": [str(stems_dir / "vocals.flac"), str(stems_dir / "instrumental.flac")],
            "model_load_count": 1,
            "worker_request_count": 1,
            "persistent_worker_process_used": True,
            "per_request_external_runtime_subprocess_used": False,
        }

    monkeypatch.setattr(AudioSeparatorResidentExecutor, "_run_worker_request", fake_worker_request)
    mix_path = tmp_path / "mix.flac"
    _write_stem(mix_path)
    payload = {
        "route_id": "audio_separator_bs_roformer_vocals",
        "batch_id": "w000:c001:audio-separator",
        "cache_key": "sep_audio_separator_timeout",
        "source_audio_path": str(mix_path),
        "output_dir": str(tmp_path / "audio_separator"),
        "stems": ["vocals", "instrumental"],
        "command": [
            "python",
            "/app/tools/export_audio_separator_vocal_full.py",
            str(mix_path),
            str(tmp_path / "out"),
            "--model-filename",
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "--model-file-dir",
            "/app/models/audio_separator",
            "--audio-separator-python",
            "/opt/audio-separator-venv/bin/python",
            "--require-cuda-provider",
        ],
        "timeout_sec": 321,
        "require_gpu": True,
        "device": "cuda",
        "prefer_resident": True,
    }

    status, response = separator_server._handle_separate(payload)

    assert status == 200
    assert response["status"] == "ok"
    assert captured["timeout_sec"] == 321


def test_audio_separator_resident_accepts_runner_relative_exporter_path(tmp_path: Path) -> None:
    executor = AudioSeparatorResidentExecutor("audio_separator_bs_roformer_vocals")
    source = tmp_path / "mix.flac"
    _write_stem(source)

    request = executor._build_worker_request(
        command=[
            "python",
            "tools/export_audio_separator_vocal_full.py",
            str(source),
            str(tmp_path / "out"),
            "--model-filename",
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "--model-file-dir",
            "/app/models/audio_separator",
            "--audio-separator-python",
            "/opt/audio-separator-venv/bin/python",
            "--require-cuda-provider",
        ],
        source_audio_path=source,
        output_dir=tmp_path / "out",
        stems=("vocals",),
        timeout_sec=900,
    )

    assert request["model_filename"] == "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    assert request["timeout_sec"] == 900


def test_audio_separator_worker_timeout_stops_process_before_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_resident_separator_executors_for_tests()
    processes: list[object] = []

    class QuietStdout:
        def readline(self) -> str:
            time.sleep(0.05)
            return ""

    class QuietStdin:
        def write(self, _line: str) -> int:
            return 1

        def flush(self) -> None:
            return

    class FakeProcess:
        def __init__(self, command: list[str], **_kwargs: object) -> None:
            self.command = command
            self.stdin = QuietStdin()
            self.stdout = QuietStdout()
            self.returncode = None
            self.kill_called = False
            processes.append(self)

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.kill_called = True
            self.returncode = -9

    monkeypatch.setattr(resident_executor.subprocess, "Popen", FakeProcess)
    worker = resident_executor.AudioSeparatorWorkerProcess(
        audio_separator_python=Path("/opt/audio-separator-venv/bin/python"),
        worker_key="timeout-fixture",
    )
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    payload = {
        "source_audio_path": "/tmp/in.flac",
        "model_output_dir": "/tmp/out",
    }

    first = worker.request(payload, timeout_sec=1)
    second = worker.request(payload, timeout_sec=1)

    assert first["status"] == "blocked"
    assert first["reason"] == "audio_separator_worker_timeout"
    assert second["status"] == "blocked"
    assert len(processes) == 2
    assert processes[0].kill_called is True


def test_audio_separator_route_without_resident_opt_in_keeps_subprocess_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    separator_server._ROUTE_CACHE.clear()
    reset_resident_separator_executors_for_tests()
    monkeypatch.setattr(separator_server, "GLOBAL_PROGRESS_PATH", tmp_path / "global_progress.jsonl")
    stems_dir = tmp_path / "audio_separator_subprocess" / "stems"
    command = [
        sys.executable,
        "-c",
        (
            "import sys; from pathlib import Path; "
            "d=Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True); "
            "(d/'vocals.flac').write_bytes(b'fixture')"
        ),
        str(stems_dir),
    ]
    payload = {
        "route_id": "audio_separator_bs_roformer_vocals",
        "batch_id": "w000:c001:audio-separator",
        "cache_key": "sep_audio_separator_subprocess",
        "source_audio_path": str(tmp_path / "mix.flac"),
        "output_dir": str(tmp_path / "audio_separator_subprocess"),
        "stems": ["vocals"],
        "command": command,
        "timeout_sec": 5,
        "require_gpu": False,
        "device": "cpu",
    }

    status, response = separator_server._handle_separate(payload)

    assert status == 200
    assert response["status"] == "ok"
    assert response["resident_backend_used"] is False
    assert response["resident_capable"] is False


def test_persistent_server_unavailable_blocks_as_infra_gap(tmp_path: Path) -> None:
    config = SeparatorBackendConfig(
        mode=MODE_PERSISTENT_SERVER,
        server_url="http://127.0.0.1:9",
        require_gpu=True,
        device="cuda",
        request_timeout_sec=0.1,
    ).validated()

    manifest = execute_recipe_candidate(
        candidate={
            "candidate_id": "c001",
            "target_specific_commands": [
                {
                    "role": "vocals",
                    "lane": "vocal_lane",
                    "route_id": "mdx23c_vocals",
                    "command_template": ["fixture-vocal", "{window_mix}", "{output_dir}"],
                }
            ],
        },
        window={
            "window_id": "w000",
            "mix_path": str(tmp_path / "mix.flac"),
            "sample_rate": SAMPLE_RATE,
            "sample_count": 64,
            "channel_count": 1,
        },
        window_dir=tmp_path / "w000",
        command_runner=lambda _command, _timeout: None,
        timeout_sec=1,
        route_cache={},
        separator_backend_config=config,
    )

    assert manifest["verdict"] == "blocked_unknown"
    assert manifest["execution_status"] == "failed_separator_backend_unavailable"
    assert manifest["root_cause"] == "infra_gap"
    assert manifest["delivery_allowed"] is False
