from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI
from scipy.signal import butter, sosfiltfilt

from app.banquet_runtime import (
    banquet_ready_state,
    cache_snapshot as banquet_cache_snapshot,
    maybe_unload_idle_runtime as maybe_unload_idle_banquet_runtime,
    prefetch_runtime as prefetch_banquet_runtime,
    run_banquet_query,
)
from app.runtime import (
    cache_snapshot as clap_cache_snapshot,
    encode_audio,
    encode_text,
    maybe_unload_idle_runtime as maybe_unload_idle_clap_runtime,
    prefetch_runtime as prefetch_clap_runtime,
)
from app.schemas import RunRequest, RunResult
from app.utils import (
    DRY_RUN,
    QUERYSEP_BACKEND_DEFAULT,
    QUERYSEP_BANQUET_ENABLED,
    QUERYSEP_BANQUET_QUERY_LENGTH_SECONDS,
    QUERYSEP_BANQUET_QUERY_TOP_K,
    QUERYSEP_PREFETCH_ON_START,
)

app = FastAPI(title="querysep-worker", version="0.5.0")

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}
TARGET_SR = 48000
BANQUET_SR = 44100
EMBED_WINDOW_SEC = 8.0
EMBED_TOP_WINDOWS = 3

PROMPT_TEXTS = {
    "strings_orchestra": "orchestral strings ensemble with violins, violas, cellos and sustained string section",
    "synth_lead": "prominent synthesizer lead melody, bright electronic lead instrument",
    "synth_pad": "wide atmospheric synth pad, sustained electronic texture and warm pad",
    "arps_plucks": "fast arpeggio plucks, plucked synth notes and rhythmic melodic plucks",
    "brass_woodwinds": "brass and woodwind section, horns, trumpets, saxophone, clarinet and flute",
    "organ_reeds": "organ and reed keyboard, harmonium, accordion or reed organ texture",
    "fx_risers_impacts": "sound effects, risers, impacts, transitions and cinematic effects",
    "ambience_residual": "background ambience, room tail, diffuse residual texture and remaining atmosphere",
}
BANQUET_SUPPORTED_PROMPTS = set(PROMPT_TEXTS) - {"ambience_residual"}
HARMONIC_RESIDUAL_QUERY_CANDIDATE_TYPE = "banquet_byoq_harmonic_residual_query"
BANQUET_QUERY_MODE_DEFAULT = "default"
BANQUET_QUERY_MODE_HARMONIC_RESIDUAL_SINGLE = "harmonic_residual_single"
BANQUET_QUERY_MODE_POLICIES = {
    BANQUET_QUERY_MODE_HARMONIC_RESIDUAL_SINGLE: {
        "allowed_components": ("bus_harmonic_residual",),
        "top_k": 1,
    },
}
SUPPORTED_BANQUET_QUERY_MODES = {
    BANQUET_QUERY_MODE_DEFAULT,
    *BANQUET_QUERY_MODE_POLICIES.keys(),
}

# CLAP still matters here, but now as query-exemplar selection for the real Banquet decoder.
BANQUET_QUERY_BIASES = {
    "strings_orchestra": {
        "bus_harmonic_residual": 0.18,
        "bus_other": 0.16,
        "bus_piano": 0.1,
        "input_harmonic": 0.06,
        "bus_guitar": 0.02,
    },
    "synth_lead": {
        "bus_other": 0.16,
        "bus_piano": 0.09,
        "input_high_band": 0.08,
        "input_mid_band": 0.05,
    },
    "synth_pad": {
        "bus_other": 0.17,
        "bus_piano": 0.08,
        "input_harmonic": 0.06,
        "input_high_band": 0.05,
    },
    "arps_plucks": {
        "bus_guitar": 0.16,
        "bus_other": 0.13,
        "bus_piano": 0.1,
        "input_high_band": 0.08,
    },
    "brass_woodwinds": {
        "bus_other": 0.16,
        "bus_piano": 0.09,
        "input_mid_band": 0.08,
        "input_harmonic": 0.05,
    },
    "organ_reeds": {
        "bus_piano": 0.14,
        "bus_other": 0.12,
        "input_mid_band": 0.08,
        "input_harmonic": 0.05,
    },
    "fx_risers_impacts": {
        "bus_other": 0.18,
        "input_high_band": 0.12,
        "input_mid_band": 0.05,
    },
}


@app.on_event("startup")
def warm_querysep_runtimes() -> None:
    app.state.startup_prefetch = {}
    app.state.startup_prefetch_errors = {}
    if not QUERYSEP_PREFETCH_ON_START:
        return

    try:
        app.state.startup_prefetch["clap_router"] = prefetch_clap_runtime()
    except Exception as exc:  # noqa: BLE001
        app.state.startup_prefetch_errors["clap_router"] = str(exc)

    if QUERYSEP_BANQUET_ENABLED:
        try:
            app.state.startup_prefetch["banquet_byoq"] = prefetch_banquet_runtime()
        except Exception as exc:  # noqa: BLE001
            app.state.startup_prefetch_errors["banquet_byoq"] = str(exc)


def output_root_for(request: RunRequest) -> Path:
    target = request.target_stem or request.prompt or request.task
    root = Path(request.output_root) / request.phase_id / f"{request.step_index:02d}_{target}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def output_path_for(request: RunRequest, output_root: Path) -> Path:
    target = request.target_stem or request.prompt or request.task
    return output_root / f"{target}.wav"


def resolve_prompt(request: RunRequest) -> str:
    prompt = request.prompt or request.target_stem or "ambience_residual"
    return prompt if prompt in PROMPT_TEXTS else "ambience_residual"


def resolve_backend(request: RunRequest, prompt: str) -> str:
    raw_backend = str((request.raw_step or {}).get("backend") or QUERYSEP_BACKEND_DEFAULT).strip().lower()
    if request.task == "residual_export" or prompt == "ambience_residual":
        return "clap_router"
    if raw_backend in {"banquet", "banquet_byoq"}:
        return "banquet_byoq"
    if raw_backend in {"clap", "clap_router"}:
        return "clap_router"
    if raw_backend == "auto" and QUERYSEP_BANQUET_ENABLED and prompt in BANQUET_SUPPORTED_PROMPTS:
        ready = banquet_ready_state()
        if ready["checkpoint_exists"]:
            return "banquet_byoq"
    return "clap_router"


def load_audio(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32, copy=False)


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        return (audio / peak).astype(np.float32, copy=False)
    return audio.astype(np.float32, copy=False)


def rms_level(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def band_filter(audio: np.ndarray, sr: int, low_hz: float | None, high_hz: float | None) -> np.ndarray:
    nyquist = sr * 0.5
    if low_hz is None:
        low_hz = 20.0
    if high_hz is None:
        high_hz = nyquist - 10.0
    low_hz = max(10.0, min(low_hz, nyquist - 20.0))
    high_hz = max(low_hz + 20.0, min(high_hz, nyquist - 10.0))
    sos = butter(4, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32, copy=False)


def derive_input_components(audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    harmonic, percussive = librosa.effects.hpss(audio)
    return {
        "input_full": audio,
        "input_harmonic": harmonic.astype(np.float32, copy=False),
        "input_percussive": percussive.astype(np.float32, copy=False),
        "input_low_band": band_filter(audio, sr, 120.0, 900.0),
        "input_mid_band": band_filter(audio, sr, 900.0, 4200.0),
        "input_high_band": band_filter(audio, sr, 4200.0, 14000.0),
    }


def build_component_bank(request: RunRequest, prompt: str) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    base_audio = load_audio(Path(request.input_path))
    components = derive_input_components(base_audio, TARGET_SR)
    metadata: dict[str, dict[str, object]] = {
        name: {
            "origin": "derived_from_input",
            "source_path": request.input_path,
            "rms": round(rms_level(audio), 6),
            "duration_sec": round(len(audio) / TARGET_SR, 3),
        }
        for name, audio in components.items()
    }

    for key, path_str in sorted(request.build_from_paths.items()):
        path = Path(path_str)
        if not path.exists() or not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        try:
            audio = load_audio(path)
        except Exception:
            continue
        if audio.size == 0 or rms_level(audio) < 1e-5:
            continue
        component_name = f"bus_{key}"
        components[component_name] = audio
        metadata[component_name] = {
            "origin": "build_from_bus",
            "source_bus": key,
            "source_path": str(path),
            "rms": round(rms_level(audio), 6),
            "duration_sec": round(len(audio) / TARGET_SR, 3),
        }

    if prompt == "ambience_residual":
        components["input_air_band"] = band_filter(base_audio, TARGET_SR, 6000.0, 17000.0)
        metadata["input_air_band"] = {
            "origin": "derived_from_input",
            "source_path": request.input_path,
            "rms": round(rms_level(components["input_air_band"]), 6),
            "duration_sec": round(len(base_audio) / TARGET_SR, 3),
        }

    return components, metadata


def component_windows(
    audio: np.ndarray,
    sr: int,
    window_sec: float = EMBED_WINDOW_SEC,
    top_k: int = EMBED_TOP_WINDOWS,
) -> list[dict[str, object]]:
    window_samples = max(1, int(window_sec * sr))
    if audio.size <= window_samples:
        return [
            {
                "audio": audio,
                "start_sample": 0,
                "end_sample": int(audio.size),
                "start_sec": 0.0,
                "end_sec": round(float(audio.size / sr), 3),
                "rms": round(rms_level(audio), 6),
            }
        ]

    hop = max(1, window_samples // 2)
    candidates: list[tuple[float, int]] = []
    for start in range(0, max(1, audio.size - window_samples + 1), hop):
        segment = audio[start:start + window_samples]
        candidates.append((rms_level(segment), start))

    candidates.sort(reverse=True)
    selected: list[dict[str, object]] = []
    used_starts: set[int] = set()
    for energy, start in candidates:
        if start in used_starts:
            continue
        segment = audio[start:start + window_samples]
        selected.append(
            {
                "audio": segment,
                "start_sample": start,
                "end_sample": start + len(segment),
                "start_sec": round(float(start / sr), 3),
                "end_sec": round(float((start + len(segment)) / sr), 3),
                "rms": round(float(energy), 6),
            }
        )
        used_starts.add(start)
        if len(selected) >= top_k:
            break

    return selected


def score_components(
    prompt: str,
    components: dict[str, np.ndarray],
) -> tuple[
    dict[str, float],
    dict[str, dict[str, object]],
    dict[str, list[dict[str, object]]],
    dict[str, object],
]:
    text_vector, runtime_meta = encode_text(PROMPT_TEXTS[prompt])
    scores: dict[str, float] = {}
    best_windows: dict[str, dict[str, object]] = {}
    ranked_windows: dict[str, list[dict[str, object]]] = {}

    for name, audio in components.items():
        scored_windows: list[dict[str, object]] = []
        for window in component_windows(audio, TARGET_SR):
            window_audio = window["audio"]
            if rms_level(window_audio) < 1e-5:
                continue
            audio_vector, _ = encode_audio(window_audio, TARGET_SR, runtime_meta=runtime_meta)
            similarity = float(np.dot(text_vector, audio_vector))
            scored_windows.append(
                {
                    **window,
                    "similarity": round(similarity, 6),
                }
            )

        if not scored_windows:
            scores[name] = -1.0
            continue

        scored_windows.sort(key=lambda item: float(item["similarity"]), reverse=True)
        ranked_windows[name] = scored_windows
        best_windows[name] = scored_windows[0]
        scores[name] = float(scored_windows[0]["similarity"])
    return scores, best_windows, ranked_windows, runtime_meta


def softmax(values: np.ndarray, temperature: float = 6.0) -> np.ndarray:
    if values.size == 0:
        return values
    shifted = values - np.max(values)
    exp_values = np.exp(shifted * temperature)
    denom = float(np.sum(exp_values))
    if denom <= 0:
        return np.full_like(values, 1.0 / max(1, len(values)))
    return exp_values / denom


def mix_top_components(components: dict[str, np.ndarray], scores: dict[str, float]) -> tuple[np.ndarray, list[dict[str, object]]]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected = ranked[: min(3, len(ranked))]
    if not selected:
        return np.zeros(1, dtype=np.float32), []

    weight_values = softmax(np.asarray([score for _, score in selected], dtype=np.float32))
    target_length = max(len(components[name]) for name, _ in selected)
    output = np.zeros(target_length, dtype=np.float32)
    summary: list[dict[str, object]] = []

    for (name, score), weight in zip(selected, weight_values, strict=True):
        audio = components[name]
        if len(audio) < target_length:
            padded = np.zeros(target_length, dtype=np.float32)
            padded[: len(audio)] = audio
            audio = padded
        output += audio * float(weight)
        summary.append(
            {
                "component": name,
                "similarity": round(float(score), 6),
                "weight": round(float(weight), 6),
            }
        )

    return peak_normalize(output), summary


def _generic_query_bias(component_name: str) -> float:
    if component_name == "input_percussive":
        return -0.45
    if component_name == "input_full":
        return -0.22
    if component_name == "bus_harmonic_residual":
        return 0.1
    if component_name.startswith("bus_"):
        return 0.05
    if component_name in {"input_harmonic", "input_mid_band", "input_high_band"}:
        return 0.02
    return 0.0


def resolve_banquet_query_mode(request: RunRequest, prompt: str) -> str:
    raw_mode = request.raw_step.get("query_mode")
    if raw_mode:
        query_mode = str(raw_mode)
        if query_mode not in SUPPORTED_BANQUET_QUERY_MODES:
            raise ValueError(f"Unsupported Banquet query_mode: {query_mode}")
        return query_mode

    if request.raw_step.get("candidate_type") == HARMONIC_RESIDUAL_QUERY_CANDIDATE_TYPE:
        return BANQUET_QUERY_MODE_HARMONIC_RESIDUAL_SINGLE

    return BANQUET_QUERY_MODE_DEFAULT


def _banquet_query_mode_policy(query_mode: str) -> dict[str, object]:
    if query_mode == BANQUET_QUERY_MODE_DEFAULT:
        return {}
    if query_mode not in BANQUET_QUERY_MODE_POLICIES:
        raise ValueError(f"Unsupported Banquet query_mode: {query_mode}")
    return BANQUET_QUERY_MODE_POLICIES[query_mode]


def _component_allowed_for_query_mode(component_name: str, query_mode_policy: dict[str, object]) -> bool:
    allowed_components = query_mode_policy.get("allowed_components")
    if allowed_components is None:
        return True
    return component_name in set(str(component) for component in allowed_components)


def _top_k_for_query_mode(query_mode_policy: dict[str, object]) -> int:
    raw_top_k = query_mode_policy.get("top_k", QUERYSEP_BANQUET_QUERY_TOP_K)
    return max(1, int(raw_top_k))


def select_banquet_query_candidates(
    prompt: str,
    scores: dict[str, float],
    best_windows: dict[str, dict[str, object]],
    ranked_windows: dict[str, list[dict[str, object]]],
    *,
    query_mode: str = BANQUET_QUERY_MODE_DEFAULT,
) -> dict[str, object]:
    ranked: list[dict[str, object]] = []
    blocked: list[dict[str, object]] = []
    prompt_biases = BANQUET_QUERY_BIASES.get(prompt, {})
    query_mode_policy = _banquet_query_mode_policy(query_mode)
    for component_name, base_score in scores.items():
        component_windows_ranked = ranked_windows.get(component_name)
        if component_name not in best_windows or not component_windows_ranked:
            continue
        if not _component_allowed_for_query_mode(component_name, query_mode_policy):
            blocked.append(
                {
                    "component": component_name,
                    "reason": "component_not_allowed_for_query_mode",
                    "query_mode": query_mode,
                    "base_similarity": round(float(base_score), 6),
                }
            )
            continue
        for window_index, window in enumerate(component_windows_ranked):
            similarity = float(window["similarity"])
            adjusted = (
                similarity
                + _generic_query_bias(component_name)
                + prompt_biases.get(component_name, 0.0)
                - (0.02 * window_index)
            )
            ranked.append(
                {
                    "component": component_name,
                    "base_similarity": round(float(base_score), 6),
                    "window_similarity": round(similarity, 6),
                    "adjusted_similarity": round(adjusted, 6),
                    "window_index": window_index,
                    "window": window,
                }
            )
    ranked.sort(key=lambda item: item["adjusted_similarity"], reverse=True)
    if not ranked:
        raise RuntimeError(
            f"No usable Banquet query candidate was found for query_mode={query_mode} "
            "from the available buses/components."
        )
    ensemble: list[dict[str, object]] = []
    used_components: set[str] = set()
    top_k = _top_k_for_query_mode(query_mode_policy)
    for item in ranked:
        component_name = str(item["component"])
        if component_name in used_components:
            continue
        ensemble.append(dict(item))
        used_components.add(component_name)
        if len(ensemble) >= top_k:
            break
    if len(ensemble) < top_k:
        for item in ranked:
            if len(ensemble) >= top_k:
                break
            if item in ensemble:
                continue
            ensemble.append(dict(item))

    weight_values = softmax(
        np.asarray([float(item["adjusted_similarity"]) for item in ensemble], dtype=np.float32)
    )
    for item, weight in zip(ensemble, weight_values, strict=True):
        item["weight"] = round(float(weight), 6)

    return {
        "best": ranked[0],
        "ranked": ranked,
        "ensemble": ensemble,
        "query_mode": query_mode,
        "blocked_components": blocked,
        "top_k": top_k,
        "allowed_components": query_mode_policy.get("allowed_components"),
    }


def _fit_query_length(audio: np.ndarray, sr: int, length_seconds: float = QUERYSEP_BANQUET_QUERY_LENGTH_SECONDS) -> np.ndarray:
    target_samples = max(1, int(round(sr * length_seconds)))
    if audio.shape[0] >= target_samples:
        return audio[:target_samples]
    repeats = (target_samples + audio.shape[0] - 1) // max(1, audio.shape[0])
    return np.tile(audio, repeats)[:target_samples]


def _load_stereo_segment(path: Path, start_sec: float, end_sec: float, target_sr: int = BANQUET_SR) -> np.ndarray:
    try:
        audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
        audio = np.asarray(audio, dtype=np.float32)
    except Exception:
        fallback, sr = librosa.load(str(path), sr=None, mono=False)
        audio = np.asarray(fallback, dtype=np.float32)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
            audio = audio.T
        elif audio.shape[0] <= 4:
            audio = audio.T

    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    elif audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]

    if sr != target_sr:
        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=target_sr, axis=-1).T.astype(np.float32)
        sr = target_sr
    start_sample = max(0, int(round(start_sec * sr)))
    end_sample = max(start_sample + 1, int(round(end_sec * sr)))
    return audio[start_sample:end_sample].astype(np.float32, copy=False)


def _load_query_candidate_audio(
    component_name: str,
    component_meta: dict[str, dict[str, object]],
    window: dict[str, object],
) -> tuple[np.ndarray, str, str | None]:
    raw_source_path = component_meta.get(component_name, {}).get("source_path")
    source_path = Path(str(raw_source_path)) if raw_source_path else None
    query_audio: np.ndarray | None = None
    query_origin = "derived_component_mono"

    if source_path and source_path.exists() and component_meta.get(component_name, {}).get("origin") == "build_from_bus":
        try:
            query_audio = _load_stereo_segment(
                source_path,
                float(window["start_sec"]),
                float(window["end_sec"]),
                target_sr=BANQUET_SR,
            )
            query_origin = "bus_stereo_segment"
        except Exception:
            query_audio = None

    if query_audio is None:
        mono_audio = np.asarray(window["audio"], dtype=np.float32)
        if TARGET_SR != BANQUET_SR:
            mono_audio = librosa.resample(mono_audio, orig_sr=TARGET_SR, target_sr=BANQUET_SR).astype(np.float32)
        mono_audio = _fit_query_length(mono_audio, BANQUET_SR)
        query_audio = np.stack([mono_audio, mono_audio], axis=1)
    else:
        left = _fit_query_length(query_audio[:, 0], BANQUET_SR)
        right = _fit_query_length(query_audio[:, 1], BANQUET_SR)
        query_audio = np.stack([left, right], axis=1)

    return query_audio.astype(np.float32, copy=False), query_origin, str(source_path) if source_path else None


def materialize_query_exemplar(
    output_root: Path,
    component_meta: dict[str, dict[str, object]],
    ensemble_candidates: list[dict[str, object]],
) -> tuple[Path, dict[str, object]]:
    query_path = output_root / "query_exemplar.wav"
    if not ensemble_candidates:
        raise RuntimeError("Banquet query ensemble is empty.")

    blended_audio: np.ndarray | None = None
    members: list[dict[str, object]] = []

    for candidate in ensemble_candidates:
        component_name = str(candidate["component"])
        weight = float(candidate.get("weight", 1.0))
        window = dict(candidate["window"])
        candidate_audio, query_origin, source_path = _load_query_candidate_audio(
            component_name,
            component_meta,
            window,
        )
        if blended_audio is None:
            blended_audio = np.zeros_like(candidate_audio, dtype=np.float32)
        blended_audio += candidate_audio * weight
        members.append(
            {
                "component": component_name,
                "weight": round(weight, 6),
                "query_origin": query_origin,
                "query_source_path": source_path,
                "start_sec": float(window["start_sec"]),
                "end_sec": float(window["end_sec"]),
                "similarity": float(window["similarity"]),
                "adjusted_similarity": float(candidate["adjusted_similarity"]),
                "window_index": int(candidate["window_index"]),
            }
        )

    assert blended_audio is not None
    blended_audio = peak_normalize(blended_audio)
    sf.write(str(query_path), blended_audio, BANQUET_SR, subtype="FLOAT")

    primary = members[0]
    return query_path, {
        "query_path": str(query_path),
        "query_origin": "ensemble_mix" if len(members) > 1 else primary["query_origin"],
        "query_component": primary["component"],
        "query_start_sec": primary["start_sec"],
        "query_end_sec": primary["end_sec"],
        "query_similarity": primary["similarity"],
        "query_source_path": primary["query_source_path"],
        "query_members": members,
    }


def planned_outputs(request: RunRequest, output_root: Path) -> dict[str, str]:
    target = request.target_stem or request.prompt or request.task
    return {target: str(output_path_for(request, output_root))}


@app.get("/health")
def health() -> dict:
    unloaded = {
        "clap_router": maybe_unload_idle_clap_runtime(),
        "banquet_byoq": maybe_unload_idle_banquet_runtime(),
    }
    return {
        "status": "ok",
        "service": "querysep-worker",
        "dry_run": DRY_RUN,
        "idle_unload": unloaded,
        "startup_prefetch": getattr(app.state, "startup_prefetch", {}),
        "startup_prefetch_errors": getattr(app.state, "startup_prefetch_errors", {}),
        "runtime": {
            "clap_router": clap_cache_snapshot(),
            "banquet_byoq": banquet_cache_snapshot(),
        },
    }


@app.get("/capabilities")
def capabilities() -> dict:
    return {
        "service": "querysep-worker",
        "tasks": ["prompt_extract", "residual_export"],
        "supported_prompts": sorted(PROMPT_TEXTS.keys()),
        "backends": {
            "clap_router": "prompt-conditioned ML router over candidate buses and derived components",
            "banquet_byoq": "real Banquet BYOQ decoder with CLAP-derived query exemplar selection",
        },
        "banquet_query_modes": sorted(SUPPORTED_BANQUET_QUERY_MODES),
        "banquet": banquet_ready_state(),
    }


def run_clap_router(
    request: RunRequest,
    prompt: str,
    output_root: Path,
    output_path: Path,
    metadata: dict[str, object],
) -> tuple[str, str]:
    components, component_meta = build_component_bank(request, prompt)
    scores, _, _, runtime_meta = score_components(prompt, components)
    mixed_audio, selected_components = mix_top_components(components, scores)
    sf.write(str(output_path), mixed_audio, TARGET_SR, subtype="FLOAT")
    metadata["components"] = component_meta
    metadata["component_scores"] = {
        key: round(float(value), 6) for key, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    }
    metadata["selected_components"] = selected_components
    metadata["runtime"] = {"clap_router": runtime_meta, "banquet_byoq": banquet_cache_snapshot()}
    return (
        "completed" if output_path.exists() else "failed",
        (
            "Real prompt-conditioned ML routing completed with CLAP audio-text similarity over "
            "candidate buses and derived input components. This is routing, not true query-separation."
        ),
    )


def run_banquet_backend(
    request: RunRequest,
    prompt: str,
    output_root: Path,
    output_path: Path,
    metadata: dict[str, object],
) -> tuple[str, str]:
    ready = banquet_ready_state()
    if not ready["checkpoint_exists"]:
        raise FileNotFoundError(
            f"Banquet backend requested but checkpoint is missing at {ready['checkpoint_path']}."
        )

    components, component_meta = build_component_bank(request, prompt)
    scores, best_windows, ranked_windows, clap_runtime_meta = score_components(prompt, components)
    query_mode = resolve_banquet_query_mode(request, prompt)
    candidate_selection = select_banquet_query_candidates(
        prompt,
        scores,
        best_windows,
        ranked_windows,
        query_mode=query_mode,
    )
    best_candidate = candidate_selection["best"]
    query_path, query_meta = materialize_query_exemplar(
        output_root,
        component_meta,
        candidate_selection["ensemble"],
    )
    banquet_runtime_meta = run_banquet_query(
        Path(request.input_path),
        query_path,
        output_path,
        stem_name=request.target_stem or prompt or "target",
    )

    metadata["components"] = component_meta
    metadata["component_scores"] = {
        key: round(float(value), 6) for key, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    }
    metadata["query_mode"] = candidate_selection["query_mode"]
    metadata["query_selection"] = {
        "mode": candidate_selection["query_mode"],
        "top_k": candidate_selection["top_k"],
        "allowed_components": (
            list(candidate_selection["allowed_components"])
            if candidate_selection.get("allowed_components") is not None
            else None
        ),
        "blocked_component_count": len(candidate_selection["blocked_components"]),
        "blocked_components": candidate_selection["blocked_components"][:16],
    }
    metadata["query_candidate"] = {
        "component": best_candidate["component"],
        "base_similarity": best_candidate["base_similarity"],
        "window_similarity": best_candidate["window_similarity"],
        "adjusted_similarity": best_candidate["adjusted_similarity"],
        "window_index": best_candidate["window_index"],
        "window": {
            "start_sec": best_candidate["window"]["start_sec"],
            "end_sec": best_candidate["window"]["end_sec"],
            "rms": best_candidate["window"]["rms"],
            "similarity": best_candidate["window"]["similarity"],
        },
    }
    metadata["query_candidate_ranking"] = [
        {
            "component": item["component"],
            "base_similarity": item["base_similarity"],
            "window_similarity": item["window_similarity"],
            "adjusted_similarity": item["adjusted_similarity"],
            "weight": item.get("weight"),
            "window_index": item["window_index"],
            "window_start_sec": item["window"]["start_sec"],
            "window_end_sec": item["window"]["end_sec"],
        }
        for item in candidate_selection["ranked"][:8]
    ]
    metadata["query_ensemble"] = [
        {
            "component": item["component"],
            "base_similarity": item["base_similarity"],
            "window_similarity": item["window_similarity"],
            "adjusted_similarity": item["adjusted_similarity"],
            "weight": item["weight"],
            "window_index": item["window_index"],
            "window_start_sec": item["window"]["start_sec"],
            "window_end_sec": item["window"]["end_sec"],
        }
        for item in candidate_selection["ensemble"]
    ]
    metadata["query_exemplar"] = query_meta
    metadata["runtime"] = {
        "clap_router": clap_runtime_meta,
        "banquet_byoq": banquet_runtime_meta,
    }
    return (
        "completed" if output_path.exists() else "failed",
        (
            "Real Banquet BYOQ separation completed. CLAP was used only to derive an ensemble query exemplar from "
            "available buses/components; the actual target stem was decoded by Banquet."
        ),
    )


@app.post("/run")
def run(request: RunRequest) -> dict:
    output_root = output_root_for(request)
    output_path = output_path_for(request, output_root)
    prompt = resolve_prompt(request)
    backend = resolve_backend(request, prompt)
    named_outputs = planned_outputs(request, output_root)

    metadata = {
        "prompt": prompt,
        "prompt_text": PROMPT_TEXTS[prompt],
        "output_root": str(output_root),
        "routing_mode": backend,
        "banquet_state": banquet_ready_state(),
    }
    status = "dry_run_completed"
    notes = f"Dry-run {backend} request assembled."
    command = [backend, prompt, request.input_path]

    if not request.dry_run:
        input_path = Path(request.input_path)
        if not input_path.exists():
            status = "failed"
            notes = "Query extraction could not start because the input bus was missing."
        else:
            try:
                if backend == "banquet_byoq":
                    status, notes = run_banquet_backend(request, prompt, output_root, output_path, metadata)
                    command.append(str(output_root / "query_exemplar.wav"))
                else:
                    status, notes = run_clap_router(request, prompt, output_root, output_path, metadata)
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                metadata["error"] = str(exc)
                notes = f"{backend} failed during query selection or output synthesis."

    result = RunResult(
        service="querysep-worker",
        status=status,
        task=request.task,
        target_stem=request.target_stem,
        command=command,
        output_files=list(named_outputs.values()),
        named_outputs=named_outputs if status == "completed" else {},
        metadata=metadata,
        notes=notes,
    )
    return result.model_dump()
