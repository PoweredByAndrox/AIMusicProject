from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Mapping

import essentia.standard as essentia_std
import librosa
import numpy as np
import soundfile as sf

from worker.instrument_melody_profile import (
    FEATURE_SAMPLE_RATE,
    NOTE_NAMES,
    separate_track_stems,
)

try:  # pragma: no cover
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover
    tf = None


EMBED_SAMPLE_RATE = 16000
EFFNET_PATCH_SIZE = 128
EFFNET_PATCH_HOP = 62
MUSICNN_PATCH_SIZE = 187
MUSICNN_PATCH_HOP = 93
EFFNET_BATCH_SIZE = 64
DEFAULT_SEPARATOR_MODEL = "htdemucs_6s"
DEFAULT_CACHE_ROOT = Path(os.getenv("TRACK_INSTRUMENT_PROFILE_CACHE_DIR", "/app/storage/tmp/track_instrument_profile"))
DEFAULT_BASE_MODELS_DIR = Path(os.getenv("MODELS_PATH", "/app/models"))
EFFECTIVE_ROLES = ("bass", "chords", "fx", "melody", "percussion")
EXTRA_MODEL_URLS = {
    "mtg_jamendo_instrument-discogs-effnet-1.pb": "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb",
    "mtg_jamendo_instrument-discogs-effnet-1.json": "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.json",
    "nsynth_instrument-discogs-effnet-1.pb": "https://essentia.upf.edu/models/classification-heads/nsynth_instrument/nsynth_instrument-discogs-effnet-1.pb",
    "nsynth_instrument-discogs-effnet-1.json": "https://essentia.upf.edu/models/classification-heads/nsynth_instrument/nsynth_instrument-discogs-effnet-1.json",
    "timbre-discogs-effnet-1.pb": "https://essentia.upf.edu/models/classification-heads/timbre/timbre-discogs-effnet-1.pb",
    "timbre-discogs-effnet-1.json": "https://essentia.upf.edu/models/classification-heads/timbre/timbre-discogs-effnet-1.json",
    "msd-musicnn-1.pb": "https://essentia.upf.edu/models/autotagging/msd/msd-musicnn-1.pb",
    "fs_loop_ds-msd-musicnn-1.pb": "https://essentia.upf.edu/models/classification-heads/fs_loop_ds/fs_loop_ds-msd-musicnn-1.pb",
    "fs_loop_ds-msd-musicnn-1.json": "https://essentia.upf.edu/models/classification-heads/fs_loop_ds/fs_loop_ds-msd-musicnn-1.json",
}
EMBED_MODEL_URL = "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb"


class FrozenGraphRunner:
    """Small TF1 runner for frozen GraphDef models with one or more inputs."""

    def __init__(self, pb_path: Path, inputs: Mapping[str, str], output_tensor: str):
        if tf is None:
            raise RuntimeError("tensorflow is required for track instrument profile inference")

        tf.compat.v1.disable_eager_execution()

        self._graph = tf.compat.v1.Graph()
        with self._graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(Path(pb_path).read_bytes())
            tf.import_graph_def(graph_def, name="")
            self._inputs = {
                key: self._graph.get_tensor_by_name(tensor_name)
                for key, tensor_name in inputs.items()
            }
            self._output = self._graph.get_tensor_by_name(output_tensor)

        cfg = tf.compat.v1.ConfigProto()
        try:  # pragma: no cover
            cfg.gpu_options.allow_growth = True  # type: ignore[attr-defined]
        except Exception:
            pass
        self._sess = tf.compat.v1.Session(graph=self._graph, config=cfg)

    def predict(self, feeds: Mapping[str, Any]) -> np.ndarray:
        return np.asarray(
            self._sess.run(
                self._output,
                feed_dict={
                    self._inputs[name]: value
                    for name, value in feeds.items()
                },
            ),
            dtype=np.float32,
        )


def _download_if_missing(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    with contextlib.closing(urllib.request.urlopen(url)) as response:
        destination.write_bytes(response.read())
    return destination


def _ensure_auxiliary_models(
    cache_root: Path,
    *,
    base_models_dir: Path = DEFAULT_BASE_MODELS_DIR,
) -> dict[str, Path]:
    model_root = cache_root / "models"
    model_root.mkdir(parents=True, exist_ok=True)

    embed_model = base_models_dir / "discogs-effnet-bs64-1.pb"
    if not embed_model.exists():
        embed_model = _download_if_missing(EMBED_MODEL_URL, model_root / "discogs-effnet-bs64-1.pb")

    resolved = {"discogs_effnet": embed_model}
    for file_name, url in EXTRA_MODEL_URLS.items():
        resolved[file_name] = _download_if_missing(url, model_root / file_name)
    return resolved


def _load_labels(json_path: Path) -> list[str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for key in ("classes", "class_labels", "labels", "classes_labels"):
        value = data.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    raise RuntimeError(f"Unsupported label metadata schema: {json_path}")


def _resample(audio: np.ndarray, target_sr: int) -> np.ndarray:
    if FEATURE_SAMPLE_RATE == target_sr:
        return np.asarray(audio, dtype=np.float32)
    return librosa.resample(
        np.asarray(audio, dtype=np.float32),
        orig_sr=FEATURE_SAMPLE_RATE,
        target_sr=target_sr,
    ).astype(np.float32)


def _build_patches(audio: np.ndarray, *, patch_size: int, patch_hop: int) -> np.ndarray:
    tf_input = essentia_std.TensorflowInputMusiCNN()
    bands = [
        np.asarray(tf_input(np.asarray(frame, dtype=np.float32)), dtype=np.float32)
        for frame in essentia_std.FrameGenerator(audio, frameSize=512, hopSize=256, startFromZero=True)
    ]
    if not bands:
        bands = [np.zeros((96,), dtype=np.float32)]

    stacked = np.asarray(bands, dtype=np.float32)
    if stacked.shape[0] < patch_size:
        pad_count = patch_size - stacked.shape[0]
        pad_frame = stacked[-1:] if stacked.size else np.zeros((1, 96), dtype=np.float32)
        stacked = np.concatenate([stacked, np.repeat(pad_frame, pad_count, axis=0)], axis=0)

    max_start = max(0, int(stacked.shape[0] - patch_size))
    starts = list(range(0, max_start + 1, patch_hop)) or [0]
    return np.asarray([stacked[start:start + patch_size] for start in starts], dtype=np.float32)


def _predict_effnet_embedding(audio: np.ndarray, embedder: FrozenGraphRunner) -> np.ndarray:
    patches = _build_patches(_resample(audio, EMBED_SAMPLE_RATE), patch_size=EFFNET_PATCH_SIZE, patch_hop=EFFNET_PATCH_HOP)
    original_count = patches.shape[0]
    remainder = original_count % EFFNET_BATCH_SIZE
    if remainder:
        pad = np.repeat(patches[-1:], EFFNET_BATCH_SIZE - remainder, axis=0)
        patches = np.concatenate([patches, pad], axis=0)

    outputs = []
    for index in range(0, patches.shape[0], EFFNET_BATCH_SIZE):
        outputs.append(embedder.predict({"patches": patches[index:index + EFFNET_BATCH_SIZE]}))
    patch_embeddings = np.concatenate(outputs, axis=0)[:original_count]
    return np.mean(patch_embeddings, axis=0, keepdims=True).astype(np.float32)


def _predict_musicnn_embedding(audio: np.ndarray, musicnn_embedder: FrozenGraphRunner) -> np.ndarray:
    patches = _build_patches(_resample(audio, EMBED_SAMPLE_RATE), patch_size=MUSICNN_PATCH_SIZE, patch_hop=MUSICNN_PATCH_HOP)
    patch_embeddings = musicnn_embedder.predict({"bands": patches, "training": False})
    return np.mean(np.asarray(patch_embeddings, dtype=np.float32), axis=0, keepdims=True)


def _format_label_scores(labels: list[str], scores: np.ndarray, *, limit: int, min_score: float = 0.0) -> list[dict[str, Any]]:
    flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    ordered_indexes = flat_scores.argsort()[::-1]
    results: list[dict[str, Any]] = []
    for index in ordered_indexes:
        score = float(flat_scores[int(index)])
        if score < min_score and results:
            break
        results.append({"label": labels[int(index)], "score": round(score, 4)})
        if len(results) >= limit:
            break
    return results


def _summarize_notes_subprocess(audio_paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    if not audio_paths:
        return {}

    script = """
import contextlib
import io
import json
import sys
from pathlib import Path
import basic_pitch
import basic_pitch.inference as basic_pitch_inference
from basic_pitch.inference import predict

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
DEFAULT_MODEL_ROOT = Path(basic_pitch.__file__).resolve().parent / 'saved_models' / 'icassp_2022'
DEFAULT_TFLITE_MODEL_PATH = DEFAULT_MODEL_ROOT / 'nmp.tflite'
DEFAULT_ONNX_MODEL_PATH = DEFAULT_MODEL_ROOT / 'nmp.onnx'
MODEL_PATH = DEFAULT_TFLITE_MODEL_PATH

try:
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        _original_model_init = basic_pitch_inference.Model.__init__

        def _patched_model_init(self, model_path):
            model_path = Path(model_path)
            if str(model_path).endswith('.onnx'):
                self.model_type = basic_pitch_inference.Model.MODEL_TYPES.ONNX
                self.model = ort.InferenceSession(
                    str(model_path),
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                )
                return
            return _original_model_init(self, model_path)

        basic_pitch_inference.Model.__init__ = _patched_model_init
        MODEL_PATH = DEFAULT_ONNX_MODEL_PATH
except Exception:
    MODEL_PATH = DEFAULT_TFLITE_MODEL_PATH

def summarize(path):
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, note_events = predict(path, model_or_model_path=str(MODEL_PATH))
    note_name_counts = {}
    note_class_counts = {}
    confidences = []
    min_midi = None
    max_midi = None
    for start, end, pitch_midi, confidence, pitch_bend in note_events:
        midi = int(round(float(pitch_midi)))
        octave = (midi // 12) - 1
        note_name = f"{NOTE_NAMES[midi % 12]}{octave}"
        note_class = NOTE_NAMES[midi % 12]
        note_name_counts[note_name] = note_name_counts.get(note_name, 0) + 1
        note_class_counts[note_class] = note_class_counts.get(note_class, 0) + 1
        confidences.append(float(confidence))
        min_midi = midi if min_midi is None else min(min_midi, midi)
        max_midi = midi if max_midi is None else max(max_midi, midi)
    top_note_names = sorted(note_name_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    top_note_classes = sorted(note_class_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    return {
        'note_event_count': len(note_events),
        'top_note_names': [{'note': name, 'count': int(count)} for name, count in top_note_names],
        'top_note_classes': [{'note': name, 'count': int(count)} for name, count in top_note_classes],
        'pitch_range_midi': {'min': min_midi, 'max': max_midi} if min_midi is not None else None,
        'mean_confidence': round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
    }

payload = {}
args = sys.argv[1:]
for index in range(0, len(args), 2):
    stem_name = args[index]
    audio_path = args[index + 1]
    payload[stem_name] = summarize(audio_path)
print(json.dumps(payload))
"""
    command = [sys.executable, "-c", script]
    for stem_name, audio_path in audio_paths.items():
        command.extend([stem_name, str(audio_path)])
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or f"exit_code={exc.returncode}"
        raise RuntimeError(f"basic_pitch note summarizer subprocess failed: {detail}") from exc
    stdout = completed.stdout.strip()
    if not stdout:
        stderr = completed.stderr.strip()
        detail = stderr or "empty stdout"
        raise RuntimeError(f"basic_pitch note summarizer returned no JSON output: {detail}")
    return json.loads(stdout)


def _aggregate_global_note_inventory(stems: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    note_name_counts: dict[str, int] = {}
    note_class_counts: dict[str, int] = {}

    for stem_data in stems.values():
        notes = stem_data.get("note_inventory")
        if not isinstance(notes, dict):
            continue
        for item in notes.get("top_note_names", []):
            note_name_counts[item["note"]] = note_name_counts.get(item["note"], 0) + int(item["count"])
        for item in notes.get("top_note_classes", []):
            note_class_counts[item["note"]] = note_class_counts.get(item["note"], 0) + int(item["count"])

    top_note_names = sorted(note_name_counts.items(), key=lambda item: (-item[1], item[0]))[:12]
    top_note_classes = sorted(note_class_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    return {
        "top_note_names": [{"note": note, "count": int(count)} for note, count in top_note_names],
        "top_note_classes": [{"note": note, "count": int(count)} for note, count in top_note_classes],
    }


def _profile_cache_path(audio_path: Path, separator_model: str, cache_root: Path) -> Path:
    stat = audio_path.stat()
    fingerprint = hashlib.sha1(
        "|".join(
            [
                str(audio_path.resolve()),
                str(int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))),
                str(int(stat.st_size)),
                separator_model,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return cache_root / "profiles" / f"{audio_path.stem}_{fingerprint}.json"


def build_track_instrument_profile(
    audio_path: str | Path,
    *,
    separator_model: str = DEFAULT_SEPARATOR_MODEL,
    device: str | None = None,
    cache_root: str | Path | None = None,
    base_models_dir: str | Path | None = None,
) -> dict[str, Any]:
    resolved_audio_path = Path(audio_path)
    if not resolved_audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {resolved_audio_path}")

    resolved_cache_root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
    resolved_base_models_dir = Path(base_models_dir) if base_models_dir is not None else DEFAULT_BASE_MODELS_DIR
    resolved_cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = _profile_cache_path(resolved_audio_path, separator_model, resolved_cache_root)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    model_paths = _ensure_auxiliary_models(resolved_cache_root, base_models_dir=resolved_base_models_dir)
    label_maps = {
        "jamendo": _load_labels(model_paths["mtg_jamendo_instrument-discogs-effnet-1.json"]),
        "nsynth": _load_labels(model_paths["nsynth_instrument-discogs-effnet-1.json"]),
        "timbre": _load_labels(model_paths["timbre-discogs-effnet-1.json"]),
        "roles": _load_labels(model_paths["fs_loop_ds-msd-musicnn-1.json"]),
    }

    stems = separate_track_stems(resolved_audio_path, model_name=separator_model, device=device)

    embedder = FrozenGraphRunner(
        model_paths["discogs_effnet"],
        inputs={"patches": "serving_default_melspectrogram:0"},
        output_tensor="PartitionedCall:1",
    )
    jamendo_head = FrozenGraphRunner(
        model_paths["mtg_jamendo_instrument-discogs-effnet-1.pb"],
        inputs={"embedding": "model/Placeholder:0"},
        output_tensor="model/Sigmoid:0",
    )
    nsynth_head = FrozenGraphRunner(
        model_paths["nsynth_instrument-discogs-effnet-1.pb"],
        inputs={"embedding": "model/Placeholder:0"},
        output_tensor="model/Softmax:0",
    )
    timbre_head = FrozenGraphRunner(
        model_paths["timbre-discogs-effnet-1.pb"],
        inputs={"embedding": "model/Placeholder:0"},
        output_tensor="model/Softmax:0",
    )
    musicnn_embedder = FrozenGraphRunner(
        model_paths["msd-musicnn-1.pb"],
        inputs={
            "bands": "model/Placeholder:0",
            "training": "model/Placeholder_1:0",
        },
        output_tensor="model/dense/BiasAdd:0",
    )
    role_head = FrozenGraphRunner(
        model_paths["fs_loop_ds-msd-musicnn-1.pb"],
        inputs={"embedding": "serving_default_model_Placeholder:0"},
        output_tensor="PartitionedCall:0",
    )

    energy_by_stem = {
        stem_name: float(np.sqrt(np.mean(np.asarray(stem_audio, dtype=np.float32) ** 2))) if np.asarray(stem_audio).size else 0.0
        for stem_name, stem_audio in stems.items()
    }
    total_energy = sum(energy_by_stem.values()) or 1.0

    result: dict[str, Any] = {
        "audio_path": str(resolved_audio_path),
        "separator_model": separator_model,
        "sample_rate": FEATURE_SAMPLE_RATE,
        "stems": {},
        "global_note_inventory": {},
        "warnings": [],
    }

    transcription_candidates: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="track_instrument_profile_") as temp_dir:
        temp_root = Path(temp_dir)
        for stem_name, stem_audio in stems.items():
            effnet_embedding = _predict_effnet_embedding(stem_audio, embedder)
            musicnn_embedding = _predict_musicnn_embedding(stem_audio, musicnn_embedder)

            jamendo_scores = jamendo_head.predict({"embedding": effnet_embedding})[0]
            nsynth_scores = nsynth_head.predict({"embedding": effnet_embedding})[0]
            timbre_scores = timbre_head.predict({"embedding": effnet_embedding})[0]
            role_scores = role_head.predict({"embedding": musicnn_embedding})[0]

            top_roles = _format_label_scores(label_maps["roles"], role_scores, limit=3)
            stem_result = {
                "energy_ratio": round(float(energy_by_stem[stem_name] / total_energy), 4),
                "instrument_labels": {
                    "jamendo": _format_label_scores(label_maps["jamendo"], jamendo_scores, limit=5, min_score=0.08),
                    "nsynth": _format_label_scores(label_maps["nsynth"], nsynth_scores, limit=3, min_score=0.01),
                    "timbre": _format_label_scores(label_maps["timbre"], timbre_scores, limit=2),
                },
                "role_labels": top_roles,
                "note_inventory": {
                    "transcription_mode": "pending",
                    "note_event_count": 0,
                    "top_note_names": [],
                    "top_note_classes": [],
                    "pitch_range_midi": None,
                    "mean_confidence": 0.0,
                },
            }

            if stem_name == "drums" or (top_roles and top_roles[0]["label"] == "percussion"):
                stem_result["note_inventory"]["transcription_mode"] = "skipped_percussion"
            else:
                stem_wav_path = temp_root / f"{stem_name}.wav"
                sf.write(stem_wav_path, np.asarray(stem_audio, dtype=np.float32), FEATURE_SAMPLE_RATE)
                transcription_candidates[stem_name] = stem_wav_path
                stem_result["note_inventory"]["transcription_mode"] = "basic_pitch_subprocess"

            result["stems"][stem_name] = stem_result

        note_summaries = _summarize_notes_subprocess(transcription_candidates)
        for stem_name, summary in note_summaries.items():
            result["stems"][stem_name]["note_inventory"].update(summary)

    result["global_note_inventory"] = _aggregate_global_note_inventory(result["stems"])

    for stem_name in ("guitar", "piano"):
        if stem_name in result["stems"]:
            top_role = result["stems"][stem_name]["role_labels"][0]["label"]
            if top_role == "fx":
                result["warnings"].append(
                    f"{stem_name} stem still looks mixed/noisy under {separator_model}; top role resolved as fx"
                )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
