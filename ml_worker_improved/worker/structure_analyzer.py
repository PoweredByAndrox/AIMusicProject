"""
Structure Analyzer - Music structure analysis using All-In-One model.

Uses allin1 (All-In-One Music Structure Analyzer) for:
- BPM detection (precise)
- Beat and downbeat tracking
- Segment boundary detection
- Segment label classification (intro, verse, chorus, bridge, outro, etc.)

Model: All-In-One (Taejun Kim et al.)
Paper: "All-In-One Metrical And Functional Structure Analysis With
        Neighborhood Attentions on Demixed Audio"

REQUIRES: allin1 package with natten backend (no fallback)
"""

import importlib
import json
import logging
import os
import socket
import shutil
import tempfile
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import redis
import torch
import allin1  # Required - no fallback
import allin1.models.loaders as allin1_loaders
from .device_policy import resolve_torch_device, visible_cuda_summary
from .disk_telemetry import emit_disk_telemetry, snapshot_disk_state

logger = logging.getLogger(__name__)
disk_logger = logging.getLogger('disk_telemetry')

DEVICE = str(resolve_torch_device(task_name='StructureAnalyzer', allow_cpu=False))
PATCH_MARKER_PATH = (
    Path(os.getenv('STORAGE_PATH', '/app/storage')) / 'logs' / 'structure_analyzer_patch.log'
)
_LOCAL_MODEL_CACHE: Dict[Tuple[str, str], torch.nn.Module] = {}
STRUCTURE_GPU_LOCK_KEY = (
    f"{os.getenv('REDIS_PREFIX', 'ai_music_recommendations_database_')}ml_worker:structure_gpu_lock"
)
STRUCTURE_GPU_LOCK_TTL_SECONDS = max(30, int(os.getenv('STRUCTURE_GPU_LOCK_TTL_SECONDS', '900')))
STRUCTURE_GPU_LOCK_WAIT_SECONDS = max(1, int(os.getenv('STRUCTURE_GPU_LOCK_WAIT_SECONDS', '900')))
STRUCTURE_GPU_LOCK_POLL_SECONDS = max(0.05, float(os.getenv('STRUCTURE_GPU_LOCK_POLL_SECONDS', '0.25')))
STRUCTURE_CACHE_DIR = (
    Path(os.getenv('STORAGE_PATH', '/app/storage')) / 'analysis_cache' / 'structure'
)


def _append_patch_marker(event: str, **payload: Any) -> None:
    """Write an unconditional host-visible marker before logging is configured."""
    try:
        PATCH_MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {'event': event, 'payload': payload}
        with PATCH_MARKER_PATH.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + '\n')
    except Exception:
        pass


_append_patch_marker('module.loaded', device=DEVICE, visible_cuda=visible_cuda_summary())
logger.info(f"allin1 module loaded, device={DEVICE}")


def _create_structure_lock_redis_client() -> redis.Redis:
    """Create a small Redis client for cross-worker structure GPU locking."""
    return redis.Redis(
        host=os.getenv('REDIS_HOST', 'redis'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD') or None,
        decode_responses=True,
    )


def _release_structure_gpu_lock(redis_client: redis.Redis, owner_token: str) -> None:
    """Release the structure GPU lock only if we still own it."""
    release_script = """
    if redis.call('get', KEYS[1]) == ARGV[1] then
        return redis.call('del', KEYS[1])
    end
    return 0
    """
    redis_client.eval(release_script, 1, STRUCTURE_GPU_LOCK_KEY, owner_token)


@contextmanager
def _structure_gpu_lock(device: str, audio_count: int):
    """Serialize allin1/demucs GPU work across workers to avoid OOM kills."""
    if device != 'cuda':
        yield
        return

    redis_client = _create_structure_lock_redis_client()
    owner_token = f"{socket.gethostname()}:{os.getpid()}:{time.time_ns()}"
    deadline = time.monotonic() + STRUCTURE_GPU_LOCK_WAIT_SECONDS
    wait_logged = False

    while True:
        acquired = redis_client.set(
            STRUCTURE_GPU_LOCK_KEY,
            owner_token,
            nx=True,
            ex=STRUCTURE_GPU_LOCK_TTL_SECONDS,
        )
        if acquired:
            logger.info(
                "Acquired structure GPU lock: key=%s audio_count=%s ttl=%ss",
                STRUCTURE_GPU_LOCK_KEY,
                audio_count,
                STRUCTURE_GPU_LOCK_TTL_SECONDS,
            )
            break

        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Timed out waiting for structure GPU lock "
                f"(key={STRUCTURE_GPU_LOCK_KEY}, wait={STRUCTURE_GPU_LOCK_WAIT_SECONDS}s)"
            )

        if not wait_logged:
            logger.info(
                "Waiting for structure GPU lock: key=%s audio_count=%s",
                STRUCTURE_GPU_LOCK_KEY,
                audio_count,
            )
            wait_logged = True

        time.sleep(STRUCTURE_GPU_LOCK_POLL_SECONDS)

    try:
        yield
    finally:
        try:
            _release_structure_gpu_lock(redis_client, owner_token)
            logger.info("Released structure GPU lock: key=%s", STRUCTURE_GPU_LOCK_KEY)
        finally:
            try:
                redis_client.close()
            except Exception:
                pass


def _resolve_hf_hub_cache_dir(cache_dir: Optional[str] = None) -> Path:
    raw_cache_dir = (
        cache_dir
        or os.getenv('HUGGINGFACE_HUB_CACHE')
        or os.getenv('HF_HOME')
        or '/app/storage/cache/huggingface/hub'
    )
    cache_path = Path(raw_cache_dir)
    if cache_path.name != 'hub' and (cache_path / 'hub').exists():
        return cache_path / 'hub'
    return cache_path


def _resolve_allin1_checkpoint(filename: str, cache_dir: Optional[str] = None) -> Path:
    cache_root = _resolve_hf_hub_cache_dir(cache_dir)
    repo_dir = cache_root / 'models--taejunkim--allinone'
    ref_path = repo_dir / 'refs' / 'main'
    candidates: List[Path] = []

    if ref_path.exists():
        revision = ref_path.read_text(encoding='utf-8').strip()
        if revision:
            candidates.append(repo_dir / 'snapshots' / revision / filename)

    snapshots_dir = repo_dir / 'snapshots'
    if snapshots_dir.exists():
        candidates.extend(sorted(snapshots_dir.glob(f'*/{filename}')))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Missing cached allin1 checkpoint '{filename}' under '{cache_root}'."
    )


def _patch_allin1_hf_loader() -> None:
    """Force allin1 to use local cached checkpoints and never hit Hugging Face at runtime."""
    _append_patch_marker(
        'patch.enter',
        cache_dir=str(_resolve_hf_hub_cache_dir()),
    )
    if getattr(allin1_loaders, "_aimusic_local_cache_patch", False):
        _append_patch_marker('patch.skip_already_applied')
        return

    allin1_models = importlib.import_module('allin1.models')
    allin1_analyze_module = importlib.import_module('allin1.analyze')
    cache_dir = str(_resolve_hf_hub_cache_dir())
    name_to_file = dict(allin1_loaders.NAME_TO_FILE)
    ensemble_models = {
        model_name: list(model_names)
        for model_name, model_names in allin1_loaders.ENSEMBLE_MODELS.items()
    }

    def _hf_hub_download_local(*args, **kwargs):
        filename = kwargs.get('filename')
        if filename is None and len(args) >= 2:
            filename = args[1]
        checkpoint_path = _resolve_allin1_checkpoint(str(filename), cache_dir=cache_dir)
        _append_patch_marker(
            'hf_hub_download.intercepted',
            filename=str(filename),
            checkpoint_path=str(checkpoint_path),
        )
        return str(checkpoint_path)

    def _load_pretrained_model_local(
        model_name: Optional[str] = None,
        cache_dir_arg: Optional[str] = None,
        device: Optional[str] = None,
    ):
        target_device = device or ('cuda' if torch.cuda.device_count() else 'cpu')
        target_model_name = model_name or list(name_to_file.keys())[0]
        _append_patch_marker(
            'load_pretrained_model.enter',
            model_name=target_model_name,
            device=target_device,
        )

        if target_model_name in ensemble_models:
            return _load_ensemble_model_local(target_model_name, cache_dir_arg, target_device)

        cache_key = (target_model_name, str(target_device))
        if cache_key in _LOCAL_MODEL_CACHE:
            _append_patch_marker(
                'load_pretrained_model.cache_hit',
                model_name=target_model_name,
                device=target_device,
            )
            return _LOCAL_MODEL_CACHE[cache_key]

        checkpoint_path = _resolve_allin1_checkpoint(
            name_to_file[target_model_name],
            cache_dir=cache_dir_arg or cache_dir,
        )
        checkpoint = torch.load(checkpoint_path, map_location=target_device)
        config = allin1_loaders.OmegaConf.create(checkpoint['config'])
        model = allin1_loaders.AllInOne(config).to(target_device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        _LOCAL_MODEL_CACHE[cache_key] = model
        _append_patch_marker(
            'load_pretrained_model.loaded',
            model_name=target_model_name,
            device=target_device,
            checkpoint_path=str(checkpoint_path),
        )
        return model

    def _load_ensemble_model_local(
        model_name: Optional[str] = None,
        cache_dir_arg: Optional[str] = None,
        device: Optional[str] = None,
    ):
        target_device = device or ('cuda' if torch.cuda.device_count() else 'cpu')
        target_model_name = model_name or list(ensemble_models.keys())[0]
        cache_key = (target_model_name, str(target_device))

        if cache_key in _LOCAL_MODEL_CACHE:
            _append_patch_marker(
                'load_ensemble_model.cache_hit',
                model_name=target_model_name,
                device=target_device,
            )
            return _LOCAL_MODEL_CACHE[cache_key]

        models = [
            _load_pretrained_model_local(child_model, cache_dir_arg, target_device)
            for child_model in ensemble_models[target_model_name]
        ]
        ensemble = allin1_loaders.Ensemble(models).to(target_device)
        ensemble.eval()
        _LOCAL_MODEL_CACHE[cache_key] = ensemble
        _append_patch_marker(
            'load_ensemble_model.loaded',
            model_name=target_model_name,
            device=target_device,
            child_count=len(models),
        )
        return ensemble

    allin1_loaders.hf_hub_download = _hf_hub_download_local
    allin1_loaders.load_pretrained_model = _load_pretrained_model_local
    allin1_loaders.load_ensemble_model = _load_ensemble_model_local
    allin1_models.load_pretrained_model = _load_pretrained_model_local
    allin1_analyze_module.load_pretrained_model = _load_pretrained_model_local
    allin1_loaders._aimusic_local_cache_patch = True
    _append_patch_marker(
        'patch.applied',
        cache_dir=cache_dir,
        targets=['allin1.models.load_pretrained_model', 'allin1.analyze.load_pretrained_model'],
    )
    logger.info("Patched allin1 loader for local-only checkpoint cache_dir=%s", cache_dir)


_patch_allin1_hf_loader()


@dataclass
class Segment:
    """A musical segment with timing and label."""
    start: float
    end: float
    label: str
    duration: float = field(init=False)

    def __post_init__(self):
        self.duration = self.end - self.start


@dataclass
class StructureAnalysisResult:
    """Result of music structure analysis."""
    bpm: float
    beats: List[float]
    downbeats: List[float]
    segments: List[Segment]
    duration: float
    beat_regularity: float = 0.0
    embeddings: Optional[np.ndarray] = None
    activations: Optional[Dict[str, np.ndarray]] = None


class StructureAnalyzer:
    """
    Music structure analyzer using All-In-One neural network.

    Provides accurate:
    - BPM detection
    - Beat/downbeat tracking
    - Segment boundaries and labels

    REQUIRES: allin1 package (no fallback)
    """

    def __init__(self, model_name: str = 'harmonix-all'):
        """
        Initialize the analyzer.

        Args:
            model_name: Model to use:
                - 'harmonix-all': Ensemble of 8 models (best accuracy)
                - 'harmonix-fold0' to 'harmonix-fold7': Single models (faster)
        """
        self.model_name = model_name
        self.device = DEVICE
        logger.info(f"StructureAnalyzer initialized with model={model_name}, device={DEVICE}")

    def structure_cache_path(self, track_id: int) -> Path:
        """Return the sidecar cache path for one track's structure analysis."""
        return STRUCTURE_CACHE_DIR / f"track_{int(track_id)}.json"

    @staticmethod
    def _audio_signature(audio_path: str) -> Tuple[int, int]:
        stat = Path(audio_path).stat()
        return int(getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1_000_000_000))), int(stat.st_size)

    def save_cached_result(self, track_id: int, audio_path: str, result: StructureAnalysisResult) -> Path:
        """Persist a structure result sidecar for exact reuse by segment analysis."""
        STRUCTURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        audio_mtime_ns, audio_size = self._audio_signature(audio_path)
        payload = {
            'track_id': int(track_id),
            'audio_path': str(audio_path),
            'audio_mtime_ns': audio_mtime_ns,
            'audio_size': audio_size,
            'model_name': self.model_name,
            'bpm': float(result.bpm),
            'beats': [float(x) for x in result.beats],
            'downbeats': [float(x) for x in result.downbeats],
            'duration': float(result.duration),
            'beat_regularity': float(result.beat_regularity),
            'segments': [
                {
                    'start': float(seg.start),
                    'end': float(seg.end),
                    'label': str(seg.label),
                }
                for seg in result.segments
            ],
        }
        cache_path = self.structure_cache_path(track_id)
        cache_path.write_text(json.dumps(payload, ensure_ascii=True), encoding='utf-8')
        return cache_path

    def load_cached_result(self, track_id: int, audio_path: str) -> Optional[StructureAnalysisResult]:
        """Load a previously persisted structure result when it still matches the audio file."""
        cache_path = self.structure_cache_path(track_id)
        if not cache_path.exists():
            return None

        try:
            payload = json.loads(cache_path.read_text(encoding='utf-8'))
            if payload.get('model_name') != self.model_name:
                return None

            audio_mtime_ns, audio_size = self._audio_signature(audio_path)
            if int(payload.get('audio_mtime_ns', -1)) != audio_mtime_ns:
                return None
            if int(payload.get('audio_size', -1)) != audio_size:
                return None

            segments = [
                Segment(
                    start=float(seg['start']),
                    end=float(seg['end']),
                    label=str(seg['label']),
                )
                for seg in payload.get('segments', [])
            ]
            return StructureAnalysisResult(
                bpm=float(payload.get('bpm', 0.0)),
                beats=[float(x) for x in payload.get('beats', [])],
                downbeats=[float(x) for x in payload.get('downbeats', [])],
                segments=segments,
                duration=float(payload.get('duration', 0.0)),
                beat_regularity=float(payload.get('beat_regularity', 0.0)),
            )
        except Exception as e:
            logger.warning(f"Failed to load structure cache for track {track_id}: {e}")
            return None

    def _convert_analysis_result(self, audio_path: str, result: Any, include_embeddings: bool, include_activations: bool) -> StructureAnalysisResult:
        """Convert allin1 result object into the local dataclass."""
        bpm: float = float(result.bpm) if result.bpm is not None else 0.0
        if result.bpm is None:
            logger.warning(
                f"allin1 returned bpm=None for {audio_path} "
                f"(beats={len(result.beats)}, likely no percussive content). "
                f"Using bpm=0.0 as sentinel - caller should fall back to AudioAnalyzer BPM."
            )

        segments = [
            Segment(start=seg.start, end=seg.end, label=seg.label)
            for seg in result.segments
        ]

        beats = np.array(result.beats)
        if len(beats) > 1 and bpm > 0.0:
            ibis = np.diff(beats)
            expected_ibi = 60.0 / bpm
            regularity = 1.0 - min(1.0, np.std(ibis - expected_ibi) / expected_ibi)
        else:
            regularity = 0.0

        duration = segments[-1].end if segments else 0.0

        logger.info(
            f"Structure analysis complete: BPM={bpm:.1f}, "
            f"segments={len(segments)}, regularity={regularity:.2%}"
        )

        return StructureAnalysisResult(
            bpm=bpm,
            beats=list(result.beats),
            downbeats=list(result.downbeats),
            segments=segments,
            duration=duration,
            beat_regularity=regularity,
            embeddings=result.embeddings if include_embeddings else None,
            activations=result.activations if include_activations else None,
        )

    def analyze_many(
        self,
        audio_paths: List[str],
        include_embeddings: bool = False,
        include_activations: bool = False,
    ) -> List[StructureAnalysisResult]:
        """Analyze multiple tracks in one allin1 invocation."""
        if not audio_paths:
            return []

        logger.info(f"Analyzing structure batch: size={len(audio_paths)}")
        allow_cpu_fallback = os.getenv('ALLOW_CPU_STRUCTURE_FALLBACK', '0') == '1'
        tmp_root = Path(os.getenv('ALLIN1_TMP_ROOT', '/app/storage/tmp/allin1'))
        tmp_root.mkdir(parents=True, exist_ok=True)
        primary_audio_path = audio_paths[0]

        # Emit a single lightweight telemetry snapshot before analysis
        emit_disk_telemetry(
            disk_logger,
            'structure_analysis.start',
            audio_path=primary_audio_path,
            device=self.device,
            audio_count=len(audio_paths),
            snapshot=snapshot_disk_state(
                storage_path=Path(os.getenv('STORAGE_PATH', '/app/storage')),
                root_path=Path('/'),
                tmp_root=tmp_root,
                audio_path=Path(primary_audio_path),
            ),
        )

        # Clean up stale temp dirs from previous runs
        for child in tmp_root.iterdir():
            if child.is_dir() and (child.name.startswith('demix-') or child.name.startswith('spec-')):
                shutil.rmtree(child, ignore_errors=True)

        devices_to_try = [self.device]
        if self.device == 'cuda' and allow_cpu_fallback:
            devices_to_try.append('cpu')

        last_error = None
        batch_results = None
        for device in devices_to_try:
            demix_dir = Path(tempfile.mkdtemp(prefix='demix-', dir=str(tmp_root)))
            spec_dir = Path(tempfile.mkdtemp(prefix='spec-', dir=str(tmp_root)))
            try:
                logger.info(
                    "Structure analysis byproducts: device=%s demix_dir=%s spec_dir=%s keep_byproducts=%s batch_size=%s",
                    device,
                    demix_dir,
                    spec_dir,
                    False,
                    len(audio_paths),
                )
                with _structure_gpu_lock(device=device, audio_count=len(audio_paths)):
                    batch_results = allin1.analyze(
                        paths=list(audio_paths),
                        model=self.model_name,
                        device=device,
                        include_embeddings=include_embeddings,
                        include_activations=include_activations,
                        demix_dir=demix_dir,
                        spec_dir=spec_dir,
                        keep_byproducts=False,
                    )

                if device != self.device:
                    logger.info(f"Structure batch analysis succeeded on fallback device={device}")

                break
            except Exception as e:
                last_error = e
                logger.error(
                    "Structure analysis failed: device=%s audio_count=%s error=%s",
                    device, len(audio_paths), e,
                )
                err_str = str(e).lower()
                is_oom = any(
                    key in err_str
                    for key in (
                        'sigkill',
                        'signal 9',
                        'killed',
                        'out of memory',
                        'cuda out of memory',
                        'oom',
                    )
                )

                if device == 'cuda' and is_oom:
                    if allow_cpu_fallback:
                        logger.warning(
                            f"Structure batch analysis OOM on GPU (device={device}), "
                            f"retrying on CPU: {e}"
                        )
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        continue

                    raise RuntimeError(
                        "Structure batch analysis hit GPU OOM and CPU fallback is disabled "
                        "to avoid persistent demix output growth."
                    ) from e

                raise
            finally:
                shutil.rmtree(demix_dir, ignore_errors=True)
                shutil.rmtree(spec_dir, ignore_errors=True)
                logger.info("Cleaned up temp dirs: demix=%s spec=%s", demix_dir, spec_dir)
        else:
            raise RuntimeError(
                f"Structure analysis failed on all devices ({devices_to_try}): {last_error}"
            ) from last_error

        if not isinstance(batch_results, list):
            batch_results = [batch_results]

        return [
            self._convert_analysis_result(path, result, include_embeddings, include_activations)
            for path, result in zip(audio_paths, batch_results)
        ]

    def analyze(
        self,
        audio_path: str,
        include_embeddings: bool = False,
        include_activations: bool = False
    ) -> StructureAnalysisResult:
        """
        Analyze music structure using All-In-One model.

        Args:
            audio_path: Path to audio file
            include_embeddings: Include neural network embeddings
            include_activations: Include frame-level activations

        Returns:
            StructureAnalysisResult with BPM, beats, segments, etc.

        Raises:
            RuntimeError: If analysis fails on all configured devices
        """
        return self.analyze_many(
            [audio_path],
            include_embeddings=include_embeddings,
            include_activations=include_activations,
        )[0]

    @staticmethod
    def _snap_event_to_grid(
        point: float,
        segment_end: float,
        beats: List[float],
        downbeats: List[float],
        max_forward: float,
    ) -> float:
        """Move event markers onto the first musically valid beat/downbeat inside the segment."""
        for grid_values in (downbeats, beats):
            if not grid_values:
                continue
            grid = np.array(sorted(grid_values), dtype=np.float32)
            forward = grid[
                (grid >= point - 0.15)
                & (grid <= min(segment_end, point + max_forward))
            ]
            if forward.size > 0:
                return float(forward[0])

            nearby = grid[np.abs(grid - point) <= 1.0]
            if nearby.size > 0:
                return float(nearby[int(np.argmin(np.abs(nearby - point)))])

        return float(point)

    @staticmethod
    def _dedupe_points(points: List[float], min_gap: float = 0.75) -> List[float]:
        deduped: List[float] = []
        for point in sorted(points):
            if not deduped or abs(point - deduped[-1]) >= min_gap:
                deduped.append(float(point))
        return deduped

    def get_drop_points(self, result: StructureAnalysisResult) -> List[float]:
        """Extract drop points from analysis (chorus starts)."""
        drops = []
        beat_window = 2.0
        if len(result.beats) > 1:
            beat_window = max(
                1.5,
                float(np.median(np.diff(np.array(result.beats, dtype=np.float32)))) * 4.0,
            )
        for i, seg in enumerate(result.segments):
            if seg.label == 'chorus':
                if i > 0 and result.segments[i - 1].label in ('verse', 'bridge', 'inst', 'prechorus'):
                    drops.append(
                        self._snap_event_to_grid(
                            seg.start,
                            seg.end,
                            result.beats,
                            result.downbeats,
                            beat_window,
                        )
                    )
        return self._dedupe_points(drops)

    def get_breakdown_points(self, result: StructureAnalysisResult) -> List[float]:
        """Extract breakdown points (bridge/break starts)."""
        breakdowns = []
        beat_window = 2.0
        if len(result.beats) > 1:
            beat_window = max(
                1.5,
                float(np.median(np.diff(np.array(result.beats, dtype=np.float32)))) * 4.0,
            )
        for seg in result.segments:
            if seg.label in ('bridge', 'break', 'inst'):
                breakdowns.append(
                    self._snap_event_to_grid(
                        seg.start,
                        seg.end,
                        result.beats,
                        result.downbeats,
                        beat_window,
                    )
                )
        return self._dedupe_points(breakdowns)

    def get_intro_outro(self, result: StructureAnalysisResult) -> Tuple[float, float]:
        """Get intro end and outro start points."""
        intro_end = 0.0
        outro_start = result.duration

        for seg in result.segments:
            if seg.label == 'intro':
                intro_end = seg.end
            elif seg.label == 'outro':
                outro_start = seg.start
                break

        return intro_end, outro_start
