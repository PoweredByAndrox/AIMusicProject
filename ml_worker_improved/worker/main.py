"""
ML Worker Main Entry Point

Starts the queue worker and registers task handlers.
Requirements: 13.1, 13.2, 13.3, 13.4
"""

import os
import sys
import json
import logging
import multiprocessing as mp
import queue as queue_module
import signal
import socket
import time
import threading
import traceback
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any

import numpy as np
import librosa

from .queue_worker import QueueWorker, TaskCancelledError
from .audio_analyzer import AudioAnalyzer
from .preference_model import PreferenceModel
from .dj_analyzer import AIDJAnalyzer
from .device_policy import resolve_torch_device, visible_cuda_summary
from .disk_telemetry import emit_disk_telemetry, snapshot_disk_state, start_disk_telemetry_sampler
from .structure_analyzer import StructureAnalyzer
from .segment_index import SegmentIndexManager
from .segment_melodic_refinement import refine_segments_with_melodic_continuity
from .segment_rating_remap import build_rating_safe_segment_remap, evaluate_rating_safe_remap_acceptance
from .segment_preference_model import SegmentPreferenceModel, create_segment_model
from .querysep_long_tail_build import build_querysep_long_tail_stems
from .stem_source_build import build_track_playback_stems
from .stem_tuning_render import render_stem_tuning

# Configure logging with file output
LOG_DIR = Path(os.getenv('STORAGE_PATH', '/app/storage')) / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_NAME = os.getenv('ML_WORKER_LOG_NAME', 'ml_worker.log')
ERROR_LOG_NAME = os.getenv('ML_WORKER_ERROR_LOG_NAME', 'ml_worker_errors.log')
DISK_LOG_NAME = os.getenv('ML_WORKER_DISK_LOG_NAME', 'ml_worker_disk.log')

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# File handler with rotation (10MB max, keep 5 backups)
file_handler = RotatingFileHandler(
    LOG_DIR / LOG_NAME,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Error-only file handler
error_handler = RotatingFileHandler(
    LOG_DIR / ERROR_LOG_NAME,
    maxBytes=5*1024*1024,  # 5MB
    backupCount=3,
    encoding='utf-8'
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d'))

disk_handler = RotatingFileHandler(
    LOG_DIR / DISK_LOG_NAME,
    maxBytes=10*1024*1024,
    backupCount=5,
    encoding='utf-8'
)
disk_handler.setLevel(logging.INFO)
disk_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler, error_handler]
)


def _load_existing_auto_examples_for_track(worker: QueueWorker, track_id: int) -> list[dict[str, Any]]:
    rows: list[tuple[Any, ...]] = []
    with worker.db_conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT pe.id, pe.start_time, pe.end_time, pe.rating, pe.is_positive
            FROM pattern_examples pe
            JOIN patterns p ON p.id = pe.pattern_id
            WHERE pe.track_id = %s
              AND p.name LIKE '%%[auto%%'
            ORDER BY pe.start_time, pe.id
            """,
            (track_id,),
        )
        rows = cursor.fetchall()

        example_ids = [int(row[0]) for row in rows]
        rating_rows: list[tuple[Any, ...]] = []
        if example_ids:
            cursor.execute(
                """
                SELECT user_id, pattern_example_id, rating, is_positive
                FROM user_pattern_example_ratings
                WHERE pattern_example_id = ANY(%s)
                ORDER BY user_id, pattern_example_id
                """,
                (example_ids,),
            )
            rating_rows = cursor.fetchall()
        ratings_by_example: dict[int, list[dict[str, Any]]] = {}
        for user_id, pattern_example_id, rating, is_positive in rating_rows:
            ratings_by_example.setdefault(int(pattern_example_id), []).append(
                {
                    'user_id': int(user_id),
                    'rating': float(rating) if rating is not None else None,
                    'is_positive': bool(is_positive) if is_positive is not None else None,
                }
            )

    examples: list[dict[str, Any]] = []
    for example_id, start_time, end_time, rating, is_positive in rows:
        examples.append(
            {
                'example_id': int(example_id),
                'start_time': float(start_time),
                'end_time': float(end_time),
                'rating': float(rating) if rating is not None else None,
                'is_positive': bool(is_positive) if is_positive is not None else None,
                'user_ratings': ratings_by_example.get(int(example_id), []),
            }
        )
    return examples


def _apply_segment_rating_remap(worker: QueueWorker, remap_payload: dict[str, Any]) -> None:
    user_updates = remap_payload.get('user_updates') or []
    legacy_updates = remap_payload.get('legacy_updates') or []
    if not user_updates and not legacy_updates:
        return

    with worker.db_conn.cursor() as cursor:
        for item in legacy_updates:
            cursor.execute(
                """
                UPDATE pattern_examples
                SET rating = %s,
                    is_positive = %s,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (
                    float(item['rating']),
                    bool(item['is_positive']),
                    int(item['pattern_example_id']),
                ),
            )

        for item in user_updates:
            cursor.execute(
                """
                INSERT INTO user_pattern_example_ratings (
                    user_id,
                    pattern_example_id,
                    rating,
                    is_positive,
                    created_at,
                    updated_at
                )
                VALUES (%s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (user_id, pattern_example_id)
                DO UPDATE SET
                    rating = EXCLUDED.rating,
                    is_positive = EXCLUDED.is_positive,
                    updated_at = NOW()
                """,
                (
                    int(item['user_id']),
                    int(item['pattern_example_id']),
                    float(item['rating']),
                    bool(item['is_positive']),
                ),
            )

logger = logging.getLogger(__name__)
logger.info(f"Logging to: {LOG_DIR / LOG_NAME}")

disk_logger = logging.getLogger('disk_telemetry')
disk_logger.setLevel(logging.INFO)
disk_logger.propagate = False
if not any(getattr(handler, 'baseFilename', None) == str((LOG_DIR / DISK_LOG_NAME).resolve()) for handler in disk_logger.handlers):
    disk_logger.addHandler(disk_handler)
emit_disk_telemetry(
    disk_logger,
    'disk_logger.ready',
    log_path=str(LOG_DIR / DISK_LOG_NAME),
    snapshot=snapshot_disk_state(
        storage_path=Path(os.getenv('STORAGE_PATH', '/app/storage')),
        root_path=Path('/'),
    ),
)
start_disk_telemetry_sampler(
    disk_logger,
    storage_path=Path(os.getenv('STORAGE_PATH', '/app/storage')),
    root_path=Path('/'),
    tmp_root=Path(os.getenv('ALLIN1_TMP_ROOT', '/app/storage/tmp/allin1')),
)

MIN_LIKES_FOR_TRAINING = 3
MAX_SEGMENT_ANALYSIS_DURATION = float(os.getenv('MAX_SEGMENT_ANALYSIS_DURATION', '900'))
SEGMENT_ANALYSIS_LOCK_KEY = (
    f"{os.getenv('REDIS_PREFIX', 'ai_music_recommendations_database_')}ml_worker:segment_analysis_lock"
)
SEGMENT_ANALYSIS_LOCK_HEARTBEAT_KEY = (
    f"{os.getenv('REDIS_PREFIX', 'ai_music_recommendations_database_')}ml_worker:segment_analysis_lock_heartbeat"
)
SEGMENT_ANALYSIS_LOCK_TTL_SECONDS = max(30, int(os.getenv('SEGMENT_ANALYSIS_LOCK_TTL_SECONDS', '300')))
SEGMENT_ANALYSIS_LOCK_HEARTBEAT_INTERVAL_SECONDS = max(
    1.0,
    float(
        os.getenv(
            'SEGMENT_ANALYSIS_LOCK_HEARTBEAT_INTERVAL_SECONDS',
            str(min(30, max(5, SEGMENT_ANALYSIS_LOCK_TTL_SECONDS // 3))),
        )
    ),
)
SEGMENT_ANALYSIS_LOCK_WAIT_SECONDS = max(1, int(os.getenv('SEGMENT_ANALYSIS_LOCK_WAIT_SECONDS', '7200')))
SEGMENT_ANALYSIS_LOCK_POLL_SECONDS = max(
    0.05,
    float(os.getenv('SEGMENT_ANALYSIS_LOCK_POLL_SECONDS', '0.25')),
)


def _create_segment_analysis_lock_redis_client():
    import redis

    return redis.Redis(
        host=os.getenv('REDIS_HOST', 'redis'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD') or None,
        decode_responses=True,
    )


def _release_segment_analysis_lock(redis_client, owner_token: str) -> None:
    release_script = """
    if redis.call('get', KEYS[1]) == ARGV[1] then
        redis.call('del', KEYS[1])
        if redis.call('get', KEYS[2]) == ARGV[1] then
            redis.call('del', KEYS[2])
        end
        return 1
    end
    return 0
    """
    redis_client.eval(
        release_script,
        2,
        SEGMENT_ANALYSIS_LOCK_KEY,
        SEGMENT_ANALYSIS_LOCK_HEARTBEAT_KEY,
        owner_token,
    )


def _refresh_segment_analysis_lock(redis_client, owner_token: str) -> bool:
    refresh_script = """
    if redis.call('get', KEYS[1]) == ARGV[1] then
        redis.call('expire', KEYS[1], tonumber(ARGV[2]))
        redis.call('set', KEYS[2], ARGV[1], 'EX', tonumber(ARGV[2]))
        return 1
    end
    return 0
    """
    refreshed = redis_client.eval(
        refresh_script,
        2,
        SEGMENT_ANALYSIS_LOCK_KEY,
        SEGMENT_ANALYSIS_LOCK_HEARTBEAT_KEY,
        owner_token,
        int(SEGMENT_ANALYSIS_LOCK_TTL_SECONDS),
    )
    return bool(refreshed)


def _cleanup_stale_segment_analysis_lock(redis_client, reason: str) -> bool:
    lock_owner = redis_client.get(SEGMENT_ANALYSIS_LOCK_KEY)
    heartbeat_owner = redis_client.get(SEGMENT_ANALYSIS_LOCK_HEARTBEAT_KEY)

    if lock_owner is None:
        if heartbeat_owner is not None:
            redis_client.delete(SEGMENT_ANALYSIS_LOCK_HEARTBEAT_KEY)
            logger.warning(
                "Removed orphaned segment analysis heartbeat: key=%s reason=%s owner=%s",
                SEGMENT_ANALYSIS_LOCK_HEARTBEAT_KEY,
                reason,
                heartbeat_owner,
            )
            return True
        return False

    if heartbeat_owner == lock_owner:
        return False

    cleanup_script = """
    local lock_owner = redis.call('get', KEYS[1])
    local heartbeat_owner = redis.call('get', KEYS[2])
    if not lock_owner then
        if heartbeat_owner then
            redis.call('del', KEYS[2])
            return 1
        end
        return 0
    end
    if heartbeat_owner == lock_owner then
        return 0
    end
    redis.call('del', KEYS[1])
    if heartbeat_owner then
        redis.call('del', KEYS[2])
    end
    return 1
    """
    removed = redis_client.eval(
        cleanup_script,
        2,
        SEGMENT_ANALYSIS_LOCK_KEY,
        SEGMENT_ANALYSIS_LOCK_HEARTBEAT_KEY,
    )
    if removed:
        logger.warning(
            "Removed stale segment analysis lock: key=%s heartbeat_key=%s reason=%s lock_owner=%s heartbeat_owner=%s",
            SEGMENT_ANALYSIS_LOCK_KEY,
            SEGMENT_ANALYSIS_LOCK_HEARTBEAT_KEY,
            reason,
            lock_owner,
            heartbeat_owner,
        )
    return bool(removed)


@contextmanager
def _segment_analysis_task_lock(track_id: int):
    redis_client = _create_segment_analysis_lock_redis_client()
    owner_token = f"{socket.gethostname()}:{os.getpid()}:{track_id}:{time.time_ns()}"
    deadline = time.monotonic() + SEGMENT_ANALYSIS_LOCK_WAIT_SECONDS
    wait_logged = False
    heartbeat_stop = threading.Event()
    heartbeat_thread = None

    while True:
        _cleanup_stale_segment_analysis_lock(redis_client, reason=f"wait-loop track_id={track_id}")
        acquired = redis_client.set(
            SEGMENT_ANALYSIS_LOCK_KEY,
            owner_token,
            nx=True,
            ex=SEGMENT_ANALYSIS_LOCK_TTL_SECONDS,
        )
        if acquired:
            logger.info(
                "Acquired segment analysis lock: key=%s track_id=%s ttl=%ss",
                SEGMENT_ANALYSIS_LOCK_KEY,
                track_id,
                SEGMENT_ANALYSIS_LOCK_TTL_SECONDS,
            )
            redis_client.set(
                SEGMENT_ANALYSIS_LOCK_HEARTBEAT_KEY,
                owner_token,
                ex=SEGMENT_ANALYSIS_LOCK_TTL_SECONDS,
            )
            break

        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Timed out waiting for segment analysis lock "
                f"(key={SEGMENT_ANALYSIS_LOCK_KEY}, track_id={track_id}, wait={SEGMENT_ANALYSIS_LOCK_WAIT_SECONDS}s)"
            )

        if not wait_logged:
            logger.info(
                "Waiting for segment analysis lock: key=%s track_id=%s",
                SEGMENT_ANALYSIS_LOCK_KEY,
                track_id,
            )
            wait_logged = True

        time.sleep(SEGMENT_ANALYSIS_LOCK_POLL_SECONDS)

    def heartbeat_loop():
        heartbeat_client = _create_segment_analysis_lock_redis_client()
        try:
            while not heartbeat_stop.wait(SEGMENT_ANALYSIS_LOCK_HEARTBEAT_INTERVAL_SECONDS):
                refreshed = _refresh_segment_analysis_lock(heartbeat_client, owner_token)
                if not refreshed:
                    logger.warning(
                        "Segment analysis lock heartbeat stopped: key=%s track_id=%s owner=%s",
                        SEGMENT_ANALYSIS_LOCK_KEY,
                        track_id,
                        owner_token,
                    )
                    return
        finally:
            try:
                heartbeat_client.close()
            except Exception:
                pass

    heartbeat_thread = threading.Thread(
        target=heartbeat_loop,
        name=f"segment-analysis-lock-heartbeat-{track_id}",
        daemon=True,
    )
    heartbeat_thread.start()

    try:
        yield
    finally:
        try:
            heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1.0)
            _release_segment_analysis_lock(redis_client, owner_token)
            logger.info(
                "Released segment analysis lock: key=%s track_id=%s",
                SEGMENT_ANALYSIS_LOCK_KEY,
                track_id,
            )
        finally:
            try:
                redis_client.close()
            except Exception:
                pass


def _enqueue_segment_analysis_if_needed(worker: QueueWorker, track_id: int, duration: float) -> bool:
    """Queue segment analysis right after track analysis if segments are still missing."""
    if duration < 30.0:
        logger.info(f"Track {track_id}: skipping automatic segment analysis (duration={duration:.1f}s)")
        return False
    if duration > MAX_SEGMENT_ANALYSIS_DURATION:
        logger.info(
            f"Track {track_id}: skipping automatic segment analysis "
            f"(duration={duration:.1f}s exceeds limit={MAX_SEGMENT_ANALYSIS_DURATION:.1f}s)"
        )
        return False

    if worker.redis_client is None:
        logger.warning(f"Track {track_id}: automatic segment analysis unavailable (Redis not connected)")
        return False

    try:
        with worker.db_conn.cursor() as cursor:
            cursor.execute("SELECT EXISTS(SELECT 1 FROM pattern_examples WHERE track_id = %s)", (track_id,))
            has_segments = bool(cursor.fetchone()[0])
    except Exception as e:
        logger.warning(f"Track {track_id}: failed to check existing segments: {e}")
        return False

    if has_segments:
        logger.info(f"Track {track_id}: skipping automatic segment analysis (segments already exist)")
        return False

    redis_prefix = os.getenv('REDIS_PREFIX', 'ai_music_recommendations_database_')
    status_key = f"{redis_prefix}segment_analysis_status:{track_id}"

    try:
        if worker.redis_client.exists(status_key):
            logger.info(f"Track {track_id}: segment analysis already queued or in progress")
            return False

        payload = {
            'type': 'analyze_segments',
            'task_id': f"auto_segment_analysis_{track_id}_{int(datetime.now().timestamp())}",
            'track_id': int(track_id),
            'created_at': datetime.now().isoformat(),
        }
        status_payload = {
            'status': 'queued',
            'task_id': payload['task_id'],
            'track_id': int(track_id),
            'progress': 0,
            'message': 'Waiting in queue...',
            'created_at': datetime.now().isoformat(),
        }
        worker.redis_client.setex(status_key, 300, json.dumps(status_payload))
        worker.redis_client.rpush(worker.QUEUE_NAME, json.dumps(payload))
        logger.info(f"Track {track_id}: queued automatic segment analysis ({payload['task_id']})")
        return True
    except Exception as e:
        logger.warning(f"Track {track_id}: failed to queue automatic segment analysis: {e}")
        return False


def trigger_auto_training_if_needed(worker: QueueWorker) -> bool:
    """Check if auto-training should be triggered and do it."""
    with worker.db_conn.cursor() as cursor:
        # Count liked tracks with embeddings
        cursor.execute("""
            SELECT COUNT(*) FROM tracks 
            WHERE is_liked = true AND embedding_path IS NOT NULL
        """)
        liked_count = cursor.fetchone()[0]
        
        if liked_count < MIN_LIKES_FOR_TRAINING:
            return False
        
        # Check if already training
        cursor.execute("SELECT status FROM preference_model_status LIMIT 1")
        row = cursor.fetchone()
        if row and row[0] == 'training':
            return False
        
        # Get liked track IDs
        cursor.execute("SELECT id FROM tracks WHERE is_liked = true AND embedding_path IS NOT NULL")
        liked_track_ids = [r[0] for r in cursor.fetchall()]
        
        cursor.execute("SELECT id FROM fragments WHERE is_liked = true AND embedding_path IS NOT NULL")
        liked_fragment_ids = [r[0] for r in cursor.fetchall()]
    
    logger.info(f"Auto-training triggered with {len(liked_track_ids)} liked tracks")
    
    # Train directly (we're already in the worker)
    task_data = {
        'liked_track_ids': liked_track_ids,
        'liked_fragment_ids': liked_fragment_ids,
    }
    
    try:
        handle_model_training(task_data, worker)
        return True
    except Exception as e:
        logger.error(f"Auto-training failed: {e}")
        return False


def _prepare_track_analysis_context(task_data: dict, worker: QueueWorker) -> dict:
    """Validate a track_analysis task and collect reusable context."""
    track_id = task_data['track_id']
    audio_path = task_data['audio_path']

    logger.info(f"Analyzing track {track_id}: {audio_path}")
    emit_disk_telemetry(
        disk_logger,
        'track_analysis.start',
        track_id=track_id,
        task_id=task_data.get('task_id'),
        snapshot=snapshot_disk_state(
            storage_path=Path(os.getenv('STORAGE_PATH', '/app/storage')),
            root_path=Path('/'),
            audio_path=Path(audio_path),
        ),
    )

    if not Path(audio_path).exists():
        logger.error(f"Audio file not found: {audio_path}")
        worker.update_track_status(track_id, 'failed')
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    duration = 0.0
    max_duration_seconds = 600.0
    try:
        duration = float(librosa.get_duration(path=audio_path))
        logger.info(f"Track {track_id} duration: {duration:.1f}s")
        if duration > max_duration_seconds:
            logger.warning(
                f"Track {track_id} skipped: duration {duration:.0f}s > {max_duration_seconds:.0f}s limit"
            )
            worker.update_track_status(track_id, 'failed')
            raise ValueError(
                f"Track too long ({duration:.0f}s > {max_duration_seconds:.0f}s limit), skipped"
            )
    except ValueError:
        raise
    except Exception as e:
        logger.warning(f"Could not determine duration for track {track_id}, proceeding: {e}")

    with worker.db_conn.cursor() as cursor:
        cursor.execute("SELECT id FROM tracks WHERE id = %s", (track_id,))
        if not cursor.fetchone():
            logger.error(f"Track {track_id} not found in database")
            raise ValueError(f"Track {track_id} not found in database")

    worker.update_track_status(track_id, 'processing')
    return {
        'task_data': task_data,
        'track_id': int(track_id),
        'audio_path': str(audio_path),
        'duration': float(duration),
    }


def _apply_structure_to_track_analysis(
    analysis: dict,
    structure,
    structure_analyzer: StructureAnalyzer,
) -> None:
    """Merge structure results into the main track analysis payload."""
    if structure is not None:
        analysis['bpm'] = structure.bpm
        analysis['intro_point'] = structure_analyzer.get_intro_outro(structure)[0]
        analysis['outro_point'] = structure_analyzer.get_intro_outro(structure)[1]
        analysis['drop_points'] = structure_analyzer.get_drop_points(structure)
        analysis['breakdown_points'] = structure_analyzer.get_breakdown_points(structure)
        analysis['beat_regularity'] = structure.beat_regularity
        return

    analysis.setdefault('intro_point', 0.0)
    analysis.setdefault('outro_point', 0.0)
    analysis.setdefault('drop_points', [])
    analysis.setdefault('breakdown_points', [])
    analysis.setdefault('beat_regularity', 0.0)


def _persist_track_analysis_result(
    context: dict,
    analysis: dict,
    worker: QueueWorker,
    structure_analyzer: StructureAnalyzer,
    structure=None,
) -> None:
    """Persist track analysis output and emit final telemetry."""
    track_id = int(context['track_id'])
    audio_path = str(context['audio_path'])
    duration = float(context.get('duration', 0.0))
    task_data = context['task_data']

    embedding_dir = Path(os.getenv('STORAGE_PATH', '/app/storage')) / 'embeddings'
    embedding_dir.mkdir(parents=True, exist_ok=True)

    embedding_path = embedding_dir / f"track_{track_id}.npy"
    np.save(str(embedding_path), analysis['embedding'])

    if structure is not None:
        structure_analyzer.save_cached_result(track_id, audio_path, structure)

    analysis['embedding_path'] = str(embedding_path)
    worker.update_track_analysis(track_id, analysis)
    _enqueue_segment_analysis_if_needed(worker, track_id, duration)
    emit_disk_telemetry(
        disk_logger,
        'track_analysis.finish',
        track_id=track_id,
        task_id=task_data.get('task_id'),
        genre=analysis.get('genre'),
        bpm=analysis.get('bpm'),
        embedding_path=str(embedding_path),
        snapshot=snapshot_disk_state(
            storage_path=Path(os.getenv('STORAGE_PATH', '/app/storage')),
            root_path=Path('/'),
            audio_path=Path(audio_path),
        ),
    )

    logger.info(f"Track {track_id} analysis complete: {analysis.get('genre')}, {analysis['bpm']} BPM")


def _load_or_analyze_structure(
    track_id: int,
    audio_path: str,
    structure_analyzer: StructureAnalyzer,
):
    """Reuse a cached structure result when available; otherwise run allin1."""
    cached_structure = structure_analyzer.load_cached_result(track_id, audio_path)
    if cached_structure is not None:
        logger.info(
            f"Track {track_id}: loaded cached structure result "
            f"(segments={len(cached_structure.segments)}, beats={len(cached_structure.beats)})"
        )
        return cached_structure

    structure = structure_analyzer.analyze(audio_path)
    logger.info(
        f"Track {track_id}: computed structure result "
        f"(segments={len(structure.segments)}, beats={len(structure.beats)})"
    )
    return structure


def _estimate_fallback_beat_grids(
    track_id: int,
    audio_signal: np.ndarray,
    sample_rate: int,
    hop_length: int,
    track_duration: float,
    bpm_hint: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Estimate beat/downbeat grids from audio when allin1 returns empty beat data."""
    try:
        onset_env = librosa.onset.onset_strength(
            y=audio_signal,
            sr=sample_rate,
            hop_length=hop_length,
        )
        start_bpm = float(bpm_hint) if bpm_hint and bpm_hint > 0 else 140.0
        estimated_tempo, beat_times = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=hop_length,
            start_bpm=start_bpm,
            units='time',
        )

        beat_grid = np.array(sorted(set(float(x) for x in np.atleast_1d(beat_times))), dtype=np.float32)
        beat_grid = beat_grid[(beat_grid >= 0.0) & (beat_grid <= track_duration)]
        if beat_grid.size == 0:
            logger.warning("Track %s: fallback beat estimation produced zero beats", track_id)
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), None

        downbeat_grid = beat_grid[::4].copy()
        tempo_value = float(np.atleast_1d(estimated_tempo).reshape(-1)[0])
        logger.info(
            "Track %s: fallback beat grid estimated via librosa (beats=%s, downbeats=%s, tempo=%.2f)",
            track_id,
            int(beat_grid.size),
            int(downbeat_grid.size),
            tempo_value,
        )
        return beat_grid, downbeat_grid, tempo_value
    except Exception as e:
        logger.warning("Track %s: fallback beat estimation failed: %s", track_id, e)
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), None


def handle_track_analysis_batch(
    task_batch: list[dict],
    worker: QueueWorker,
    analyzer: AudioAnalyzer,
    structure_analyzer: StructureAnalyzer,
) -> list[dict]:
    """Handle a micro-batch of track analysis tasks inside one worker process.

    Optimized pipeline:
    1. Prepare contexts (parallel DB queries via threads)
    2. Check structure cache (avoid Demucs for already-analyzed tracks)
    3. Run Demucs/allin1 structure analysis (GPU batch)
    4. Run Essentia audio analysis (parallel I/O + batched GPU + parallel CPU)
    5. Persist results
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time as _time

    batch_start = _time.monotonic()
    prepared_contexts: list[dict] = []
    outcomes: list[dict] = []

    # ── Phase 1: Prepare contexts ────────────────────────────────────
    for task_data in task_batch:
        try:
            prepared_contexts.append(_prepare_track_analysis_context(task_data, worker))
        except Exception as e:
            logger.error(f"Track {task_data.get('track_id')} preparation failed: {e}")
            outcomes.append(
                {
                    'task_id': task_data.get('task_id'),
                    'track_id': task_data.get('track_id'),
                    'status': 'failed',
                    'error': str(e),
                }
            )

    if not prepared_contexts:
        return outcomes

    track_ids = [context['track_id'] for context in prepared_contexts]
    audio_paths = [context['audio_path'] for context in prepared_contexts]
    logger.info(f"Analyzing track batch size={len(prepared_contexts)} track_ids={track_ids}")

    # ── Phase 2: Check structure cache (skip Demucs for cached tracks) ──
    structure_results = [None] * len(prepared_contexts)
    uncached_indices = []
    uncached_audio_paths = []

    for idx, context in enumerate(prepared_contexts):
        cached = structure_analyzer.load_cached_result(context['track_id'], context['audio_path'])
        if cached is not None:
            structure_results[idx] = cached
            logger.info(f"Track {context['track_id']}: using cached structure result")
        else:
            uncached_indices.append(idx)
            uncached_audio_paths.append(context['audio_path'])

    # ── Phase 3: Structure analysis (GPU batch — only uncached tracks) ──
    if uncached_audio_paths:
        try:
            structure_batch = structure_analyzer.analyze_many(uncached_audio_paths)
            if len(structure_batch) != len(uncached_indices):
                raise RuntimeError(
                    f"Structure batch size mismatch: expected {len(uncached_indices)}, got {len(structure_batch)}"
                )
            for batch_idx, original_idx in enumerate(uncached_indices):
                structure_results[original_idx] = structure_batch[batch_idx]
            logger.info(
                "Track batch structure analysis complete: analyzed=%s cached=%s track_ids=%s",
                len(uncached_indices),
                len(prepared_contexts) - len(uncached_indices),
                track_ids,
            )
        except Exception as e:
            logger.warning(f"Track batch structure analysis failed; retrying per track: {e}")
            for original_idx in uncached_indices:
                context = prepared_contexts[original_idx]
                try:
                    structure_results[original_idx] = structure_analyzer.analyze(context['audio_path'])
                except Exception as single_error:
                    logger.warning(
                        f"Structure analysis failed for track {context['track_id']} (batch fallback): {single_error}"
                    )
                    structure_results[original_idx] = None
    else:
        logger.info("All tracks had cached structure results, skipping Demucs entirely")

    # ── Phase 4: Audio analysis (parallel I/O + batched GPU + parallel CPU) ──
    analysis_results = [None] * len(prepared_contexts)
    try:
        analysis_batch = analyzer.analyze_full_batch(audio_paths)
        if len(analysis_batch) != len(prepared_contexts):
            raise RuntimeError(
                f"Audio batch size mismatch: expected {len(prepared_contexts)}, got {len(analysis_batch)}"
            )
        analysis_results = list(analysis_batch)
        logger.info(
            "Track batch audio analysis complete: size=%s track_ids=%s",
            len(prepared_contexts),
            track_ids,
        )
    except Exception as e:
        logger.warning(f"Track batch audio analysis failed; retrying per track: {e}")
        for idx, context in enumerate(prepared_contexts):
            try:
                analysis_results[idx] = analyzer.analyze_full(context['audio_path'])
            except Exception as single_error:
                logger.warning(
                    f"Audio analysis failed for track {context['track_id']} (batch fallback): {single_error}"
                )
                analysis_results[idx] = None

    # ── Phase 5: Persist results ─────────────────────────────────────
    for context, structure, analysis in zip(prepared_contexts, structure_results, analysis_results):
        track_id = context['track_id']
        task_data = context['task_data']
        try:
            if analysis is None:
                raise RuntimeError("Audio analysis did not produce a result")
            _apply_structure_to_track_analysis(analysis, structure, structure_analyzer)
            _persist_track_analysis_result(context, analysis, worker, structure_analyzer, structure)
            outcomes.append(
                {
                    'task_id': task_data.get('task_id'),
                    'track_id': track_id,
                    'status': 'completed',
                    'analysis': analysis,
                }
            )
        except Exception as e:
            logger.exception(f"Track {track_id} analysis failed in batch: {e}")
            outcomes.append(
                {
                    'task_id': task_data.get('task_id'),
                    'track_id': track_id,
                    'status': 'failed',
                    'error': str(e),
                }
            )

    elapsed = _time.monotonic() - batch_start
    completed_count = sum(1 for o in outcomes if o.get('status') == 'completed')
    per_track = elapsed / max(1, len(prepared_contexts))
    logger.info(
        "Batch pipeline complete: total=%.1fs per_track=%.1fs completed=%d/%d track_ids=%s",
        elapsed, per_track, completed_count, len(prepared_contexts), track_ids,
    )

    return outcomes


def _handle_track_analysis_legacy(task_data: dict, worker: QueueWorker, analyzer: AudioAnalyzer, structure_analyzer: StructureAnalyzer) -> dict:
    """Handle track analysis task with All-In-One BPM/structure detection."""
    track_id = task_data['track_id']
    audio_path = task_data['audio_path']
    
    logger.info(f"Analyzing track {track_id}: {audio_path}")
    emit_disk_telemetry(
        disk_logger,
        'track_analysis.start',
        track_id=track_id,
        task_id=task_data.get('task_id'),
        snapshot=snapshot_disk_state(
            storage_path=Path(os.getenv('STORAGE_PATH', '/app/storage')),
            root_path=Path('/'),
            audio_path=Path(audio_path),
        ),
    )
    
    # Check if audio file exists - fail fast if not
    if not Path(audio_path).exists():
        logger.error(f"Audio file not found: {audio_path}")
        worker.update_track_status(track_id, 'failed')
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Skip tracks longer than 10 minutes (600s) — too heavy for Demucs
    MAX_DURATION_SECONDS = 600
    try:
        duration = librosa.get_duration(path=audio_path)
        logger.info(f"Track {track_id} duration: {duration:.1f}s")
        if duration > MAX_DURATION_SECONDS:
            logger.warning(f"Track {track_id} skipped: duration {duration:.0f}s > {MAX_DURATION_SECONDS}s limit")
            worker.update_track_status(track_id, 'failed')
            raise ValueError(f"Track too long ({duration:.0f}s > {MAX_DURATION_SECONDS}s limit), skipped")
    except ValueError:
        raise  # re-raise our own "too long" ValueError
    except Exception as e:
        logger.warning(f"Could not determine duration for track {track_id}, proceeding: {e}")

    # Check if track exists in database
    with worker.db_conn.cursor() as cursor:
        cursor.execute("SELECT id FROM tracks WHERE id = %s", (track_id,))
        if not cursor.fetchone():
            logger.error(f"Track {track_id} not found in database")
            raise ValueError(f"Track {track_id} not found in database")
    
    # Update track status to processing (track was in 'pending' until now)
    worker.update_track_status(track_id, 'processing')
    
    # Use All-In-One for BPM and structure (more accurate)
    structure = None
    try:
        structure = structure_analyzer.analyze(audio_path)
        logger.info(f"All-In-One: BPM={structure.bpm:.1f}, segments={len(structure.segments)}, regularity={structure.beat_regularity:.2%}")
    except Exception as e:
        logger.warning(f"Structure analysis failed (will use fallback BPM/structure): {e}")

    # Perform full analysis (embedding, genre, mood, etc.)
    analysis = analyzer.analyze_full(audio_path)

    if structure is not None:
        # Override BPM with All-In-One result (more accurate)
        analysis['bpm'] = structure.bpm
        analysis['intro_point'] = structure_analyzer.get_intro_outro(structure)[0]
        analysis['outro_point'] = structure_analyzer.get_intro_outro(structure)[1]
        analysis['drop_points'] = structure_analyzer.get_drop_points(structure)
        analysis['breakdown_points'] = structure_analyzer.get_breakdown_points(structure)
        analysis['beat_regularity'] = structure.beat_regularity
    else:
        # Fallback: keep BPM from AudioAnalyzer, set structure fields to safe defaults
        analysis.setdefault('intro_point', 0.0)
        analysis.setdefault('outro_point', 0.0)
        analysis.setdefault('drop_points', [])
        analysis.setdefault('breakdown_points', [])
        analysis.setdefault('beat_regularity', 0.0)
    
    # Save embedding
    embedding_dir = Path(os.getenv('STORAGE_PATH', '/app/storage')) / 'embeddings'
    embedding_dir.mkdir(parents=True, exist_ok=True)
    
    embedding_path = embedding_dir / f"track_{track_id}.npy"
    np.save(str(embedding_path), analysis['embedding'])
    
    if structure is not None:
        structure_analyzer.save_cached_result(int(track_id), str(audio_path), structure)

    # Update track in database
    analysis['embedding_path'] = str(embedding_path)
    worker.update_track_analysis(track_id, analysis)
    _enqueue_segment_analysis_if_needed(worker, int(track_id), float(duration))
    emit_disk_telemetry(
        disk_logger,
        'track_analysis.finish',
        track_id=track_id,
        task_id=task_data.get('task_id'),
        genre=analysis.get('genre'),
        bpm=analysis.get('bpm'),
        embedding_path=str(embedding_path),
        snapshot=snapshot_disk_state(
            storage_path=Path(os.getenv('STORAGE_PATH', '/app/storage')),
            root_path=Path('/'),
            audio_path=Path(audio_path),
        ),
    )

    logger.info(f"Track {track_id} analysis complete: {analysis.get('genre')}, {analysis['bpm']} BPM")
    
    return analysis


def handle_track_analysis(task_data: dict, worker: QueueWorker, analyzer: AudioAnalyzer, structure_analyzer: StructureAnalyzer) -> dict:
    """Handle track analysis task with All-In-One BPM/structure detection."""
    outcomes = handle_track_analysis_batch([task_data], worker, analyzer, structure_analyzer)
    if not outcomes:
        raise RuntimeError("Track analysis batch handler returned no outcomes")

    outcome = outcomes[0]
    if outcome.get('status') != 'completed':
        raise RuntimeError(str(outcome.get('error', 'Track analysis failed')))

    return outcome['analysis']


def handle_fragment_analysis(task_data: dict, worker: QueueWorker, analyzer: AudioAnalyzer, segment_index: SegmentIndexManager) -> dict:
    """Handle fragment analysis task - extracts embedding from specific segment."""
    fragment_id = task_data['fragment_id']
    track_id = task_data['track_id']
    audio_path = task_data['audio_path']
    start_time = task_data['start_time']
    end_time = task_data['end_time']
    
    logger.info(f"Analyzing fragment {fragment_id} ({start_time}-{end_time}s)")
    
    # Check if audio file exists - fail fast if not
    if not Path(audio_path).exists():
        logger.error(f"Audio file not found for fragment {fragment_id}: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Extract embedding for the SPECIFIC SEGMENT (not whole track)
    embedding = analyzer.extract_embeddings(audio_path, start_time=start_time, end_time=end_time)
    
    # Save embedding
    embedding_dir = Path(os.getenv('STORAGE_PATH', '/app/storage')) / 'embeddings'
    embedding_dir.mkdir(parents=True, exist_ok=True)
    
    embedding_path = embedding_dir / f"fragment_{fragment_id}.npy"
    np.save(str(embedding_path), embedding)
    
    # Update fragment in database
    with worker.db_conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE fragments 
            SET embedding_path = %s, analysis_status = 'completed'
            WHERE id = %s
            """,
            (str(embedding_path), fragment_id)
        )
    
    # Add to segment index for similarity search (Requirement 2.2)
    segment_index.add_segment(fragment_id, track_id, embedding, start_time, end_time)
    segment_index.save()
    
    logger.info(f"Fragment {fragment_id} analysis complete (segment {start_time:.1f}s - {end_time:.1f}s)")
    return {'embedding_path': str(embedding_path)}


def analyze_comment_sentiment(comment: str) -> dict:
    """
    Simple sentiment analysis for fragment comments.
    Returns sentiment features that can boost/reduce preference weight.
    """
    if not comment:
        return {'weight': 1.0, 'sentiment': 'neutral'}
    
    comment_lower = comment.lower()
    
    # Positive keywords (boost weight)
    positive_words = [
        # EN general positive
        'love', 'loved', 'lovely', 'amazing', 'awesome', 'great', 'best', 'favorite',
        'perfect', 'beautiful', 'incredible', 'fantastic', 'excellent', 'insane',
        'legendary', 'masterpiece', 'goated', 'goat', 'banger', 'slaps', 'slap',
        'hits', 'hits hard', 'fire', 'lit', 'dope', 'sick', 'crazy', 'wild',
        'fantabulous', 'stellar', 'brilliant', 'top tier', '10/10', '11/10',
        'on repeat', 'repeat', 'replay', 'relistening',

        # RU general positive
        'люблю', 'любимка', 'обожаю', 'обожаюсь', 'кайф', 'кайфую', 'кайфово',
        'круто', 'крутяк', 'крутейше', 'крутейший', 'класс', 'классно', 'супер',
        'шикарно', 'великолепно', 'прекрасно', 'красиво', 'бомба', 'имба', 'имбово',
        'топ', 'топчик', 'топово', 'лучший', 'лучшая', 'лучшее', 'пушка', 'пушечно',
        'огонь', 'огненно', 'разъёб', 'разнос', 'разносит', 'разрыв', 'разрывной',
        'сочно', 'жирно', 'мощно', 'сильно', 'сильнейше', 'чётко', 'чётенько',
        'потрясающе', 'офигенно', 'офигеть', 'офигенно!', 'охрененно', 'охренеть',
        'зашло', 'зашёл', 'зашла', 'вкатило', 'вкатывает', 'впёрло', 'прёт', 'прет',
        'тащит', 'тащусь', 'цепляет', 'залип', 'залипательно', 'залипаю',
        'шедевр', 'годно', 'годнота', 'кайфец', 'приятно', 'атмосферно', 'лампово',
        'ностальгия', 'ностальгично', 'вдохновляет', 'мурашки', 'мурашит', 'до мурашек',

        # RU profanity-positive (мат/жаргон)
        'ахуенно', 'охуенно', 'пиздато', 'заебись', 'заебато', 'ебать круто',
        'пиздец как круто', 'разъебало', 'разъёбало', 'охуеть как', 'ахуеть как',
        'ебануться', 'ебнуться', 'ебать', 'сука кайф', 'сука круто',

        # Music-specific EN
        'drop', 'bass', 'subbass', 'melody', 'hook', 'chorus', 'verse', 'bridge',
        'vocal', 'vocals', 'flow', 'groove', 'rhythm', 'beat', 'buildup', 'build-up',
        'breakdown', 'synth', 'lead', 'kick', 'snare', 'hi-hat', '808', 'mix', 'master',
        'mastering', 'production', 'arrangement', 'sound design', 'sounddesign',
        'clean mix', 'crispy', 'tight', 'punchy', 'wide', 'warm', 'lush', 'atmospheric',

        # Music-specific RU
        'бит', 'биток', 'кач', 'качает', 'качает жёстко', 'качает жестко', 'раскачало',
        'вибра', 'вибрация', 'виб', 'вайб', 'вайбово', 'атмосфера', 'атмосферка',
        'припев', 'куплет', 'хуки', 'хук', 'мелодия', 'гармония', 'вокал', 'голос',
        'флоу', 'читка', 'панч', 'панчи', 'рифмы', 'рифмует', 'саунд', 'саунд крутой',
        'сведение', 'мастеринг', 'продакшн', 'аранжировка', 'саунддизайн',
        'дроп', 'дропнуло', 'дропчик', 'бас', 'басуха', 'низ', 'низы', 'саб', 'саббас',
        'плотно', 'плотняк', 'чисто', 'чистый звук', 'широко', 'объёмно', 'объемно',
        'сочно звучит', 'вкусно', 'вкусный звук',

        # Actions that imply positive preference
        'в плейлист', 'в избранное', 'в любимое', 'добавил', 'добавила', 'сохранил',
        'сохранила', 'сохраню', 'сохранить', 'на репит', 'на повтор', 'на повторе',
        'переслушиваю', 'переслушал', 'переслушала', 'слушаю снова', 'слушаю ещё раз',
        'буду слушать', 'буду переслушивать',

        # Emojis / symbols (positive)
        '❤️', '❤', '💙', '💚', '💜', '🖤', '💖', '💗', '💘', '💝', '😍', '🥰', '🤩',
        '🔥', '💯', '✅', '⭐', '🌟', '✨', '👏', '🙌', '😻', '🎧', '🎶', '🔊'
    ]

    # Negative keywords (reduce weight)
    negative_words = [
        # EN general negative
        'bad', 'wack', 'weak', 'boring', 'skip', 'skipping', 'hate', 'hated',
        'worst', 'annoying', 'meh', 'mid', 'trash', 'garbage', 'awful', 'terrible',
        'cringe', 'corny', 'lame', 'overrated', 'forgettable', 'generic',
        'disappointing', 'painful', 'noise', 'noisy',

        # RU general negative
        'плохо', 'ужасно', 'отстой', 'отстойно', 'дичь', 'треш', 'трэш', 'мусор',
        'скучно', 'нудно', 'занудно', 'уныло', 'пресно', 'никак', 'не цепляет',
        'не зашло', 'не зашел', 'не зашла', 'не вкатило', 'не прёт', 'не прет',
        'надоело', 'надоедает', 'заебало', 'приелось', 'бесит', 'раздражает',
        'кринж', 'кринжово', 'стыд', 'стыдно', 'испанский стыд',
        'слабо', 'слабенько', 'посредственно', 'посредняк', 'так себе', 'такое себе',
        'не интересно', 'неинтересно', 'не нравится', 'не понравилось',
        'пропустить', 'скип', 'скипнуть', 'пролистал', 'пролистала',
        'глупо', 'тупо', 'тупняк', 'бред', 'ерунда', 'фигня', 'херня', 'лажа',
        'туфта', 'шляпа', 'шлак', 'параша', 'дно', 'днище', 'ни о чём', 'ни о чем',

        # RU profanity-negative (мат/жаргон)
        'хуйня', 'хуита', 'хуитень', 'ебанина', 'ебень', 'говно', 'говнище',
        'дерьмо', 'пиздец', 'пиздец плохо', 'пиздец ужас', 'сука бесит',
        'ебать как плохо', 'нахуй', 'иди нахуй', 'заебало', 'заебала', 'заебали',

        # Music-specific EN negative
        'offbeat', 'out of tune', 'clipping', 'distorted', 'muddy', 'flat',
        'thin', 'harsh', 'piercing', 'bad mix', 'bad mastering', 'overcompressed',
        'too loud', 'too quiet', 'no dynamics',

        # Music-specific RU negative
        'не качает', 'не качнуло', 'не раскачало', 'пусто', 'пустоват', 'пустовато',
        'грязно', 'грязный звук', 'каша', 'засрано', 'перегруз', 'перегружено',
        'плоско', 'тонко', 'режет уши', 'сверлит', 'шум', 'шумно',
        'бит мимо', 'вокал мимо', 'фальш', 'фальшиво', 'не в тон', 'не строит',
        'плохое сведение', 'плохой микс', 'плохой мастер', 'перекомпресс',
        'однообразно', 'монотонно', 'затянуто', 'затянули', 'слишком долго',
        'нет дропа', 'дроп слабый', 'без дропа', 'нет хука', 'хука нет',

        # Actions that imply negative preference
        'удалил', 'удалила', 'не сохраню', 'не добавлю', 'не в плейлист',
        'пропускаю', 'пропустил', 'пропустила', 'скипаю', 'скипнул', 'скипнула',

        # Emojis / symbols (negative)
        '💩', '🤮', '🤢', '🙄', '😒', '😴', '🤦', '🤦‍♂️', '🤦‍♀️', '❌', '⛔', '🚫'
    ]
    
    positive_count = sum(1 for word in positive_words if word in comment_lower)
    negative_count = sum(1 for word in negative_words if word in comment_lower)
    
    # Calculate weight multiplier (0.5 to 2.0)
    if positive_count > negative_count:
        weight = min(2.0, 1.0 + positive_count * 0.2)
        sentiment = 'positive'
    elif negative_count > positive_count:
        weight = max(0.5, 1.0 - negative_count * 0.2)
        sentiment = 'negative'
    else:
        weight = 1.0
        sentiment = 'neutral'
    
    return {'weight': weight, 'sentiment': sentiment}


def handle_model_training(task_data: dict, worker: QueueWorker) -> dict:
    """Handle preference model training task with fragment, segment ratings and comment support.
    
    IMPORTANT: On failure, resets preference_model_status back to previous state
    so training can be retried. Without this, a failed training leaves the status
    stuck at 'training' forever.
    """
    user_id = task_data.get('user_id')

    # Remember previous status so we can restore on failure
    _prev_status = 'not_trained'
    try:
        with worker.db_conn.cursor() as cursor:
            if user_id is None:
                cursor.execute("SELECT status FROM preference_model_status WHERE user_id IS NULL LIMIT 1")
            else:
                cursor.execute("SELECT status FROM preference_model_status WHERE user_id = %s LIMIT 1", (int(user_id),))
            row = cursor.fetchone()
            if row:
                _prev_status = row[0]
    except Exception:
        pass  # If we can't read, default to not_trained is safe

    try:
        return _handle_model_training_inner_v2(task_data, worker)
    except Exception as e:
        # Reset preference_model_status so training can be retried
        logger.error(f"Model training failed, resetting status to '{_prev_status}': {e}")
        try:
            with worker.db_conn.cursor() as cursor:
                reset_status = _prev_status if _prev_status != 'training' else 'not_trained'
                if user_id is None:
                    cursor.execute(
                        "UPDATE preference_model_status SET status = %s WHERE user_id IS NULL",
                        (reset_status,)
                    )
                else:
                    cursor.execute(
                        "UPDATE preference_model_status SET status = %s WHERE user_id = %s",
                        (reset_status, int(user_id))
                    )
        except Exception as db_err:
            logger.error(f"Failed to reset preference_model_status: {db_err}")
        raise  # Re-raise so process_task marks queue_tasks as failed


def _handle_model_training_inner(task_data: dict, worker: QueueWorker) -> dict:
    """Inner implementation of model training (extracted for error recovery wrapper).
    
    NOTE: Redis message only contains {type, task_id}. The full payload
    (liked_track_ids, liked_fragments, etc.) is stored in queue_tasks table.
    We read it from DB by task_id.
    """
    liked_track_ids = task_data.get('liked_track_ids', [])
    liked_fragment_ids = task_data.get('liked_fragment_ids', [])
    
    # FIX Issue A: Redis message only has task_id — read full payload from queue_tasks
    if not liked_track_ids:
        queue_task_id = task_data.get('task_id')
        if queue_task_id:
            try:
                with worker.db_conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT payload FROM queue_tasks WHERE id = %s",
                        (queue_task_id,)
                    )
                    row = cursor.fetchone()
                    if row and row[0]:
                        payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                        liked_track_ids = payload.get('liked_track_ids', [])
                        liked_fragment_ids = payload.get('liked_fragment_ids', [])
                        logger.info(f"Loaded payload from queue_tasks id={queue_task_id}: "
                                    f"{len(liked_track_ids)} liked tracks, {len(liked_fragment_ids)} liked fragments")
                    else:
                        logger.warning(f"No payload found in queue_tasks id={queue_task_id}")
            except Exception as e:
                logger.error(f"Failed to read payload from queue_tasks: {e}")
    
    logger.info(f"Training model with {len(liked_track_ids)} liked tracks, {len(liked_fragment_ids)} liked fragments")
    
    # Load embeddings for liked items
    storage_path = Path(os.getenv('STORAGE_PATH', '/app/storage'))
    embeddings_dir = storage_path / 'embeddings'
    
    liked_embeddings = []
    liked_weights = []  # Weight based on rating and comment sentiment
    # Track which paths are already loaded to avoid duplicates in all_embeddings
    all_emb_paths_seen: set = set()
    all_embeddings = []

    def _load_unique(path: str) -> "np.ndarray | None":
        """Load .npy only if not already in all_embeddings (dedup by path)."""
        p = str(Path(path).resolve())
        emb = np.load(path)
        if p not in all_emb_paths_seen:
            all_emb_paths_seen.add(p)
            all_embeddings.append(emb)
        return emb

    # Load track embeddings
    for track_id in liked_track_ids:
        embedding_path = embeddings_dir / f"track_{track_id}.npy"
        if embedding_path.exists():
            emb = _load_unique(str(embedding_path))
            liked_embeddings.append(emb)
            liked_weights.append(1.0)  # Default weight for liked tracks
    
    # Load fragment embeddings with comment analysis
    with worker.db_conn.cursor() as cursor:
        cursor.execute("""
            SELECT f.id, f.embedding_path, f.comment, f.track_id
            FROM fragments f
            WHERE f.is_liked = true AND f.embedding_path IS NOT NULL
        """)
        fragment_rows = cursor.fetchall()
        
        for row in fragment_rows:
            frag_id, emb_path, comment, track_id = row
            if emb_path and Path(emb_path).exists():
                emb = _load_unique(emb_path)
                liked_embeddings.append(emb)
                
                # Base weight for liked fragments
                base_weight = 1.0
                
                # Analyze comment sentiment
                comment_analysis = analyze_comment_sentiment(comment)
                final_weight = base_weight * comment_analysis['weight']
                
                liked_weights.append(final_weight)
                
                if comment:
                    logger.info(f"Fragment {frag_id}: comment sentiment={comment_analysis['sentiment']}, weight={final_weight:.2f}")
    
    # Load segment ratings from pattern_examples (new rating system)
    with worker.db_conn.cursor() as cursor:
        cursor.execute("""
            SELECT pe.id, pe.track_id, pe.rating, t.embedding_path
            FROM pattern_examples pe
            JOIN tracks t ON pe.track_id = t.id
            WHERE pe.rating IS NOT NULL AND pe.rating >= 3 AND t.embedding_path IS NOT NULL
        """)
        segment_rows = cursor.fetchall()
        
        for row in segment_rows:
            seg_id, track_id, rating, emb_path = row
            if emb_path and Path(emb_path).exists():
                emb = _load_unique(emb_path)
                liked_embeddings.append(emb)
                
                # Weight based on rating: 5 stars = 2.0, 3 stars = 0.5
                weight = (rating / 5.0) * 2.0  # 5 stars = 2.0, 3 stars = 1.2
                liked_weights.append(weight)
                
                logger.info(f"Segment {seg_id}: rating={rating}, weight={weight:.2f}")
    
    # Load all remaining track embeddings for negative sampling (skip already-loaded ones)
    with worker.db_conn.cursor() as cursor:
        cursor.execute("SELECT id, embedding_path FROM tracks WHERE embedding_path IS NOT NULL")
        rows = cursor.fetchall()
        for row in rows:
            track_id, emb_path = row
            if emb_path and Path(emb_path).exists():
                _load_unique(emb_path)  # dedup handled inside
    
    if len(liked_embeddings) == 0:
        raise ValueError("No embeddings found for liked items")
    
    liked_embeddings = np.array(liked_embeddings)
    liked_weights = np.array(liked_weights)
    all_embeddings = np.array(all_embeddings)
    
    logger.info(f"Loaded {len(liked_embeddings)} liked embeddings (with weights), {len(all_embeddings)} total embeddings")
    
    # Train model with weighted samples
    model = PreferenceModel(embedding_dim=liked_embeddings.shape[1], learning_rate=0.003)
    history = model.train(liked_embeddings, all_embeddings, epochs=300, sample_weights=liked_weights)
    
    # Save model
    models_dir = Path(os.getenv('MODELS_PATH', '/app/models'))
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / 'preference_model.npz'
    model.save(str(model_path))
    
    # Update preference_model_status in database
    with worker.db_conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE preference_model_status 
            SET status = 'trained', model_path = %s, trained_at = %s, likes_count = %s
            WHERE id = (SELECT id FROM preference_model_status LIMIT 1)
            """,
            (str(model_path), datetime.now(), len(liked_track_ids) + len(liked_fragment_ids) + len(segment_rows))
        )
    
    # FIX Issue B: Score ALL tracks with the trained model and write preference_score to DB
    # Two-phase approach: first collect all raw scores, then min-max normalize to 0.05..0.95
    # Sigmoid was compressing everything to 0.48-0.55 because raw scores are small (~0.3 range)
    logger.info("Calculating preference_score for all tracks with embeddings...")
    scored_count = 0
    try:
        with worker.db_conn.cursor() as cursor:
            cursor.execute("SELECT id, embedding_path FROM tracks WHERE embedding_path IS NOT NULL")
            all_track_rows = cursor.fetchall()
        
        # Phase 1: Collect all raw scores
        score_pairs = []  # [(track_id, raw_score), ...]
        batch_ids = []
        batch_embeddings = []
        BATCH_SIZE = 500
        
        for track_id, emb_path in all_track_rows:
            if emb_path and Path(emb_path).exists():
                try:
                    emb = np.load(emb_path)
                    batch_ids.append(track_id)
                    batch_embeddings.append(emb)
                except Exception:
                    continue
            
            if len(batch_ids) >= BATCH_SIZE:
                emb_matrix = np.array(batch_embeddings)
                scores = model.predict(emb_matrix)
                for tid, sc in zip(batch_ids, scores):
                    score_pairs.append((tid, float(sc)))
                batch_ids = []
                batch_embeddings = []
        
        # Flush remaining
        if batch_ids:
            emb_matrix = np.array(batch_embeddings)
            scores = model.predict(emb_matrix)
            for tid, sc in zip(batch_ids, scores):
                score_pairs.append((tid, float(sc)))
        
        if not score_pairs:
            logger.warning("No tracks to score")
        else:
            # Phase 2: Rank-based normalization with temperature-scaled sigmoid
            # Raw scores from BPR model are often clustered. Rank-based approach
            # converts to percentile ranks first, giving uniform distribution,
            # then applies sigmoid for smooth S-curve centered at median.
            raw_scores = np.array([s for _, s in score_pairs])
            track_ids_arr = np.array([tid for tid, _ in score_pairs])
            n = len(raw_scores)
            
            logger.info(f"Raw scores: min={raw_scores.min():.6f}, max={raw_scores.max():.6f}, "
                        f"mean={raw_scores.mean():.6f}, std={raw_scores.std():.6f}")
            
            # Count ties to log clustering severity
            unique_raw, raw_counts = np.unique(raw_scores, return_counts=True)
            max_tie_count = raw_counts.max()
            logger.info(f"Unique raw scores: {len(unique_raw)}, largest tie cluster: {max_tie_count} tracks")
            
            if raw_scores.std() < 1e-10:
                logger.warning("All raw scores identical, assigning 0.5")
                normalized = np.full(n, 0.5)
            else:
                # Tiebreaking: add deterministic micro-jitter based on track_id
                # This breaks ties within same raw score cluster so ranks spread out.
                # Jitter magnitude is 1/100th of the smallest gap between distinct scores,
                # so it never changes the ordering of tracks with different raw scores.
                if len(unique_raw) > 1:
                    sorted_unique = np.sort(unique_raw)
                    min_gap = np.min(np.diff(sorted_unique))
                    jitter_scale = min_gap / 100.0
                else:
                    jitter_scale = 1e-8
                
                # Deterministic hash per track_id → stable jitter in [0, jitter_scale)
                jitter = np.array([
                    (hash(int(tid)) % 10000) / 10000.0 * jitter_scale
                    for tid in track_ids_arr
                ])
                jittered_scores = raw_scores + jitter
                
                # Step 1: Convert to percentile ranks (0..1) using jittered scores
                order = np.argsort(jittered_scores)
                ranks = np.empty(n, dtype=float)
                ranks[order] = np.arange(n, dtype=float)
                # Normalize ranks to (0, 1) — avoid exact 0 and 1
                percentiles = (ranks + 0.5) / n  # range: [0.5/n, (n-0.5)/n]
                
                # Step 2: Map percentiles through inverse-sigmoid then temperature-sigmoid
                # This gives smooth S-curve: low percentiles → low scores, high → high
                # percentile 0.5 → score 0.5 (median track)
                # Temperature T controls steepness: lower T = more contrast
                T = 1.0  # T=1.0 gives percentile-based scoring with smooth S-curve
                # p5≈0.05, p25≈0.25, p50=0.50, p75≈0.75, p95≈0.95
                
                # Convert percentile to logit space, scale by T, then sigmoid back
                # logit(p) = ln(p/(1-p)), then sigmoid(logit(p)/T) reshapes the curve
                eps = 1e-7
                percentiles_clipped = np.clip(percentiles, eps, 1 - eps)
                logits = np.log(percentiles_clipped / (1 - percentiles_clipped))
                normalized = 1.0 / (1.0 + np.exp(-logits * T))
                
                # Scale to [0.05, 0.95] linearly to preserve differentiation at extremes
                # (clamp creates flat plateaus at top/bottom where all tracks get same score)
                normalized = 0.05 + normalized * 0.90
                
                logger.info(f"Rank-based normalization: "
                            f"result range=[{normalized.min():.4f}, {normalized.max():.4f}], "
                            f"p5={np.percentile(normalized, 5):.4f}, "
                            f"p25={np.percentile(normalized, 25):.4f}, "
                            f"p50={np.percentile(normalized, 50):.4f}, "
                            f"p75={np.percentile(normalized, 75):.4f}, "
                            f"p95={np.percentile(normalized, 95):.4f}")
            
            # Phase 3: Write to DB
            with worker.db_conn.cursor() as cursor:
                for (tid, _), norm_sc in zip(score_pairs, normalized):
                    cursor.execute(
                        "UPDATE tracks SET preference_score = %s WHERE id = %s",
                        (float(norm_sc), tid)
                    )
            scored_count = len(score_pairs)
            
            logger.info(f"Updated preference_score for {scored_count} tracks "
                        f"(normalized range: {normalized.min():.4f}..{normalized.max():.4f})")
    except Exception as e:
        logger.error(f"Failed to calculate preference_scores: {e}")
        # Non-fatal: model is saved, scores can be recalculated later
    
    logger.info(f"Model training complete, saved to {model_path}")
    return {'model_path': str(model_path), 'final_loss': history['loss'][-1] if history['loss'] else 0}


def _handle_model_training_inner_v2(task_data: dict, worker: QueueWorker) -> dict:
    """
    Improved preference model training.

    Key changes vs legacy implementation:
    - Uses multiple feedback signals (likes, 4-5 star ratings, frequent plays)
    - Uses fragments and rated segments as positive/negative samples without polluting
      the negative sampling pool (all_embeddings is TRACKS ONLY)
    - Scores tracks using max(track_score, best_segment_score) when segment embeddings exist
      so "I like this moment" influences recommendations.
    """
    run_started_at = datetime.now()
    queue_task_id = task_data.get('task_id')
    payload: dict = {}

    if queue_task_id:
        try:
            with worker.db_conn.cursor() as cursor:
                cursor.execute("SELECT payload FROM queue_tasks WHERE id = %s", (queue_task_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        except Exception as e:
            logger.warning(f"Failed to read payload from queue_tasks id={queue_task_id}: {e}")

    user_id = payload.get('user_id', task_data.get('user_id'))
    try:
        user_id = int(user_id) if user_id is not None else None
    except Exception:
        user_id = None
    user_scoped_training = user_id is not None

    liked_track_ids = payload.get('liked_track_ids', task_data.get('liked_track_ids', [])) or []
    high_rated_track_ids = payload.get('high_rated_track_ids') or []
    low_rated_track_ids = payload.get('low_rated_track_ids') or []
    frequently_played_track_ids = payload.get('frequently_played_track_ids') or []
    play_counts = payload.get('play_counts') or {}
    skipped_track_ids = payload.get('skipped_track_ids') or []
    skip_data = payload.get('skip_data') or {}
    completion_rates = payload.get('completion_rates') or {}

    # Fallback: derive missing signals from DB (older payloads)
    if not high_rated_track_ids:
        try:
            with worker.db_conn.cursor() as cursor:
                if user_scoped_training:
                    cursor.execute("SELECT track_id FROM track_ratings WHERE user_id = %s AND rating >= 4", (user_id,))
                else:
                    cursor.execute("SELECT track_id FROM track_ratings WHERE rating >= 4")
                high_rated_track_ids = [r[0] for r in cursor.fetchall()]
        except Exception:
            high_rated_track_ids = []

    if not low_rated_track_ids:
        try:
            with worker.db_conn.cursor() as cursor:
                if user_scoped_training:
                    cursor.execute("SELECT track_id FROM track_ratings WHERE user_id = %s AND rating <= 2", (user_id,))
                else:
                    cursor.execute("SELECT track_id FROM track_ratings WHERE rating <= 2")
                low_rated_track_ids = [r[0] for r in cursor.fetchall()]
        except Exception:
            low_rated_track_ids = []

    if user_scoped_training and not liked_track_ids:
        try:
            with worker.db_conn.cursor() as cursor:
                cursor.execute("SELECT track_id FROM user_track_likes WHERE user_id = %s", (user_id,))
                liked_track_ids = [r[0] for r in cursor.fetchall()]
        except Exception:
            liked_track_ids = []

    if user_scoped_training and not frequently_played_track_ids:
        try:
            with worker.db_conn.cursor() as cursor:
                cursor.execute("""
                    SELECT track_id, COUNT(*) AS play_count
                    FROM listening_history
                    WHERE user_id = %s
                    GROUP BY track_id
                    HAVING COUNT(*) >= 2
                """, (user_id,))
                rows = cursor.fetchall()
                frequently_played_track_ids = [r[0] for r in rows]
                for tid, c in rows:
                    play_counts[str(int(tid))] = int(c)
        except Exception:
            frequently_played_track_ids = []

    if not user_scoped_training and not frequently_played_track_ids:
        try:
            with worker.db_conn.cursor() as cursor:
                cursor.execute("""
                    SELECT track_id, COUNT(*) AS play_count
                    FROM listening_history
                    GROUP BY track_id
                    HAVING COUNT(*) >= 2
                """)
                rows = cursor.fetchall()
                frequently_played_track_ids = [r[0] for r in rows]
                for tid, c in rows:
                    play_counts[str(int(tid))] = int(c)
        except Exception:
            frequently_played_track_ids = []

    logger.info(
        "Training model (v2) with signals: "
        f"user_id={user_id}, "
        f"liked_tracks={len(liked_track_ids)}, "
        f"high_rated_tracks={len(high_rated_track_ids)}, "
        f"low_rated_tracks={len(low_rated_track_ids)}, "
        f"frequently_played_tracks={len(frequently_played_track_ids)}"
    )

    def _load_embedding(path: str):
        if not path:
            return None
        try:
            p = Path(path)
            if not p.exists():
                return None
            emb = np.load(str(p)).astype(np.float32)
            norm = float(np.linalg.norm(emb) + 1e-12)
            return (emb / norm).astype(np.float32)
        except Exception:
            return None

    # Aggregate positives by embedding file path (avoid duplicates, sum weights).
    pos_idx: dict = {}
    pos_embeddings: list = []
    pos_weights: list = []

    def add_pos(path: str, w: float) -> None:
        emb = _load_embedding(path)
        if emb is None:
            return
        key = str(Path(path).resolve())
        if key in pos_idx:
            pos_weights[pos_idx[key]] += float(w)
            return
        pos_idx[key] = len(pos_embeddings)
        pos_embeddings.append(emb)
        pos_weights.append(float(w))

    # Explicit negative pool (segments user disliked).
    neg_idx: dict = {}
    neg_embeddings: list = []
    neg_weights: list = []

    def add_neg(path: str, w: float) -> None:
        emb = _load_embedding(path)
        if emb is None:
            return
        key = str(Path(path).resolve())
        if key in neg_idx:
            neg_weights[neg_idx[key]] += float(w)
            return
        neg_idx[key] = len(neg_embeddings)
        neg_embeddings.append(emb)
        neg_weights.append(float(w))

    # Per-source counters for run summaries.
    pos_by_source = {'track': 0, 'fragment': 0, 'segment': 0}
    neg_by_source = {'track': 0, 'fragment': 0, 'segment': 0}

    def add_pos_src(path: str, w: float, source: str) -> None:
        before = len(pos_embeddings)
        add_pos(path, w)
        if len(pos_embeddings) > before:
            pos_by_source[source] = int(pos_by_source.get(source, 0)) + 1

    def add_neg_src(path: str, w: float, source: str) -> None:
        before = len(neg_embeddings)
        add_neg(path, w)
        if len(neg_embeddings) > before:
            neg_by_source[source] = int(neg_by_source.get(source, 0)) + 1

    # Track-level positive weights
    track_weight_by_id: dict[int, float] = {}
    for tid in liked_track_ids:
        try:
            tid_i = int(tid)
        except Exception:
            continue
        track_weight_by_id[tid_i] = track_weight_by_id.get(tid_i, 0.0) + 1.0

    # Rating bonus
    if high_rated_track_ids:
        try:
            with worker.db_conn.cursor() as cursor:
                if user_scoped_training:
                    cursor.execute(
                        "SELECT track_id, rating FROM track_ratings WHERE user_id = %s AND track_id = ANY(%s)",
                        (user_id, list(map(int, high_rated_track_ids)))
                    )
                else:
                    cursor.execute(
                        "SELECT track_id, rating FROM track_ratings WHERE track_id = ANY(%s)",
                        (list(map(int, high_rated_track_ids)),)
                    )
                for tid, rating in cursor.fetchall():
                    # Additive bonus: 4 -> +0.5, 5 -> +1.0
                    bonus = max(0.0, float(rating) - 3.0) * 0.5
                    tid_i = int(tid)
                    track_weight_by_id[tid_i] = track_weight_by_id.get(tid_i, 0.0) + bonus
        except Exception as e:
            logger.warning(f"Failed to load track ratings: {e}")

    low_track_weight_by_id: dict[int, float] = {}
    if low_rated_track_ids:
        try:
            with worker.db_conn.cursor() as cursor:
                if user_scoped_training:
                    cursor.execute(
                        "SELECT track_id, rating FROM track_ratings WHERE user_id = %s AND track_id = ANY(%s)",
                        (user_id, list(map(int, low_rated_track_ids)))
                    )
                else:
                    cursor.execute(
                        "SELECT track_id, rating FROM track_ratings WHERE track_id = ANY(%s)",
                        (list(map(int, low_rated_track_ids)),)
                    )
                for tid, rating in cursor.fetchall():
                    base = 1.0 + max(0.0, 2.0 - float(rating)) * 0.5
                    tid_i = int(tid)
                    low_track_weight_by_id[tid_i] = max(low_track_weight_by_id.get(tid_i, 0.0), base)
                    # Explicit low ratings override softer positive signals on the same track.
                    track_weight_by_id.pop(tid_i, None)
        except Exception as e:
            logger.warning(f"Failed to load low-rated track ratings: {e}")

    # Play-count bonus — logarithmic scaling (diminishing returns for 100+ plays)
    import math as _math
    for tid in frequently_played_track_ids:
        try:
            tid_i = int(tid)
        except Exception:
            continue
        try:
            c = int(play_counts.get(str(tid_i), play_counts.get(tid_i, 0)) or 0)
        except Exception:
            c = 0
        # log2 scaling: 2 plays→0.0, 4→0.3, 8→0.6, 16→0.9, 32→1.2 (was capped at 0.6)
        bonus = min(1.5, max(0.0, _math.log2(max(1, c)) - 1.0) * 0.3)
        track_weight_by_id[tid_i] = track_weight_by_id.get(tid_i, 0.0) + bonus

    # Completion rate bonus — tracks listened to the end deserve more weight
    for tid_i in list(track_weight_by_id.keys()):
        tid_str = str(tid_i)
        cr = completion_rates.get(tid_str) or completion_rates.get(tid_i)
        if cr:
            rate = float(cr.get('completion_rate', 0) if isinstance(cr, dict) else cr)
            # High completion rate → up to +0.5 bonus; low rate → penalty up to -0.3
            if rate >= 0.7:
                track_weight_by_id[tid_i] += min(0.5, (rate - 0.5) * 1.0)
            elif rate < 0.3 and rate > 0.0:
                track_weight_by_id[tid_i] = max(0.1, track_weight_by_id[tid_i] - 0.3)

    # Skip-based negative signals — frequently skipped tracks become explicit negatives
    skip_neg_weight_by_id: dict[int, float] = {}
    for tid in skipped_track_ids:
        try:
            tid_i = int(tid)
        except Exception:
            continue
        tid_str = str(tid_i)
        sd = skip_data.get(tid_str) or skip_data.get(tid_i)
        if not sd:
            continue
        skip_count = int(sd.get('skip_count', 0) if isinstance(sd, dict) else 0)
        avg_pos = float(sd.get('avg_skip_position', 50) if isinstance(sd, dict) else 50)
        if skip_count < 2:
            continue
        # Early skips (avg position < 30s) are much stronger negative signals
        # Late skips (avg > 120s) are weaker (user listened most of the track)
        if avg_pos < 30:
            base_weight = 1.5  # Strong: skipped within first 30 seconds
        elif avg_pos < 60:
            base_weight = 1.0  # Medium: skipped within first minute
        else:
            base_weight = 0.5  # Weak: skipped late
        # Scale by skip frequency: log2(count) gives diminishing returns
        weight = base_weight * min(2.0, _math.log2(max(1, skip_count)))
        # Don't add as negative if user also explicitly liked it (trust explicit over implicit)
        if tid_i not in track_weight_by_id:
            skip_neg_weight_by_id[tid_i] = max(skip_neg_weight_by_id.get(tid_i, 0.0), weight)

    # Add skip-based negatives to explicit negative pool
    if skip_neg_weight_by_id:
        try:
            with worker.db_conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, embedding_path FROM tracks WHERE id = ANY(%s) AND embedding_path IS NOT NULL",
                    (list(skip_neg_weight_by_id.keys()),)
                )
                for tid, emb_path in cursor.fetchall():
                    add_neg_src(emb_path, skip_neg_weight_by_id.get(int(tid), 1.0), 'track')
        except Exception as e:
            logger.warning(f"Failed to load skip-based negative track embeddings: {e}")

    logger.info(
        "Skip-based signals: skipped_tracks=%d, skip_negatives_added=%d",
        len(skipped_track_ids),
        len(skip_neg_weight_by_id),
    )

    # Load track embeddings for positive track IDs
    if track_weight_by_id:
        try:
            with worker.db_conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, embedding_path FROM tracks WHERE id = ANY(%s) AND embedding_path IS NOT NULL",
                    (list(track_weight_by_id.keys()),)
                )
                for tid, emb_path in cursor.fetchall():
                    add_pos_src(emb_path, track_weight_by_id.get(int(tid), 1.0), 'track')
        except Exception as e:
            logger.warning(f"Failed to load positive track embeddings: {e}")

    if low_track_weight_by_id:
        try:
            with worker.db_conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, embedding_path FROM tracks WHERE id = ANY(%s) AND embedding_path IS NOT NULL",
                    (list(low_track_weight_by_id.keys()),)
                )
                for tid, emb_path in cursor.fetchall():
                    add_neg_src(emb_path, low_track_weight_by_id.get(int(tid), 1.0), 'track')
        except Exception as e:
            logger.warning(f"Failed to load negative track embeddings: {e}")

    fragment_rows = []
    with worker.db_conn.cursor() as cursor:
        if user_scoped_training:
            cursor.execute("""
                SELECT id, embedding_path, comment, is_liked
                FROM fragments
                WHERE user_id = %s
                  AND is_liked IS NOT NULL
                  AND embedding_path IS NOT NULL
            """, (user_id,))
        else:
            cursor.execute("""
                SELECT id, embedding_path, comment, is_liked
                FROM fragments
                WHERE is_liked IS NOT NULL AND embedding_path IS NOT NULL
            """)
        fragment_rows = cursor.fetchall()

    for frag_id, emb_path, comment, is_liked in fragment_rows:
        ca = analyze_comment_sentiment(comment)
        if bool(is_liked):
            add_pos_src(emb_path, 1.0 * float(ca['weight']), 'fragment')
        else:
            # Invert: more negative language should increase negative sample weight.
            add_neg_src(emb_path, 1.0 / max(0.5, float(ca['weight'])), 'fragment')

    # Rated segments (pattern_examples) when segment embedding_path exists
    segment_rows = []
    segments_rated_total = 0
    try:
        with worker.db_conn.cursor() as cursor:
            if user_scoped_training:
                cursor.execute("""
                    SELECT pe.id, upr.rating, pe.embedding_path, COALESCE(p.importance, 1) AS importance
                    FROM user_pattern_example_ratings upr
                    JOIN pattern_examples pe ON pe.id = upr.pattern_example_id
                    JOIN patterns p ON p.id = pe.pattern_id
                    WHERE upr.user_id = %s
                      AND pe.embedding_path IS NOT NULL
                """, (user_id,))
            else:
                cursor.execute("""
                    SELECT pe.id, pe.rating, pe.embedding_path, COALESCE(p.importance, 1) AS importance
                    FROM pattern_examples pe
                    JOIN patterns p ON p.id = pe.pattern_id
                    WHERE pe.rating IS NOT NULL AND pe.embedding_path IS NOT NULL
                """)
            segment_rows = cursor.fetchall()
    except Exception as e:
        logger.warning(f"Failed to load segment ratings: {e}")

    # For "coverage" metrics: count rated segments regardless of embedding availability.
    segments_rated_total = int(len(segment_rows))
    try:
        with worker.db_conn.cursor() as cursor:
            if user_scoped_training:
                cursor.execute("SELECT COUNT(*) FROM user_pattern_example_ratings WHERE user_id = %s", (user_id,))
            else:
                cursor.execute("SELECT COUNT(*) FROM pattern_examples WHERE rating IS NOT NULL")
            segments_rated_total = int(cursor.fetchone()[0] or 0)
    except Exception:
        pass

    for seg_id, rating, emb_path, importance in segment_rows:
        try:
            r = float(rating)
            imp = float(importance or 1)
        except Exception:
            continue

        imp_scale = 1.0 + 0.2 * max(0.0, imp - 1.0)

        if r >= 3.0:
            base = 0.5 + max(0.0, r - 3.0) * 0.5
            add_pos_src(emb_path, min(3.0, base * imp_scale), 'segment')
        elif r <= 2.0:
            base = 1.0 + max(0.0, 2.0 - r) * 0.5
            add_neg_src(emb_path, min(3.0, base * imp_scale), 'segment')

    if not pos_embeddings:
        raise ValueError("No positive embeddings found for training")

    liked_embeddings = np.array(pos_embeddings, dtype=np.float32)
    liked_weights = np.array(pos_weights, dtype=np.float32)

    explicit_neg_embeddings = np.array(neg_embeddings, dtype=np.float32) if neg_embeddings else None
    explicit_neg_weights = np.array(neg_weights, dtype=np.float32) if neg_weights else None

    # Negative sampling pool: track embeddings only
    all_track_embeddings: list = []
    all_track_paths: set = set()
    all_track_score_rows: list[tuple[int, np.ndarray]] = []
    embedding_cache: dict[str, np.ndarray] = {}
    with worker.db_conn.cursor() as cursor:
        cursor.execute("SELECT id, embedding_path FROM tracks WHERE embedding_path IS NOT NULL")
        for track_id, emb_path in cursor.fetchall():
            if not emb_path:
                continue
            key = str(Path(emb_path).resolve())
            emb = embedding_cache.get(key)
            if emb is None:
                emb = _load_embedding(emb_path)
                if emb is not None:
                    embedding_cache[key] = emb
            if emb is None:
                continue
            all_track_score_rows.append((int(track_id), emb))
            if key not in all_track_paths:
                all_track_paths.add(key)
                all_track_embeddings.append(emb)

    if not all_track_embeddings:
        raise ValueError("No track embeddings found for negative sampling")

    all_embeddings = np.array(all_track_embeddings, dtype=np.float32)

    logger.info(
        "Training dataset sizes (v2): "
        f"positives={len(liked_embeddings)} "
        f"(tracks={len(track_weight_by_id)}, fragments={len(fragment_rows)}, segments={len(segment_rows)}), "
        f"explicit_negatives={0 if explicit_neg_embeddings is None else len(explicit_neg_embeddings)}, "
        f"all_tracks={len(all_embeddings)}"
    )

    model = PreferenceModel(embedding_dim=liked_embeddings.shape[1], learning_rate=0.003)
    history = model.train(
        liked_embeddings,
        all_embeddings,
        epochs=300,
        sample_weights=liked_weights,
        explicit_negative_embeddings=explicit_neg_embeddings,
        explicit_negative_weights=explicit_neg_weights,
        explicit_negative_prob=0.6,
        use_hard_negatives=True
    )

    # Save model
    models_dir = Path(os.getenv('MODELS_PATH', '/app/models'))
    if user_scoped_training:
        models_dir = models_dir / 'users' / str(user_id)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / 'preference_model.npz'
    model.save(str(model_path))

    # Update preference_model_status in database (create row if missing)
    trained_at = datetime.now()
    likes_count = int(len(liked_embeddings))
    with worker.db_conn.cursor() as cursor:
        if user_scoped_training:
            cursor.execute("SELECT id FROM preference_model_status WHERE user_id = %s LIMIT 1", (user_id,))
        else:
            cursor.execute("SELECT id FROM preference_model_status WHERE user_id IS NULL LIMIT 1")
        row = cursor.fetchone()
        if row:
            if user_scoped_training:
                cursor.execute(
                    """
                    UPDATE preference_model_status
                    SET status = 'trained',
                        model_path = %s,
                        trained_at = %s,
                        likes_count = %s,
                        updated_at = %s
                    WHERE id = %s AND user_id = %s
                    """,
                    (str(model_path), trained_at, likes_count, trained_at, int(row[0]), user_id)
                )
            else:
                cursor.execute(
                    """
                    UPDATE preference_model_status
                    SET status = 'trained',
                        model_path = %s,
                        trained_at = %s,
                        likes_count = %s,
                        updated_at = %s
                    WHERE id = %s
                    """,
                    (str(model_path), trained_at, likes_count, trained_at, int(row[0]))
                )
        else:
            if user_scoped_training:
                cursor.execute(
                    """
                    INSERT INTO preference_model_status
                        (user_id, status, model_path, trained_at, likes_count, created_at, updated_at)
                    VALUES
                        (%s, 'trained', %s, %s, %s, %s, %s)
                    """,
                    (user_id, str(model_path), trained_at, likes_count, trained_at, trained_at)
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO preference_model_status
                        (status, model_path, trained_at, likes_count, created_at, updated_at)
                    VALUES
                        ('trained', %s, %s, %s, %s, %s)
                    """,
                    (str(model_path), trained_at, likes_count, trained_at, trained_at)
                )

    # Score all tracks. If a track has segment embeddings, use:
    # raw = max(track_embedding_score, best_segment_score)
    logger.info("Calculating preference_score for all tracks with embeddings (v2)...")

    # Load segment embeddings per track (best-effort)
    segment_paths_by_track: dict[int, list] = {}
    if not user_scoped_training:
        try:
            with worker.db_conn.cursor() as cursor:
                cursor.execute("""
                    SELECT pe.track_id, pe.embedding_path, COALESCE(p.importance, 1) AS importance
                    FROM pattern_examples pe
                    JOIN patterns p ON p.id = pe.pattern_id
                    WHERE pe.embedding_path IS NOT NULL
                """)
                for track_id, emb_path, importance in cursor.fetchall():
                    if not emb_path:
                        continue
                    if not Path(emb_path).exists():
                        continue
                    tid = int(track_id)
                    segment_paths_by_track.setdefault(tid, []).append((float(importance or 1), emb_path))

            for tid, items in segment_paths_by_track.items():
                items.sort(key=lambda x: x[0], reverse=True)
                segment_paths_by_track[tid] = [p for _imp, p in items]
        except Exception as e:
            logger.warning(f"Failed to load segment embedding paths for scoring: {e}")
            segment_paths_by_track = {}

    def _build_anchor_bank(
        embeddings: "np.ndarray | None",
        max_items: int,
    ) -> "np.ndarray | None":
        if embeddings is None or len(embeddings) == 0:
            return None

        emb_arr = np.asarray(embeddings, dtype=np.float32)
        return emb_arr[:max_items]

    POS_ANCHOR_TOP_K = 12
    NEG_ANCHOR_TOP_K = 12
    pos_anchor_bank = _build_anchor_bank(liked_embeddings, max_items=384)
    neg_anchor_bank = _build_anchor_bank(explicit_neg_embeddings, max_items=192)

    def _anchor_margin(candidate_embeddings: "np.ndarray | list") -> np.ndarray:
        return model.score_anchor_margin(
            candidate_embeddings,
            positive_anchor_embeddings=pos_anchor_bank,
            negative_anchor_embeddings=neg_anchor_bank,
            positive_top_k=POS_ANCHOR_TOP_K,
            negative_top_k=NEG_ANCHOR_TOP_K,
        )

    scored_count = 0
    tracks_with_segments = 0
    segment_improved = 0
    score_distribution_meta = {
        'mode': 'anchor_margin_robust_sigmoid',
        'raw_std': 0.0,
        'normalized_min': 0.0,
        'normalized_max': 0.0,
        'normalized_p50': 0.0,
        'normalized_p95': 0.0,
    }

    try:
        positive_reference_scores = _anchor_margin(liked_embeddings)
        negative_reference_source = (
            explicit_neg_embeddings
            if explicit_neg_embeddings is not None and len(explicit_neg_embeddings) > 0
            else all_embeddings
        )
        if negative_reference_source is all_embeddings and len(all_embeddings) > 1024:
            negative_reference_source = all_embeddings[np.random.choice(len(all_embeddings), size=1024, replace=False)]
        negative_reference_scores = _anchor_margin(negative_reference_source)

        logger.info(
            "Score calibration references (anchor_margin): "
            f"pos_median={np.median(positive_reference_scores):.6f}, "
            f"pos_p25={np.percentile(positive_reference_scores, 25):.6f}, "
            f"neg_median={np.median(negative_reference_scores):.6f}, "
            f"neg_p75={np.percentile(negative_reference_scores, 75):.6f}"
        )

        score_pairs: list = []  # [(track_id, raw_score), ...]
        MAX_SEGMENTS_PER_TRACK = 6
        track_raw_scores_by_id: dict[int, float] = {}
        if all_track_score_rows:
            batched_track_embeddings = np.array([emb for _, emb in all_track_score_rows], dtype=np.float32)
            batched_track_raw_scores = _anchor_margin(batched_track_embeddings)
            track_raw_scores_by_id = {
                int(track_id): float(raw_score)
                for (track_id, _), raw_score in zip(all_track_score_rows, batched_track_raw_scores)
            }

        for track_id, emb in all_track_score_rows:
            track_raw = track_raw_scores_by_id.get(int(track_id))
            if track_raw is None:
                continue
            best_seg_raw = None

            seg_paths = segment_paths_by_track.get(int(track_id))
            if seg_paths:
                tracks_with_segments += 1
                seg_embs = []
                for p in seg_paths[:MAX_SEGMENTS_PER_TRACK]:
                    se = _load_embedding(p)
                    if se is not None:
                        seg_embs.append(se)
                if seg_embs:
                    seg_scores = _anchor_margin(np.array(seg_embs, dtype=np.float32))
                    best_seg_raw = float(np.max(seg_scores))
                    if best_seg_raw > track_raw:
                        segment_improved += 1

            raw = max(track_raw, best_seg_raw) if best_seg_raw is not None else track_raw
            score_pairs.append((int(track_id), raw))

        if not score_pairs:
            logger.warning("No tracks to score")
        else:
            raw_scores = np.array([s for _, s in score_pairs])

            logger.info(
                f"Raw scores: min={raw_scores.min():.6f}, max={raw_scores.max():.6f}, "
                f"mean={raw_scores.mean():.6f}, std={raw_scores.std():.6f}"
            )

            unique_raw, raw_counts = np.unique(raw_scores, return_counts=True)
            logger.info(f"Unique raw scores: {len(unique_raw)}, largest tie cluster: {raw_counts.max()} tracks")

            normalized = PreferenceModel.calibrate_scores(raw_scores)

            score_distribution_meta = {
                'mode': 'anchor_margin_robust_sigmoid',
                'raw_std': float(raw_scores.std()),
                'normalized_min': float(normalized.min()),
                'normalized_max': float(normalized.max()),
                'normalized_p50': float(np.percentile(normalized, 50)),
                'normalized_p95': float(np.percentile(normalized, 95)),
            }

            logger.info(
                "Robust-score calibration (anchor_margin): "
                f"range=[{normalized.min():.4f}, {normalized.max():.4f}], "
                f"p50={np.percentile(normalized, 50):.4f}, "
                f"p75={np.percentile(normalized, 75):.4f}, "
                f"p90={np.percentile(normalized, 90):.4f}, "
                f"p95={np.percentile(normalized, 95):.4f}, "
                f"p99={np.percentile(normalized, 99):.4f}"
            )

            # Write to DB (best-effort include score_calculated_at)
            try:
                with worker.db_conn.cursor() as cursor:
                    for (tid, _), norm_sc in zip(score_pairs, normalized):
                        if user_scoped_training:
                            cursor.execute(
                                """
                                INSERT INTO user_track_scores
                                    (user_id, track_id, preference_score, score_calculated_at, created_at, updated_at)
                                VALUES
                                    (%s, %s, %s, NOW(), NOW(), NOW())
                                ON CONFLICT (user_id, track_id)
                                DO UPDATE SET
                                    preference_score = EXCLUDED.preference_score,
                                    score_calculated_at = EXCLUDED.score_calculated_at,
                                    updated_at = NOW()
                                """,
                                (user_id, int(tid), float(norm_sc))
                            )
                        else:
                            cursor.execute(
                                "UPDATE tracks SET preference_score = %s, score_calculated_at = NOW() WHERE id = %s",
                                (float(norm_sc), int(tid))
                            )
                scored_count = len(score_pairs)
            except Exception:
                with worker.db_conn.cursor() as cursor:
                    for (tid, _), norm_sc in zip(score_pairs, normalized):
                        if user_scoped_training:
                            cursor.execute(
                                """
                                INSERT INTO user_track_scores
                                    (user_id, track_id, preference_score, created_at, updated_at)
                                VALUES
                                    (%s, %s, %s, NOW(), NOW())
                                ON CONFLICT (user_id, track_id)
                                DO UPDATE SET
                                    preference_score = EXCLUDED.preference_score,
                                    updated_at = NOW()
                                """,
                                (user_id, int(tid), float(norm_sc))
                            )
                        else:
                            cursor.execute(
                                "UPDATE tracks SET preference_score = %s WHERE id = %s",
                                (float(norm_sc), int(tid))
                            )
                scored_count = len(score_pairs)

            logger.info(
                f"Updated {'user_track_scores' if user_scoped_training else 'preference_score'} for {scored_count} tracks. "
                f"tracks_with_segments={tracks_with_segments}, segment_improved={segment_improved}"
            )

    except Exception as e:
        logger.error(f"Failed to calculate preference_scores (v2): {e}")

    logger.info(f"Model training complete (v2), saved to {model_path}")

    # Persist training summary (best-effort) for debugging/regression.
    run_finished_at = datetime.now()
    final_loss = float(history['loss'][-1]) if history.get('loss') else 0.0

    positives_total = int(len(liked_embeddings))
    positives_tracks = int(pos_by_source.get('track', 0))
    positives_fragments = int(pos_by_source.get('fragment', 0))
    positives_segments = int(pos_by_source.get('segment', 0))

    explicit_neg_total = int(0 if explicit_neg_embeddings is None else len(explicit_neg_embeddings))
    explicit_neg_tracks = int(neg_by_source.get('track', 0))
    explicit_neg_fragments = int(neg_by_source.get('fragment', 0))
    explicit_neg_segments = int(neg_by_source.get('segment', 0))

    segments_rated_with_embedding = int(len(segment_rows))
    segment_positive_fraction = float(positives_segments / max(1, positives_total))

    summary_meta = {
        'version': 'v2',
        'user_id': user_id,
        'training_scope': 'user_ratings_only' if user_scoped_training else 'global_feedback',
        'signals': {
            'liked_tracks': int(len(liked_track_ids)),
            'high_rated_tracks': int(len(high_rated_track_ids)),
            'low_rated_tracks': int(len(low_rated_track_ids)),
            'frequently_played_tracks': int(len(frequently_played_track_ids)),
        },
        'hyperparams': {
            'epochs': 300,
            'learning_rate': 0.003,
            'explicit_negative_prob': 0.6,
            'use_hard_negatives': True,
        },
        'metrics': {
            'segment_positive_fraction': segment_positive_fraction,
        },
        'runtime': {
            'training_device': history.get('device', 'unknown'),
            'score_calibration': score_distribution_meta,
        },
    }

    logger.info(
        "Training run summary (v2): "
        + json.dumps(
            {
                'queue_task_id': queue_task_id,
                'final_loss': final_loss,
                'positives': {
                    'total': positives_total,
                    'tracks': positives_tracks,
                    'fragments': positives_fragments,
                    'segments': positives_segments,
                },
                'explicit_negatives': {
                    'total': explicit_neg_total,
                    'tracks': explicit_neg_tracks,
                    'fragments': explicit_neg_fragments,
                    'segments': explicit_neg_segments,
                },
                'segments_rated': {
                    'total': int(segments_rated_total),
                    'with_embedding': segments_rated_with_embedding,
                },
                'scoring': {
                    'tracks_scored': int(scored_count),
                    'tracks_with_segments': int(tracks_with_segments),
                    'segment_improved': int(segment_improved),
                },
            },
            sort_keys=True,
        )
    )

    try:
        qt_id = None
        try:
            qt_id = int(queue_task_id) if queue_task_id is not None else None
        except Exception:
            qt_id = None

        with worker.db_conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO preference_model_training_runs (
                    queue_task_id,
                    user_id,
                    model_path,
                    started_at,
                    finished_at,
                    final_loss,
                    positives_total,
                    positives_tracks,
                    positives_fragments,
                    positives_segments,
                    explicit_negatives_total,
                    explicit_negative_fragments,
                    explicit_negative_segments,
                    all_tracks_pool,
                    tracks_scored,
                    tracks_with_segments,
                    segment_improved,
                    segments_rated_total,
                    segments_rated_with_embedding,
                    meta,
                    created_at,
                    updated_at
                ) VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    CAST(%s AS jsonb),
                    NOW(),
                    NOW()
                )
                """,
                (
                    qt_id,
                    user_id,
                    str(model_path),
                    run_started_at,
                    run_finished_at,
                    final_loss,
                    positives_total,
                    positives_tracks,
                    positives_fragments,
                    positives_segments,
                    explicit_neg_total,
                    explicit_neg_fragments,
                    explicit_neg_segments,
                    int(len(all_embeddings)),
                    int(scored_count),
                    int(tracks_with_segments),
                    int(segment_improved),
                    int(segments_rated_total),
                    segments_rated_with_embedding,
                    json.dumps(summary_meta, sort_keys=True),
                ),
            )
    except Exception as e:
        logger.warning(f"Failed to persist training run summary (v2): {e}")

    return {'model_path': str(model_path), 'final_loss': history['loss'][-1] if history.get('loss') else 0}


def handle_dj_analysis(task_data: dict, worker: QueueWorker, dj_analyzer: AIDJAnalyzer) -> dict:
    """
    Handle DJ analysis task.
    
    Analyzes track for DJ features: Camelot key, BPM, energy, structural points.
    Saves results to track_dj_analysis table.
    
    Requirements: 27.5, 27.6
    """
    track_id = task_data['track_id']
    audio_path = task_data['audio_path']
    
    logger.info(f"DJ analyzing track {track_id}: {audio_path}")
    
    # Perform DJ analysis
    result = dj_analyzer.analyze(audio_path, track_id)
    
    # Save to database
    with worker.db_conn.cursor() as cursor:
        # Check if analysis already exists
        cursor.execute(
            "SELECT id FROM track_dj_analysis WHERE track_id = %s",
            (track_id,)
        )
        existing = cursor.fetchone()
        
        import json
        drop_points_json = json.dumps(result.drop_points)
        breakdown_points_json = json.dumps(result.breakdown_points)
        
        if existing:
            # Update existing
            cursor.execute(
                """
                UPDATE track_dj_analysis 
                SET camelot_key = %s, bpm = %s, energy_level = %s,
                    intro_point = %s, outro_point = %s,
                    drop_points = %s, breakdown_points = %s,
                    analyzed_at = %s
                WHERE track_id = %s
                """,
                (
                    result.camelot_key,
                    result.bpm,
                    result.energy_level,
                    result.intro_point,
                    result.outro_point,
                    drop_points_json,
                    breakdown_points_json,
                    datetime.now(),
                    track_id
                )
            )
        else:
            # Insert new
            cursor.execute(
                """
                INSERT INTO track_dj_analysis 
                (track_id, camelot_key, bpm, energy_level, intro_point, outro_point, 
                 drop_points, breakdown_points, analyzed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    track_id,
                    result.camelot_key,
                    result.bpm,
                    result.energy_level,
                    result.intro_point,
                    result.outro_point,
                    drop_points_json,
                    breakdown_points_json,
                    datetime.now()
                )
            )
    
    logger.info(f"DJ analysis complete for track {track_id}: {result.camelot_key}, {result.bpm} BPM")
    
    return {
        'track_id': track_id,
        'camelot_key': result.camelot_key,
        'bpm': result.bpm,
        'energy_level': result.energy_level,
    }


def handle_segment_search(
    task_data: dict,
    worker: QueueWorker,
    analyzer: AudioAnalyzer,
    segment_index: SegmentIndexManager
) -> dict:
    """
    Handle segment similarity search task.
    
    Extracts embedding from specified time range and searches for similar segments.
    
    Requirements: 3.1, 3.2, 3.3, 3.4
    """
    track_id = task_data['track_id']
    start_time = task_data['start_time']
    end_time = task_data['end_time']
    limit = task_data.get('limit', 10)
    min_similarity = task_data.get('min_similarity', 0.3)
    
    logger.info(f"Segment search: track {track_id}, {start_time:.1f}s - {end_time:.1f}s")
    
    # Get track audio path from database
    with worker.db_conn.cursor() as cursor:
        cursor.execute("SELECT file_path FROM tracks WHERE id = %s", (track_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Track {track_id} not found")
        file_path = row[0]
    
    audio_path = f"/app/storage/{file_path}"
    
    # Extract embedding for the query segment
    query_embedding = analyzer.extract_embeddings(audio_path, start_time=start_time, end_time=end_time)
    
    # Search for similar segments
    matches = segment_index.search_similar(
        query_embedding,
        k=limit,
        exclude_track_id=track_id,
        min_similarity=min_similarity
    )
    
    # Enrich results with track info
    results = []
    for match in matches:
        with worker.db_conn.cursor() as cursor:
            cursor.execute(
                "SELECT title, artist FROM tracks WHERE id = %s",
                (match.track_id,)
            )
            track_row = cursor.fetchone()
            if track_row:
                results.append({
                    'fragment_id': match.fragment_id,
                    'track_id': match.track_id,
                    'track_title': track_row[0],
                    'artist': track_row[1],
                    'start_time': match.start_time,
                    'end_time': match.end_time,
                    'similarity': match.similarity
                })
    
    logger.info(f"Segment search found {len(results)} matches")
    
    return {
        'query': {
            'track_id': track_id,
            'start_time': start_time,
            'end_time': end_time
        },
        'results': results,
        'total': len(results)
    }


def handle_embedding_recommendations(task_data: dict, worker: QueueWorker) -> dict:
    """
    Handle personalized recommendations based on embeddings.
    
    Uses cosine similarity between liked track embeddings and candidate tracks
    to find the most similar unheard tracks.
    
    Returns recommendations with explanations based on which liked tracks
    they are most similar to.
    """
    import redis
    
    limit = task_data.get('limit', 5)
    task_id = task_data.get('task_id')
    user_id = task_data.get('user_id')
    try:
        user_id = int(user_id) if user_id is not None else None
    except (TypeError, ValueError):
        user_id = None
    
    logger.info(f"Generating embedding-based recommendations (limit={limit}, user_id={user_id})")
    
    storage_path = Path(os.getenv('STORAGE_PATH', '/app/storage'))
    embeddings_dir = storage_path / 'embeddings'
    
    # Get liked tracks with embeddings
    with worker.db_conn.cursor() as cursor:
        if user_id is None:
            cursor.execute("""
                SELECT t.id, t.title, t.artist, t.genre, t.bpm, t.mood, t.embedding_path
                FROM tracks t
                WHERE t.is_liked = true AND t.embedding_path IS NOT NULL
            """)
        else:
            cursor.execute("""
                SELECT t.id, t.title, t.artist, t.genre, t.bpm, t.mood, t.embedding_path
                FROM tracks t
                JOIN user_track_likes utl ON utl.track_id = t.id
                WHERE utl.user_id = %s
                  AND t.embedding_path IS NOT NULL
            """, (user_id,))
        liked_tracks = cursor.fetchall()
        
        # Also get highly rated tracks (4-5 stars)
        if user_id is None:
            cursor.execute("""
                SELECT t.id, t.title, t.artist, t.genre, t.bpm, t.mood, t.embedding_path
                FROM tracks t
                JOIN track_ratings tr ON t.id = tr.track_id
                WHERE tr.rating >= 4 AND t.embedding_path IS NOT NULL
            """)
        else:
            cursor.execute("""
                SELECT t.id, t.title, t.artist, t.genre, t.bpm, t.mood, t.embedding_path
                FROM tracks t
                JOIN track_ratings tr ON t.id = tr.track_id
                WHERE tr.user_id = %s
                  AND tr.rating >= 4
                  AND t.embedding_path IS NOT NULL
            """, (user_id,))
        rated_tracks = cursor.fetchall()

        if user_id is None:
            cursor.execute("""
                SELECT pe.id, pe.start_time, pe.end_time, t.id, t.title, t.artist, t.genre, t.bpm, t.mood, pe.embedding_path
                FROM pattern_examples pe
                JOIN tracks t ON t.id = pe.track_id
                WHERE pe.rating >= 4
                  AND pe.embedding_path IS NOT NULL
                ORDER BY pe.rating DESC, COALESCE(pe.updated_at, pe.created_at) DESC
                LIMIT 256
            """)
        else:
            cursor.execute("""
                SELECT pe.id, pe.start_time, pe.end_time, t.id, t.title, t.artist, t.genre, t.bpm, t.mood, pe.embedding_path
                FROM user_pattern_example_ratings upr
                JOIN pattern_examples pe ON pe.id = upr.pattern_example_id
                JOIN tracks t ON t.id = pe.track_id
                WHERE upr.user_id = %s
                  AND upr.rating >= 4
                  AND pe.embedding_path IS NOT NULL
                ORDER BY upr.rating DESC, COALESCE(upr.updated_at, upr.created_at) DESC
                LIMIT 256
            """, (user_id,))
        rated_segments = cursor.fetchall()
    
    # Combine and deduplicate
    liked_map = {}
    for row in liked_tracks + rated_tracks:
        track_id, title, artist, genre, bpm, mood, emb_path = row
        if track_id not in liked_map:
            liked_map[track_id] = {
                'id': track_id,
                'title': title,
                'artist': artist,
                'genre': genre,
                'bpm': bpm,
                'mood': mood,
                'embedding_path': emb_path
            }
    
    if not liked_map and not rated_segments:
        result = {
            'recommendations': [],
            'has_preferences': False,
            'message': 'Недостаточно данных. Поставьте лайки или оценки трекам.'
        }
        _store_result(task_id, result)
        return result
    
    # Load liked embeddings
    liked_embeddings = []
    liked_info = []
    for track_id, info in liked_map.items():
        emb_path = info['embedding_path']
        if emb_path and Path(emb_path).exists():
            emb = np.load(emb_path)
            liked_embeddings.append(emb)
            liked_info.append({**info, 'source': 'track'})

    for seg_id, start_time, end_time, track_id, title, artist, genre, bpm, mood, emb_path in rated_segments:
        if emb_path and Path(emb_path).exists():
            emb = np.load(emb_path)
            liked_embeddings.append(emb)
            liked_info.append({
                'id': f"segment:{seg_id}",
                'track_id': track_id,
                'title': title,
                'artist': artist,
                'genre': genre,
                'bpm': bpm,
                'mood': mood,
                'embedding_path': emb_path,
                'source': 'segment',
                'start_time': float(start_time),
                'end_time': float(end_time),
            })
    
    if not liked_embeddings:
        result = {
            'recommendations': [],
            'has_preferences': False,
            'message': 'Не найдены embeddings для лайкнутых треков.'
        }
        _store_result(task_id, result)
        return result
    
    # Normalize liked embeddings once for cosine similarity via dot product
    liked_embeddings = np.array(liked_embeddings, dtype=np.float32)
    liked_norms = np.linalg.norm(liked_embeddings, axis=1, keepdims=True)
    liked_embeddings_norm = liked_embeddings / (liked_norms + 1e-10)

    # Get listening history to exclude
    with worker.db_conn.cursor() as cursor:
        if user_id is None:
            cursor.execute("SELECT DISTINCT track_id FROM listening_history")
        else:
            cursor.execute("SELECT DISTINCT track_id FROM listening_history WHERE user_id = %s", (user_id,))
        listened_ids = set(row[0] for row in cursor.fetchall())
    
    # Get candidate tracks (not liked, not listened, with embeddings)
    liked_ids = set(liked_map.keys())
    with worker.db_conn.cursor() as cursor:
        if user_id is None:
            cursor.execute("""
                SELECT t.id, t.title, t.artist, t.genre, t.bpm, t.mood, t.key, t.embedding_path
                FROM tracks t
                WHERE t.is_liked = false 
                  AND t.analysis_status = 'completed'
                  AND t.embedding_path IS NOT NULL
            """)
        else:
            cursor.execute("""
                SELECT t.id, t.title, t.artist, t.genre, t.bpm, t.mood, t.key, t.embedding_path
                FROM tracks t
                WHERE t.analysis_status = 'completed'
                  AND t.embedding_path IS NOT NULL
            """)
        candidates = cursor.fetchall()
    
    # Filter out listened/liked tracks
    candidates = [c for c in candidates if c[0] not in listened_ids and c[0] not in liked_ids]
    
    if not candidates:
        result = {
            'recommendations': [],
            'has_preferences': True,
            'message': 'Вы уже прослушали все доступные треки!'
        }
        _store_result(task_id, result)
        return result

    # Load all candidate embeddings in one pass, then score via matrix multiply
    # This avoids O(N*M) np.load calls and per-candidate Python loops
    valid_candidates = []
    cand_embeddings_list = []
    for candidate in candidates:
        track_id, title, artist, genre, bpm, mood, key, emb_path = candidate
        if emb_path and Path(emb_path).exists():
            cand_embeddings_list.append(np.load(emb_path).astype(np.float32))
            valid_candidates.append(candidate)

    if not valid_candidates:
        result = {
            'recommendations': [],
            'has_preferences': True,
            'message': 'Не нашёл подходящих треков.'
        }
        _store_result(task_id, result)
        return result

    # Stack and normalize: shape (n_cand, dim)
    cand_matrix = np.array(cand_embeddings_list, dtype=np.float32)
    cand_norms = np.linalg.norm(cand_matrix, axis=1, keepdims=True)
    cand_matrix_norm = cand_matrix / (cand_norms + 1e-10)

    # Similarity matrix: (n_cand, n_liked) — single BLAS call
    sim_matrix = cand_matrix_norm @ liked_embeddings_norm.T  # shape (n_cand, n_liked)

    top_genres = [info['genre'] for info in liked_info if info['genre']]
    top_moods = [info['mood'] for info in liked_info if info['mood']]

    recommendations = []
    for i, candidate in enumerate(valid_candidates):
        track_id, title, artist, genre, bpm, mood, key, emb_path = candidate
        sims = sim_matrix[i]  # shape (n_liked,)

        # Top-3 most similar liked tracks
        top_idx = np.argsort(sims)[::-1][:3]
        avg_similarity = float(np.mean(sims[top_idx]))

        reasons = []
        for j in top_idx:
            sim = float(sims[j])
            liked_track = liked_info[j]
            if liked_track.get('source') == 'segment':
                segment_label = f"{liked_track['title']} [{liked_track['start_time']:.1f}-{liked_track['end_time']:.1f}s]"
                if sim > 0.7:
                    reasons.append(f"Похож по звуку на любимый сегмент \"{segment_label}\" ({sim*100:.0f}%)")
                elif sim > 0.5:
                    reasons.append(f"Есть сходство с сегментом \"{segment_label}\" ({sim*100:.0f}%)")
                continue
            if sim > 0.7:
                reasons.append(f"Очень похож на \"{liked_track['title']}\" ({sim*100:.0f}%)")
            elif sim > 0.5:
                reasons.append(f"Похож на \"{liked_track['title']}\" ({sim*100:.0f}%)")

        if genre and genre in top_genres:
            reasons.append(f"Жанр {genre} — ваш любимый")
        if mood and mood in top_moods:
            reasons.append(f"Настроение \"{mood}\" вам подходит")

        recommendations.append({
            'track': {
                'id': track_id,
                'title': title,
                'artist': artist,
                'genre': genre,
                'bpm': bpm,
                'mood': mood,
                'key': key
            },
            'score': avg_similarity,
            'reasons': reasons[:4],
            'explanation': '. '.join(reasons[:3]) if reasons else 'Может вам понравиться'
        })

    def diversify_by_genre(rows: list[dict], max_items: int) -> list[dict]:
        pool = sorted(rows, key=lambda row: row['score'], reverse=True)[:max(max_items * 5, 30)]
        selected = []
        genre_counts: dict[str, int] = {}

        while pool and len(selected) < max_items:
            best_idx = 0
            best_adjusted_score = None
            for idx, row in enumerate(pool):
                genre_key = str((row.get('track') or {}).get('genre') or '').strip().lower()
                penalty = min(0.06, 0.025 * genre_counts.get(genre_key, 0)) if genre_key else 0.0
                adjusted_score = float(row['score']) - penalty
                if best_adjusted_score is None or adjusted_score > best_adjusted_score:
                    best_idx = idx
                    best_adjusted_score = adjusted_score

            chosen = pool.pop(best_idx)
            selected.append(chosen)
            genre_key = str((chosen.get('track') or {}).get('genre') or '').strip().lower()
            if genre_key:
                genre_counts[genre_key] = genre_counts.get(genre_key, 0) + 1

        return selected

    recommendations = diversify_by_genre(recommendations, int(limit))
    
    # Build preference summary
    bpm_values = [info['bpm'] for info in liked_info if info['bpm']]

    preference_summary = {
        'top_genres': list(set(info['genre'] for info in liked_info if info['genre']))[:3],
        'top_moods': list(set(info['mood'] for info in liked_info if info['mood']))[:2],
        'avg_bpm': int(np.mean(bpm_values)) if bpm_values else 140,
        'tracks_analyzed': len(liked_info)
    }
    
    result = {
        'recommendations': recommendations,
        'total': len(recommendations),
        'preference_summary': preference_summary,
        'has_preferences': True,
        'message': None if recommendations else 'Не нашёл подходящих треков.'
    }
    
    logger.info(f"Generated {len(recommendations)} embedding-based recommendations")
    
    # Store result in Redis for PHP to retrieve
    _store_result(task_id, result)
    
    return result


def _store_result(task_id: str, result: dict) -> None:
    """Store result in Redis for retrieval by PHP."""
    if not task_id:
        return
    
    import redis
    import json
    
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'redis'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD') or None,
        decode_responses=True
    )
    
    result_key = f"chat_recommend_result:{task_id}"
    redis_client.setex(result_key, 60, json.dumps(result))  # 60 second TTL
    logger.info(f"Stored result in Redis: {result_key}")


def _stem_tuning_render_child_entry(progress_queue, child_args: dict[str, Any]) -> None:
    def report_progress(progress: int, message: str, extra: dict[str, Any] | None = None) -> None:
        progress_queue.put({
            'kind': 'progress',
            'progress': int(progress),
            'message': str(message),
            'extra': extra or {},
        })

    try:
        result = render_stem_tuning(
            child_args['source_stem_manifest_path'],
            child_args['output_dir'],
            child_args['preset'],
            render_id=child_args['render_id'],
            latest_pointer_path=child_args['latest_pointer_path'],
            source_audio_path=child_args.get('source_audio_path'),
            raw_reference_path=child_args.get('raw_reference_path'),
            dsp_device=child_args.get('dsp_device', 'cuda'),
            progress_hook=report_progress,
        )
        progress_queue.put({
            'kind': 'result',
            'result': result,
        })
    except Exception as exc:
        progress_queue.put({
            'kind': 'error',
            'error_message': str(exc),
            'error_type': exc.__class__.__name__,
            'traceback': traceback.format_exc(),
        })


def _drain_stem_tuning_progress_event(
    event: dict[str, Any],
    report_progress,
    state: dict[str, Any],
) -> None:
    kind = str(event.get('kind') or '')
    if kind == 'progress':
        progress = int(event.get('progress') or 0)
        message = str(event.get('message') or '')
        extra = event.get('extra')
        extra_payload = extra if isinstance(extra, dict) else {}
        state['last_progress'] = progress
        state['last_message'] = message
        report_progress(progress, message, extra_payload)
        return

    if kind == 'result':
        state['result'] = event.get('result')
        return

    if kind == 'error':
        state['error_message'] = str(event.get('error_message') or 'Stem tuning render process failed')
        state['error_traceback'] = str(event.get('traceback') or '')


def handle_stem_tuning_render(task_data: dict, worker: QueueWorker) -> dict:
    task_id = int(task_data.get('task_id') or 0)
    source_stem_manifest_path = str(task_data.get('source_stem_manifest_path') or '').strip()
    output_dir = str(task_data.get('output_dir') or '').strip()
    latest_pointer_path = str(task_data.get('latest_pointer_path') or '').strip()
    render_id = str(task_data.get('render_id') or task_data.get('task_id') or '').strip()
    preset = task_data.get('preset')
    source_audio_path = task_data.get('source_audio_path')

    if source_stem_manifest_path == '' or output_dir == '' or latest_pointer_path == '':
        raise ValueError('Stem tuning render task is missing required paths')
    if render_id == '':
        raise ValueError('Stem tuning render task is missing render id')
    if not isinstance(preset, dict):
        raise ValueError('Stem tuning render task is missing a valid preset payload')

    raw_reference_path = task_data.get('raw_reference_path')
    if task_id > 0:
        worker.update_stem_tuning_render_status(
            task_id,
            'processing',
            1,
            'Task picked up by ML worker',
            render_id=render_id,
            stage='worker_start',
        )
    logger.info(
        "Stem tuning render start render_id=%s source_manifest=%s output_dir=%s",
        render_id,
        source_stem_manifest_path,
        output_dir,
    )

    def report_progress(progress: int, message: str, extra: dict[str, Any] | None = None) -> None:
        if task_id <= 0:
            return
        worker.update_stem_tuning_render_status(
            task_id,
            'processing',
            progress,
            message,
            render_id=render_id,
            **(extra or {}),
        )

    worker.clear_stem_tuning_render_cancel_request(task_id)
    child_context = mp.get_context('spawn')
    progress_queue = child_context.Queue()
    child_process = child_context.Process(
        target=_stem_tuning_render_child_entry,
        args=(progress_queue, {
            'source_stem_manifest_path': source_stem_manifest_path,
            'output_dir': output_dir,
            'preset': preset,
            'render_id': render_id,
            'latest_pointer_path': latest_pointer_path,
            'source_audio_path': source_audio_path if isinstance(source_audio_path, str) and source_audio_path.strip() else None,
            'raw_reference_path': raw_reference_path if isinstance(raw_reference_path, str) and raw_reference_path.strip() else None,
            'dsp_device': str(os.getenv('ML_TORCH_DEVICE', 'cuda')),
        }),
        daemon=False,
    )
    child_process.start()

    state: dict[str, Any] = {
        'last_progress': 1,
        'last_message': 'Task picked up by ML worker',
        'result': None,
        'error_message': None,
        'error_traceback': '',
    }

    try:
        while True:
            try:
                event = progress_queue.get(timeout=0.25)
                if isinstance(event, dict):
                    _drain_stem_tuning_progress_event(event, report_progress, state)
                while True:
                    try:
                        pending_event = progress_queue.get_nowait()
                    except queue_module.Empty:
                        break
                    if isinstance(pending_event, dict):
                        _drain_stem_tuning_progress_event(pending_event, report_progress, state)
            except queue_module.Empty:
                pass

            if state['result'] is not None and child_process.exitcode is not None:
                result = state['result']
                break

            cancel_request = worker.get_stem_tuning_render_cancel_request(task_id) if task_id > 0 else None
            if isinstance(cancel_request, dict) and bool(cancel_request.get('cancel_requested')):
                reason = str(cancel_request.get('reason') or 'Stem tuning render cancelled')
                logger.info("Cancelling stem tuning render task_id=%s render_id=%s reason=%s", task_id, render_id, reason)
                report_progress(int(state['last_progress'] or 0), reason, {
                    'stage': 'cancel_requested',
                    'cancel_requested': True,
                })
                if child_process.is_alive():
                    child_process.terminate()
                    child_process.join(timeout=5)
                    if child_process.is_alive():
                        child_process.kill()
                        child_process.join(timeout=5)
                if task_id > 0:
                    worker.update_stem_tuning_render_status(
                        task_id,
                        'cancelled',
                        int(state['last_progress'] or 0),
                        reason,
                        render_id=render_id,
                        stage='cancelled',
                        cancel_requested=True,
                    )
                raise TaskCancelledError(reason)

            if child_process.exitcode is not None:
                if state['result'] is not None:
                    result = state['result']
                    break

                error_message = str(state['error_message'] or f'Stem tuning render process exited unexpectedly with code {child_process.exitcode}')
                if state['error_traceback']:
                    logger.error(
                        "Stem tuning render child traceback task_id=%s render_id=%s\n%s",
                        task_id,
                        render_id,
                        state['error_traceback'],
                    )
                raise RuntimeError(error_message)
    except Exception as exc:
        if task_id > 0:
            worker.update_stem_tuning_render_status(
                task_id,
                'cancelled' if isinstance(exc, TaskCancelledError) else 'failed',
                int(state['last_progress'] or 100),
                str(exc),
                level='info' if isinstance(exc, TaskCancelledError) else 'error',
                render_id=render_id,
                stage='cancelled' if isinstance(exc, TaskCancelledError) else 'failed',
            )
        raise
    finally:
        worker.clear_stem_tuning_render_cancel_request(task_id)
        if child_process.is_alive():
            child_process.terminate()
            child_process.join(timeout=5)
            if child_process.is_alive():
                child_process.kill()
                child_process.join(timeout=5)
        try:
            progress_queue.close()
        except Exception:
            pass

    logger.info(
        "Stem tuning render finish render_id=%s manifest_path=%s",
        render_id,
        result.get('manifest_path'),
    )
    if task_id > 0:
        worker.update_stem_tuning_render_status(
            task_id,
            'completed',
            100,
            'Container render completed',
            render_id=render_id,
            stage='completed',
            manifest_path=result.get('manifest_path'),
        )
    return result


def handle_stem_source_build(task_data: dict, worker: QueueWorker) -> dict:
    audio_path = str(task_data.get('audio_path') or '').strip()
    output_dir = str(task_data.get('output_dir') or '').strip()
    playback_manifest_path = str(task_data.get('playback_manifest_path') or '').strip()
    latest_pointer_path = str(task_data.get('latest_pointer_path') or '').strip()
    build_id = str(task_data.get('build_id') or task_data.get('task_id') or '').strip()
    model_name = str(task_data.get('model_name') or 'hybrid_htdemucs_ft_6s').strip()

    if audio_path == '' or output_dir == '' or playback_manifest_path == '' or latest_pointer_path == '':
        raise ValueError('Stem source build task is missing required paths')
    if build_id == '':
        raise ValueError('Stem source build task is missing build id')

    logger.info(
        "Stem source build start build_id=%s audio_path=%s output_dir=%s model=%s",
        build_id,
        audio_path,
        output_dir,
        model_name,
    )

    result = build_track_playback_stems(
        audio_path,
        output_dir,
        playback_manifest_path,
        build_id=build_id,
        latest_pointer_path=latest_pointer_path,
        model_name=model_name,
        device=str(os.getenv('ML_TORCH_DEVICE', 'cuda')),
    )

    logger.info(
        "Stem source build finish build_id=%s playback_manifest=%s",
        build_id,
        result.get('playback_manifest_path'),
    )
    return result


def handle_querysep_long_tail_build(task_data: dict, worker: QueueWorker) -> dict:
    audio_path = str(task_data.get('audio_path') or '').strip()
    output_dir = str(task_data.get('output_dir') or '').strip()
    build_id = str(task_data.get('build_id') or task_data.get('task_id') or '').strip()

    if audio_path == '' or output_dir == '':
        raise ValueError('Querysep long-tail build task is missing required paths')
    if build_id == '':
        raise ValueError('Querysep long-tail build task is missing build id')

    logger.info(
        "Querysep long-tail build start build_id=%s audio_path=%s output_dir=%s targets=%s",
        build_id,
        audio_path,
        output_dir,
        task_data.get('target_stems'),
    )

    result = build_querysep_long_tail_stems(
        audio_path=audio_path,
        output_dir=output_dir,
        build_id=build_id,
        target_stems=task_data.get('target_stems'),
        source_stem_manifest_path=task_data.get('source_stem_manifest_path'),
        build_from_paths=task_data.get('build_from_paths'),
        querysep_worker_url=task_data.get('querysep_worker_url'),
        querysep_input_path=task_data.get('querysep_input_path'),
        querysep_output_root=task_data.get('querysep_output_root'),
        path_mappings=task_data.get('path_mappings'),
        track_name=task_data.get('track_name'),
        phase_id=str(task_data.get('phase_id') or 'long_tail_querysep'),
        backend=str(task_data.get('backend') or 'banquet_byoq'),
        dry_run=task_data.get('dry_run', False),
        request_timeout_seconds=task_data.get('request_timeout_seconds'),
        min_output_bytes=task_data.get('min_output_bytes'),
        min_quality_duration_sec=task_data.get('min_quality_duration_sec'),
        max_duration_delta_sec=task_data.get('max_duration_delta_sec'),
        min_quality_rms_dbfs=task_data.get('min_quality_rms_dbfs'),
        min_source_rms_ratio_db=task_data.get('min_source_rms_ratio_db'),
        min_active_presence_windows=task_data.get('min_active_presence_windows'),
        min_active_presence_rms_dbfs=task_data.get('min_active_presence_rms_dbfs'),
        active_presence_window_sec=task_data.get('active_presence_window_sec'),
        max_proxy_correlation=task_data.get('max_proxy_correlation'),
        max_windowed_proxy_correlation=task_data.get('max_windowed_proxy_correlation'),
        windowed_proxy_correlation_sec=task_data.get('windowed_proxy_correlation_sec'),
        min_windowed_proxy_rms_dbfs=task_data.get('min_windowed_proxy_rms_dbfs'),
        max_possible_other_proxy_correlation=task_data.get('max_possible_other_proxy_correlation'),
        min_possible_other_proxy_rms_ratio_db=task_data.get('min_possible_other_proxy_rms_ratio_db'),
        long_tail_candidate_paths=task_data.get('long_tail_candidate_paths'),
    )

    logger.info(
        "Querysep long-tail build finish build_id=%s manifest_path=%s",
        build_id,
        result.get('manifest_path'),
    )
    return result


def handle_train_segment_model(task_data: dict, worker: QueueWorker) -> dict:
    """
    Handle segment preference model training task.
    
    Loads labeled segments from pattern_examples, extracts mel spectrograms,
    trains contrastive learning model.
    """
    import torch
    import torchaudio
    import torchaudio.transforms as T
    import librosa
    
    logger.info("Training segment preference model...")
    
    SAMPLE_RATE = 22050
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    
    storage_path = Path(os.getenv('STORAGE_PATH', '/app/storage'))
    models_path = storage_path / 'models'
    models_path.mkdir(parents=True, exist_ok=True)
    model_path = models_path / 'segment_preference_model.pt'
    
    device = resolve_torch_device(task_name='segment_preference_training', allow_cpu=False)
    logger.info("Using device for segment preference training: %s (visible_cuda=%s)", device, visible_cuda_summary())
    
    # Mel spectrogram transform
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        normalized=True
    ).to(device)
    
    # Load labeled segments from database
    with worker.db_conn.cursor() as cursor:
        cursor.execute("""
            SELECT 
                pe.id, pe.track_id, pe.start_time, pe.end_time, pe.is_positive,
                t.file_path, t.title, t.artist
            FROM pattern_examples pe
            JOIN tracks t ON pe.track_id = t.id
            WHERE t.file_path IS NOT NULL
            ORDER BY pe.is_positive DESC
        """)
        segments = cursor.fetchall()
    
    logger.info(f"Found {len(segments)} labeled segments")
    
    liked_count = sum(1 for s in segments if s[4])
    disliked_count = sum(1 for s in segments if not s[4])
    logger.info(f"  Liked: {liked_count}, Disliked: {disliked_count}")
    
    if liked_count < 5:
        return {
            'status': 'skipped',
            'message': f'Need at least 5 liked segments, got {liked_count}'
        }
    
    # Extract features
    liked_features = []
    disliked_features = []
    
    for seg_id, track_id, start, end, is_positive, file_path, title, artist in segments:
        audio_path = f"/app/storage/{file_path}"
        
        if not Path(audio_path).exists():
            continue
        
        try:
            # Load segment
            duration = end - start
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, offset=start, duration=duration)
            
            if len(y) < SAMPLE_RATE:
                continue
            
            # Extract mel on GPU
            waveform = torch.from_numpy(y).float().to(device)
            with torch.no_grad():
                mel = mel_transform(waveform)
                mel = torch.log(mel + 1e-9)
            mel_np = mel.cpu().numpy()
            
            if is_positive:
                liked_features.append(mel_np)
            else:
                disliked_features.append(mel_np)
                
        except Exception as e:
            logger.warning(f"Error extracting segment {seg_id}: {e}")
            continue
        
        # Clear GPU cache periodically
        if (len(liked_features) + len(disliked_features)) % 50 == 0:
            torch.cuda.empty_cache()
    
    logger.info(f"Extracted: {len(liked_features)} liked, {len(disliked_features)} disliked")
    
    if len(liked_features) < 5:
        return {
            'status': 'skipped',
            'message': f'Not enough valid liked segments: {len(liked_features)}'
        }
    
    # Add random negatives if not enough disliked
    if len(disliked_features) < len(liked_features) // 2:
        logger.info("Adding random negatives from non-liked tracks...")
        
        with worker.db_conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, file_path, duration
                FROM tracks 
                WHERE is_liked = false AND file_path IS NOT NULL AND duration > 30
                ORDER BY RANDOM()
                LIMIT 50
            """)
            random_tracks = cursor.fetchall()
        
        for track_id, file_path, duration in random_tracks:
            if len(disliked_features) >= len(liked_features):
                break
            
            audio_path = f"/app/storage/{file_path}"
            if not Path(audio_path).exists():
                continue
            
            try:
                start = np.random.uniform(10, max(11, duration - 20))
                seg_duration = np.random.uniform(5, 15)
                
                y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, offset=start, duration=seg_duration)
                
                if len(y) < SAMPLE_RATE:
                    continue
                
                waveform = torch.from_numpy(y).float().to(device)
                with torch.no_grad():
                    mel = mel_transform(waveform)
                    mel = torch.log(mel + 1e-9)
                disliked_features.append(mel.cpu().numpy())
                
            except Exception as e:
                continue
        
        logger.info(f"After adding randoms: {len(disliked_features)} disliked")
    
    # Train model
    model = SegmentPreferenceModel(
        input_channels=N_MELS,
        embedding_dim=256,
        learning_rate=0.001,
        temperature=0.07,
        device=str(resolve_torch_device(task_name='segment_preference_training.model', allow_cpu=False))
    )
    
    logger.info(f"Training: {len(liked_features)} liked vs {len(disliked_features)} disliked")
    
    history = model.train(
        liked_features=liked_features,
        disliked_features=disliked_features,
        epochs=100,
        batch_size=32
    )
    
    # Compute liked centroid
    model.compute_liked_centroid(liked_features)
    
    # Save model
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Clean up
    torch.cuda.empty_cache()
    
    return {
        'status': 'trained',
        'model_path': str(model_path),
        'liked_count': len(liked_features),
        'disliked_count': len(disliked_features),
        'final_loss': history['loss'][-1] if history['loss'] else 0,
        'final_accuracy': history['accuracy'][-1] if history.get('accuracy') else 0
    }


def handle_analyze_segments(
    task_data: dict,
    worker: QueueWorker,
    analyzer,
    structure_analyzer,
) -> dict:
    """
    Handle segment analysis for a single track.
    Detects segments using novelty detection and classifies them by type.
    Creates patterns and pattern_examples in database.
    Updates status in Redis for frontend tracking.
    """
    import torch
    import torchaudio.transforms as T
    from scipy import signal
    from scipy.ndimage import uniform_filter1d
    import redis
    
    track_id = task_data['track_id']
    
    logger.info(f"Analyzing segments for track {track_id}")
    
    # Redis connection for status updates
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'redis'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD') or None,
        decode_responses=True
    )
    # PHP Laravel adds prefix automatically when reading, so we must add it when writing
    # to match the key that PHP expects
    redis_prefix = os.getenv('REDIS_PREFIX', 'ai_music_recommendations_database_')
    status_key = f"{redis_prefix}segment_analysis_status:{track_id}"
    
    def update_status(status: str, progress: int, message: str, **extra):
        """Update analysis status in Redis."""
        try:
            data = {
                'status': status,
                'progress': progress,
                'message': message,
                'track_id': track_id,
                'updated_at': datetime.now().isoformat(),
                **extra
            }
            redis_client.setex(status_key, 300, json.dumps(data))
            logger.info(f"Track {track_id}: {message} ({progress}%)")
        except Exception as e:
            logger.error(f"Failed to update status in Redis: {e}")
    
    update_status('queued', 0, 'Waiting for segment analysis slot...')

    with _segment_analysis_task_lock(track_id):
        analyzer = analyzer() if callable(analyzer) else analyzer
        structure_analyzer = structure_analyzer() if callable(structure_analyzer) else structure_analyzer

        update_status('processing', 5, 'Loading track info...')
        
        # Get track info
        with worker.db_conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, title, artist, file_path, duration
                FROM tracks WHERE id = %s
            """, (track_id,))
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Track {track_id} not found")
            
            _, title, artist, file_path, duration = row
        
        audio_path = f"/app/storage/{file_path}"
        
        if not Path(audio_path).exists():
            update_status('failed', 0, f'Audio file not found: {file_path}')
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        update_status('processing', 10, 'Loading audio file...')
        
        # Load audio
        SAMPLE_RATE = 22050
        HOP_LENGTH = 512
        N_FFT = 2048
        N_MELS = 128
        
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        track_duration = len(y) / sr
        
        if track_duration < 30:
            update_status('completed', 100, 'Track too short (< 30s)', segments_created=0)
            return {'status': 'skipped', 'message': 'Track too short (< 30s)'}
        if track_duration > MAX_SEGMENT_ANALYSIS_DURATION:
            update_status(
                'completed',
                100,
                f'Track too long for segment analysis (> {int(MAX_SEGMENT_ANALYSIS_DURATION)}s)',
                segments_created=0,
            )
            return {
                'status': 'skipped',
                'message': f'Track too long for segment analysis (> {int(MAX_SEGMENT_ANALYSIS_DURATION)}s)',
            }

        force_structure_guided = bool(task_data.get('force_structure_guided'))
        structure_guided_segmentation_enabled = (
            force_structure_guided
            or os.getenv('ENABLE_STRUCTURE_GUIDED_SEGMENTATION', '1') == '1'
        )

        update_status(
            'processing',
            15,
            'Analyzing musical structure...' if structure_guided_segmentation_enabled else 'Skipping structure analysis...',
        )

        structure_result = None
        structure_intervals = []
        beat_grid = np.array([], dtype=np.float32)
        downbeat_grid = np.array([], dtype=np.float32)

        def snap_to_grid(value: float, max_shift: float = 0.8) -> float:
            for grid in (downbeat_grid, beat_grid):
                if grid.size == 0:
                    continue
                idx = int(np.argmin(np.abs(grid - value)))
                snapped = float(grid[idx])
                if abs(snapped - value) <= max_shift:
                    return snapped
            return float(value)

        if structure_guided_segmentation_enabled:
            try:
                structure_result = _load_or_analyze_structure(track_id, audio_path, structure_analyzer)
                beat_grid = np.array(sorted(set(structure_result.beats)), dtype=np.float32)
                downbeat_grid = np.array(sorted(set(structure_result.downbeats)), dtype=np.float32)

                if beat_grid.size == 0 or downbeat_grid.size == 0:
                    fallback_beats, fallback_downbeats, fallback_tempo = _estimate_fallback_beat_grids(
                        track_id=track_id,
                        audio_signal=y,
                        sample_rate=sr,
                        hop_length=HOP_LENGTH,
                        track_duration=track_duration,
                        bpm_hint=structure_result.bpm,
                    )
                    if beat_grid.size == 0 and fallback_beats.size > 0:
                        beat_grid = fallback_beats
                        structure_result.beats = [float(x) for x in fallback_beats]
                    if downbeat_grid.size == 0 and fallback_downbeats.size > 0:
                        downbeat_grid = fallback_downbeats
                        structure_result.downbeats = [float(x) for x in fallback_downbeats]
                    if fallback_tempo is not None and float(getattr(structure_result, 'bpm', 0.0) or 0.0) <= 0.0:
                        structure_result.bpm = float(fallback_tempo)

                for idx, seg in enumerate(structure_result.segments):
                    start = 0.0 if idx == 0 else snap_to_grid(float(seg.start), max_shift=1.0)
                    end = track_duration if idx == len(structure_result.segments) - 1 else snap_to_grid(float(seg.end), max_shift=1.0)
                    end = min(track_duration, max(start, end))
                    if end - start < 2.0:
                        continue
                    structure_intervals.append({
                        'start': start,
                        'end': end,
                        'label': seg.label,
                    })

                if structure_intervals:
                    structure_intervals[0]['start'] = 0.0
                    structure_intervals[-1]['end'] = track_duration

                logger.info(
                    f"Track {track_id}: structure-guided segmentation ready "
                    f"({len(structure_intervals)} intervals, beats={len(structure_result.beats)})"
                )
            except Exception as e:
                logger.warning(f"Track {track_id}: structure-guided segmentation unavailable, using novelty fallback: {e}")
                structure_result = None
                structure_intervals = []
        else:
            logger.info(
                "Track %s: structure-guided segmentation disabled by ENABLE_STRUCTURE_GUIDED_SEGMENTATION=0; using novelty fallback",
                track_id,
            )

        update_status('processing', 20, 'Setting up GPU...')
        
        # Setup GPU
        device = resolve_torch_device(task_name='segment_analysis', allow_cpu=False)
        logger.info("Track %s: segment analysis device=%s visible_cuda=%s", track_id, device, visible_cuda_summary())
        mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        ).to(device)
        
        spec_transform = T.Spectrogram(
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        ).to(device)
        
        # Extract features on GPU
        waveform = torch.from_numpy(y).float().to(device).unsqueeze(0)
        
        update_status('processing', 30, 'Extracting audio features...')
        
        with torch.no_grad():
            mel = mel_transform(waveform)
            spec = spec_transform(waveform)
        
        # RMS energy
        rms = torch.sqrt(torch.mean(mel ** 2, dim=1)).squeeze().cpu().numpy()
        
        # Spectral centroid
        freqs = torch.linspace(0, SAMPLE_RATE/2, spec.shape[1], device=device)
        spec_sum = spec.sum(dim=1, keepdim=True) + 1e-10
        centroid = (spec * freqs.view(1, -1, 1)).sum(dim=1) / spec_sum.squeeze(1)
        centroid = centroid.squeeze().cpu().numpy()
        
        # Spectral contrast
        n_bands = 6
        band_size = spec.shape[1] // n_bands
        contrast_list = []
        for i in range(n_bands):
            band = spec[:, i*band_size:(i+1)*band_size, :]
            band_max = band.max(dim=1)[0]
            band_min = band.min(dim=1)[0] + 1e-10
            contrast_list.append(torch.log(band_max / band_min))
        contrast = torch.stack(contrast_list, dim=1).mean(dim=1).squeeze().cpu().numpy()
    
        # Normalize features
        def norm(x):
            if len(x) == 0 or np.all(np.isnan(x)):
                return np.zeros_like(x)
            x = np.nan_to_num(x, nan=0.0)
            xmin, xmax = x.min(), x.max()
            if xmax - xmin < 1e-10:
                return np.zeros_like(x)
            return (x - xmin) / (xmax - xmin)
        
        rms_norm = norm(rms)
        centroid_norm = norm(centroid)
        contrast_norm = norm(contrast)
        
        update_status('processing', 50, 'Detecting segment boundaries...')
        
        # Compute novelty
        n_frames = len(rms)
        novelty = np.zeros(n_frames)
        novelty += uniform_filter1d(np.abs(np.diff(rms_norm, prepend=rms_norm[0])), 10) * 0.4
        novelty += uniform_filter1d(np.abs(np.diff(centroid_norm, prepend=centroid_norm[0])), 10) * 0.3
        novelty += uniform_filter1d(np.abs(np.diff(contrast_norm, prepend=contrast_norm[0])), 10) * 0.3
        novelty = uniform_filter1d(novelty, 20)

        min_distance = int(3 * SAMPLE_RATE / HOP_LENGTH)
        threshold = np.percentile(novelty, 65)
        novelty_peaks, _ = signal.find_peaks(novelty, height=threshold, distance=min_distance)
        novelty_peak_times = (
            novelty_peaks.astype(np.float32) * HOP_LENGTH / SAMPLE_RATE
            if len(novelty_peaks)
            else np.array([], dtype=np.float32)
        )
        
        rms_pct = np.percentile(rms, [25, 50, 75])
        centroid_pct = np.percentile(centroid, [25, 50, 75])
        contrast_pct = np.percentile(contrast, [25, 50, 75])
        intro_window_end = min(track_duration * 0.18, 28.0)
        outro_window_start = max(track_duration * 0.82, track_duration - 28.0)

        def classify_segment_type(
            seg_rms: float,
            seg_centroid: float,
            seg_contrast: float,
            start: float,
            end: float,
            label_hint: str | None = None,
        ) -> tuple[str, float]:
            hint = (label_hint or '').strip().lower()
            rms_level = 'high' if seg_rms > rms_pct[2] else ('mid' if seg_rms > rms_pct[1] else 'low')
            cent_level = 'high' if seg_centroid > centroid_pct[2] else ('mid' if seg_centroid > centroid_pct[1] else 'low')
            contrast_level = 'high' if seg_contrast > contrast_pct[2] else ('mid' if seg_contrast > contrast_pct[1] else 'low')
            is_intro_region = end <= intro_window_end
            is_outro_region = start >= outro_window_start

            if hint in ('chorus', 'drop', 'hook'):
                if rms_level == 'high':
                    return ('drop_high' if cent_level == 'high' else 'drop_bass', 3.0)
                return ('melodic', 2.5)
            if hint in ('prechorus', 'pre-chorus', 'buildup', 'build', 'build_up'):
                return ('buildup', 2.3 if rms_level != 'low' else 1.8)
            if hint in ('bridge', 'break', 'inst', 'instrumental'):
                return ('break' if cent_level == 'high' else 'ambient', 1.5 if cent_level == 'high' else 1.0)
            if hint == 'verse':
                if rms_level == 'high':
                    return ('melodic' if cent_level == 'high' else 'verse', 2.2 if cent_level == 'high' else 1.8)
                return ('melodic' if cent_level == 'high' else 'verse', 2.0 if cent_level == 'high' else 1.5)
            if hint == 'intro':
                if is_intro_region:
                    return ('intro', 1.0)
                if is_outro_region:
                    return ('outro', 1.0)
                if rms_level == 'high':
                    return ('buildup' if contrast_level != 'low' else 'verse', 2.3 if contrast_level == 'high' else 1.8)
                if cent_level == 'high':
                    return ('melodic', 1.9)
                return ('break' if contrast_level != 'low' else 'ambient', 1.2)
            if hint == 'outro':
                if is_outro_region:
                    return ('outro', 1.0)
                if is_intro_region:
                    return ('intro', 1.0)
                if rms_level == 'high':
                    return ('buildup' if contrast_level == 'high' else 'verse', 2.0 if contrast_level == 'high' else 1.6)
                return ('break' if contrast_level != 'low' else 'ambient', 1.1)

            if rms_level == 'high':
                return ('drop_high' if cent_level == 'high' else 'drop_bass', 3.0)
            if rms_level == 'mid':
                return ('melodic' if cent_level == 'high' else 'verse', 2.0 if cent_level == 'high' else 1.5)
            return ('break' if cent_level == 'high' else 'ambient', 1.0)

        segments = []
        if structure_intervals:
            def refine_boundary(target: float, lower: float, upper: float, max_shift: float = 2.5) -> float:
                refined = float(np.clip(target, lower, upper))
                if novelty_peak_times.size > 0:
                    window_start = max(lower, refined - max_shift)
                    window_end = min(upper, refined + max_shift)
                    local_peaks = novelty_peak_times[
                        (novelty_peak_times >= window_start) & (novelty_peak_times <= window_end)
                    ]
                    if local_peaks.size > 0:
                        local_frames = np.clip((local_peaks * SAMPLE_RATE / HOP_LENGTH).astype(int), 0, len(novelty) - 1)
                        refined = float(local_peaks[int(np.argmax(novelty[local_frames]))])

                return float(np.clip(snap_to_grid(refined, max_shift=0.9), lower, upper))

            raw_boundaries = [0.0] + [float(interval['end']) for interval in structure_intervals[:-1]] + [track_duration]
            refined_boundaries = [0.0]
            for boundary in raw_boundaries[1:-1]:
                lower = refined_boundaries[-1] + 2.0
                upper = max(lower, track_duration - 2.0)
                refined_boundaries.append(refine_boundary(boundary, lower, upper))
            refined_boundaries.append(track_duration)

            for i in range(1, len(refined_boundaries) - 1):
                refined_boundaries[i] = max(refined_boundaries[i], refined_boundaries[i - 1] + 2.0)
            for i in range(len(refined_boundaries) - 2, 0, -1):
                refined_boundaries[i] = min(refined_boundaries[i], refined_boundaries[i + 1] - 2.0)

            intervals_to_classify = []
            for idx, interval in enumerate(structure_intervals):
                start = float(refined_boundaries[idx])
                end = float(refined_boundaries[idx + 1])
                if end - start < 2.0:
                    continue
                intervals_to_classify.append({
                    'start': start,
                    'end': end,
                    'label': interval.get('label'),
                })

            update_status('processing', 60, f'Classifying {len(intervals_to_classify)} refined structure segments...')
        else:
            boundaries = [0.0]
            for peak in novelty_peaks:
                boundaries.append(float(peak * HOP_LENGTH / SAMPLE_RATE))
            boundaries.append(track_duration)
            boundaries = sorted(set(boundaries))

            update_status('processing', 60, f'Classifying {len(boundaries)-1} novelty segments...')
            intervals_to_classify = [
                {'start': boundaries[i], 'end': boundaries[i + 1], 'label': None}
                for i in range(len(boundaries) - 1)
            ]

        for interval in intervals_to_classify:
            start = float(interval['start'])
            end = float(interval['end'])
            seg_dur = end - start

            if seg_dur < 3:
                continue

            sf = int(start * SAMPLE_RATE / HOP_LENGTH)
            ef = min(int(end * SAMPLE_RATE / HOP_LENGTH), len(rms))

            if sf >= ef:
                continue

            seg_rms = float(np.mean(rms[sf:ef]))
            seg_centroid = float(np.mean(centroid[sf:ef]))
            seg_contrast = float(np.mean(contrast[sf:ef]))
            seg_type, importance = classify_segment_type(
                seg_rms,
                seg_centroid,
                seg_contrast,
                start,
                end,
                interval.get('label'),
            )

            segments.append({
                'start': start,
                'end': end,
                'type': seg_type,
                'importance': importance,
                'label_hint': interval.get('label'),
            })
        
        # Merge adjacent segments of the same type
        original_count = len(segments)
        if len(segments) > 1:
            merged_segments = [segments[0]]
            for seg in segments[1:]:
                prev = merged_segments[-1]
                # Merge if same type and gap is small (< 1 second)
                if seg['type'] == prev['type'] and (seg['start'] - prev['end']) < 1.0:
                    # Extend previous segment
                    prev['end'] = seg['end']
                    prev['importance'] = max(prev['importance'], seg['importance'])
                else:
                    merged_segments.append(seg)
            segments = merged_segments
            logger.info(f"Merged segments: {len(segments)} (from {original_count} original)")

        melodic_refinement_enabled = os.getenv('ENABLE_MELODIC_SEGMENT_REFINEMENT', '1') == '1'
        if melodic_refinement_enabled and len(segments) > 1:
            update_status('processing', 65, 'Refining melodic segment boundaries...')
            refined_segments, refinement_summary = refine_segments_with_melodic_continuity(
                int(track_id),
                audio_path,
                segments,
                separator_model=os.getenv('SEGMENT_MELODIC_REFINEMENT_SEPARATOR', 'htdemucs_6s'),
                device=device.type,
            )
            if refinement_summary.get('status') == 'ok':
                logger.info(
                    "Track %s: melodic refinement merged %s boundaries (%s -> %s segments, other_events=%s, bass_events=%s)",
                    track_id,
                    refinement_summary.get('merges_applied', 0),
                    refinement_summary.get('original_segments', len(segments)),
                    refinement_summary.get('refined_segments', len(refined_segments)),
                    refinement_summary.get('other_event_count', 0),
                    refinement_summary.get('bass_event_count', 0),
                )
                segments = refined_segments
            else:
                logger.warning(
                    "Track %s: melodic refinement skipped (%s)",
                    track_id,
                    refinement_summary.get('reason') or refinement_summary.get('status'),
                )

        update_status('processing', 70, f'Saving {len(segments)} segments to database...')

        # Create patterns for each segment
        STORAGE_PATH = Path(os.getenv('STORAGE_PATH', '/app/storage'))
        PATTERNS_PATH = STORAGE_PATH / 'patterns'
        PATTERNS_PATH.mkdir(parents=True, exist_ok=True)
        SEGMENT_EMBEDDINGS_PATH = STORAGE_PATH / 'embeddings'
        SEGMENT_EMBEDDINGS_PATH.mkdir(parents=True, exist_ok=True)
        
        patterns_created = 0
        embedding_jobs = []
        created_examples: list[dict[str, Any]] = []
        rating_remap_summary: dict[str, Any] = {
            'status': 'skipped',
            'reason': 'no_prior_ratings',
            'old_examples_considered': 0,
            'user_updates_applied': 0,
            'legacy_updates_applied': 0,
        }

        previous_autocommit = worker.db_conn.autocommit
        worker.db_conn.autocommit = False
        try:
            old_auto_examples = _load_existing_auto_examples_for_track(worker, int(track_id))

            with worker.db_conn.cursor() as cursor:
                cursor.execute("""
                    DELETE FROM pattern_matches WHERE pattern_id IN (
                        SELECT p.id FROM patterns p
                        JOIN pattern_examples pe ON p.id = pe.pattern_id
                        WHERE pe.track_id = %s AND p.name LIKE '%%[auto%%'
                    )
                """, (track_id,))
                cursor.execute("""
                    DELETE FROM pattern_examples WHERE track_id = %s AND pattern_id IN (
                        SELECT id FROM patterns WHERE name LIKE '%%[auto%%'
                    )
                """, (track_id,))

            for seg in segments:
                if seg['importance'] < 1.0:
                    continue

                start_sample = int(seg['start'] * SAMPLE_RATE)
                end_sample = int(seg['end'] * SAMPLE_RATE)
                seg_audio = y[start_sample:end_sample]

                if len(seg_audio) < SAMPLE_RATE:
                    continue

                seg_tensor = torch.from_numpy(seg_audio).float().to(device)
                with torch.no_grad():
                    mel_seg = mel_transform(seg_tensor.unsqueeze(0))
                    mel_np = mel_seg.squeeze().cpu().numpy()

                    n_chroma = 12
                    chroma = np.zeros((n_chroma, mel_np.shape[1]))
                    bins_per_chroma = N_MELS // n_chroma
                    for c in range(n_chroma):
                        chroma[c] = np.mean(mel_np[c*bins_per_chroma:(c+1)*bins_per_chroma], axis=0)

                    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
                    norms[norms < 1e-10] = 1.0
                    chroma = (chroma / norms).astype(np.float32)

                if chroma.shape[1] < 20:
                    continue

                time_str = f"{int(seg['start'] // 60)}:{int(seg['start'] % 60):02d}"
                pattern_name = f"[auto:{seg['type']}] {artist or '?'} - {title[:25]} ({time_str})"

                with worker.db_conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO patterns (name, representation_type, transpose_invariant, tempo_invariant,
                                              match_threshold, importance, created_at, updated_at)
                        VALUES (%s, 'chroma', true, false, 0.70, %s, NOW(), NOW())
                        RETURNING id
                    """, (pattern_name, seg['importance']))

                    pattern_id = cursor.fetchone()[0]

                    np.save(PATTERNS_PATH / f"pattern_{pattern_id}_chroma.npy", chroma)

                    cursor.execute("""
                        INSERT INTO pattern_examples (pattern_id, track_id, start_time, end_time,
                                                      is_positive, notes, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, NULL, %s, NOW(), NOW())
                        RETURNING id
                    """, (
                        pattern_id,
                        track_id,
                        seg['start'],
                        seg['end'],
                        f"type={seg['type']};hint={(seg.get('label_hint') or 'none')}",
                    ))
                    example_id = cursor.fetchone()[0]

                if example_id is not None:
                    created_examples.append({
                        'example_id': int(example_id),
                        'start_time': float(seg['start']),
                        'end_time': float(seg['end']),
                    })
                    embedding_jobs.append({
                        'example_id': int(example_id),
                        'start_time': float(seg['start']),
                        'end_time': float(seg['end']),
                    })

                patterns_created += 1

            if old_auto_examples and created_examples:
                remap_payload = build_rating_safe_segment_remap(old_auto_examples, created_examples)
                remap_acceptance = evaluate_rating_safe_remap_acceptance(remap_payload)
                if not remap_acceptance.get('accepted', False):
                    raise RuntimeError(
                        f"rating-safe remap rejected for track {track_id}: "
                        f"{','.join(remap_acceptance.get('reasons') or ['unknown'])}"
                    )
                _apply_segment_rating_remap(worker, remap_payload)
                rating_remap_summary = {
                    'status': 'ok',
                    **remap_payload.get('summary', {}),
                    'acceptance': remap_acceptance,
                }
                logger.info(
                    "Track %s: rating-safe remap old_examples=%s user_updates=%s legacy_updates=%s unmatched=%s",
                    track_id,
                    rating_remap_summary.get('old_examples_considered', 0),
                    rating_remap_summary.get('user_updates_applied', 0),
                    rating_remap_summary.get('legacy_updates_applied', 0),
                    rating_remap_summary.get('old_examples_unmatched', 0),
                )
            else:
                rating_remap_summary = {
                    'status': 'skipped',
                    'reason': 'no_prior_ratings',
                    'old_examples_considered': len(old_auto_examples),
                    'user_updates_applied': 0,
                    'legacy_updates_applied': 0,
                    'acceptance': {
                        'accepted': True,
                        'reasons': [],
                        'user_rated_examples_total': 0,
                        'user_rated_examples_remapped': 0,
                        'legacy_rated_examples_total': 0,
                        'legacy_rated_examples_remapped': 0,
                        'user_updates_skipped': 0,
                        'legacy_updates_skipped': 0,
                    },
                }

            worker.db_conn.commit()
        except Exception:
            worker.db_conn.rollback()
            raise
        finally:
            worker.db_conn.autocommit = previous_autocommit

        # Extract and persist segment embeddings in one batched pass so we avoid
        # reloading/resampling the same track once per segment.
        if embedding_jobs:
            try:
                segment_embeddings = analyzer.extract_embeddings_batch(
                    audio_path,
                    [(job['start_time'], job['end_time']) for job in embedding_jobs],
                    preloaded_audio=y,
                    preloaded_sample_rate=SAMPLE_RATE,
                )
            except Exception as e:
                logger.warning(f"Track {track_id}: failed to extract segment embedding batch: {e}")
                segment_embeddings = [None] * len(embedding_jobs)

            for job, seg_embedding in zip(embedding_jobs, segment_embeddings):
                if seg_embedding is None:
                    logger.warning(
                        f"Track {track_id}: failed to extract embedding for segment example {job['example_id']}"
                    )
                    continue

                seg_embedding_path = SEGMENT_EMBEDDINGS_PATH / f"segment_{job['example_id']}.npy"
                np.save(str(seg_embedding_path), seg_embedding)
                with worker.db_conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE pattern_examples SET embedding_path = %s WHERE id = %s",
                        (str(seg_embedding_path), int(job['example_id']))
                    )
        
        # Clean up GPU
        torch.cuda.empty_cache()
        
        update_status('completed', 100, f'Created {patterns_created} segments', segments_created=patterns_created)
        logger.info(f"Track {track_id}: created {patterns_created} segments")
        
        return {
            'status': 'completed',
            'track_id': track_id,
            'segments_created': patterns_created,
            'total_segments': len(segments),
            'rating_remap': rating_remap_summary,
        }


def handle_backfill_pattern_example_embeddings(task_data: dict, worker: QueueWorker, analyzer: AudioAnalyzer) -> dict:
    """Backfill missing pattern_examples.embedding_path without re-segmenting tracks."""
    pattern_example_ids = task_data.get('pattern_example_ids') or []
    try:
        pattern_example_ids = [int(x) for x in pattern_example_ids]
    except Exception:
        pattern_example_ids = []

    if not pattern_example_ids:
        return {'status': 'skipped', 'message': 'No pattern_example_ids provided'}

    storage_path = Path(os.getenv('STORAGE_PATH', '/app/storage'))
    embeddings_dir = storage_path / 'embeddings'
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with worker.db_conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT pe.id, pe.track_id, pe.start_time, pe.end_time, t.file_path
            FROM pattern_examples pe
            JOIN tracks t ON t.id = pe.track_id
            WHERE pe.id = ANY(%s)
              AND pe.embedding_path IS NULL
              AND pe.start_time IS NOT NULL
              AND pe.end_time IS NOT NULL
              AND t.file_path IS NOT NULL
            """,
            (pattern_example_ids,)
        )
        rows = cursor.fetchall()

    updated = 0
    failed = 0
    rows_by_audio = {}
    for ex_id, track_id, start_time, end_time, file_path in rows:
        audio_path = f"/app/storage/{file_path}"
        rows_by_audio.setdefault(audio_path, []).append({
            'example_id': int(ex_id),
            'track_id': int(track_id),
            'start_time': float(start_time),
            'end_time': float(end_time),
        })

    for audio_path, audio_jobs in rows_by_audio.items():
        if not Path(audio_path).exists():
            failed += len(audio_jobs)
            continue

        try:
            embeddings = analyzer.extract_embeddings_batch(
                audio_path,
                [(job['start_time'], job['end_time']) for job in audio_jobs],
            )
        except Exception as e:
            logger.warning(f"PatternExample backfill batch failed for {audio_path}: {e}")
            failed += len(audio_jobs)
            continue

        for job, emb in zip(audio_jobs, embeddings):
            if emb is None:
                logger.warning(f"PatternExample {job['example_id']}: failed to backfill embedding")
                failed += 1
                continue

            out_path = embeddings_dir / f"segment_{job['example_id']}.npy"
            np.save(str(out_path), emb)
            with worker.db_conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE pattern_examples SET embedding_path = %s WHERE id = %s",
                    (str(out_path), int(job['example_id']))
                )
            updated += 1

    return {
        'status': 'completed',
        'requested': int(len(pattern_example_ids)),
        'rows_found': int(len(rows)),
        'updated': int(updated),
        'failed': int(failed),
    }


def main():
    """Main entry point."""
    logger.info("Starting ML Worker...")
    
    worker = QueueWorker()
    structure_model_name = os.getenv('STRUCTURE_ANALYZER_MODEL_NAME', 'harmonix-all')

    startup_lock_client = _create_segment_analysis_lock_redis_client()
    try:
        _cleanup_stale_segment_analysis_lock(startup_lock_client, reason='worker-startup')
    finally:
        try:
            startup_lock_client.close()
        except Exception:
            pass

    analyzer = None
    dj_analyzer = None
    structure_analyzer = None
    segment_index = None

    def get_audio_analyzer() -> AudioAnalyzer:
        nonlocal analyzer
        if analyzer is None:
            logger.info("Lazily initializing AudioAnalyzer")
            analyzer = AudioAnalyzer()
        return analyzer

    def get_dj_analyzer() -> AIDJAnalyzer:
        nonlocal dj_analyzer
        if dj_analyzer is None:
            logger.info("Lazily initializing AIDJAnalyzer")
            dj_analyzer = AIDJAnalyzer()
        return dj_analyzer

    def get_structure_analyzer() -> StructureAnalyzer:
        nonlocal structure_analyzer
        if structure_analyzer is None:
            logger.info(
                "Lazily initializing All-In-One structure analyzer: model=%s",
                structure_model_name,
            )
            structure_analyzer = StructureAnalyzer(model_name=structure_model_name)
        return structure_analyzer

    def get_segment_index() -> SegmentIndexManager:
        nonlocal segment_index
        if segment_index is None:
            logger.info("Lazily initializing SegmentIndexManager")
            segment_index = SegmentIndexManager()
        return segment_index
    
    # Register handlers
    worker.register_handler(
        'track_analysis',
        lambda task: handle_track_analysis(task, worker, get_audio_analyzer(), get_structure_analyzer())
    )
    worker.register_batch_handler(
        'track_analysis',
        lambda tasks: handle_track_analysis_batch(tasks, worker, get_audio_analyzer(), get_structure_analyzer())
    )
    worker.register_handler(
        'fragment_analysis',
        lambda task: handle_fragment_analysis(task, worker, get_audio_analyzer(), get_segment_index())
    )
    worker.register_handler(
        'model_training',
        lambda task: handle_model_training(task, worker)
    )
    worker.register_handler(
        'dj_analysis',
        lambda task: handle_dj_analysis(task, worker, get_dj_analyzer())
    )
    worker.register_handler(
        'segment_search',
        lambda task: handle_segment_search(task, worker, get_audio_analyzer(), get_segment_index())
    )
    worker.register_handler(
        'embedding_recommendations',
        lambda task: handle_embedding_recommendations(task, worker)
    )
    worker.register_handler(
        'stem_source_build',
        lambda task: handle_stem_source_build(task, worker)
    )
    worker.register_handler(
        'querysep_long_tail_build',
        lambda task: handle_querysep_long_tail_build(task, worker)
    )
    worker.register_handler(
        'stem_tuning_render',
        lambda task: handle_stem_tuning_render(task, worker)
    )
    worker.register_handler(
        'train_segment_model',
        lambda task: handle_train_segment_model(task, worker)
    )
    worker.register_handler(
        'analyze_segments',
        lambda task: handle_analyze_segments(task, worker, get_audio_analyzer, get_structure_analyzer)
    )
    worker.register_handler(
        'backfill_pattern_example_embeddings',
        lambda task: handle_backfill_pattern_example_embeddings(task, worker, get_audio_analyzer())
    )
    
    # Handle shutdown signals
    def shutdown(signum, frame):
        logger.info("Received shutdown signal")
        worker.stop()
        
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    
    # Start worker
    try:
        worker.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        logger.info("ML Worker stopped")


if __name__ == '__main__':
    main()
