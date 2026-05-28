"""
QueueWorker - Redis queue processor for ML tasks.

Requirements: 13.1, 13.2, 13.3, 13.4
"""

import json
import os
import time
import logging
import signal
import socket
import threading
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime
from contextlib import contextmanager

import redis
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class TaskTimeoutError(Exception):
    """Raised when a task exceeds its timeout."""
    pass


class TaskCancelledError(Exception):
    """Raised when a task is cancelled or superseded intentionally."""
    pass


@contextmanager
def task_timeout(seconds: int, task_info: str = ""):
    """Context manager for task timeout using threading (works on all platforms)."""
    result = {'timed_out': False}
    
    def timeout_handler():
        result['timed_out'] = True
    
    timer = threading.Timer(seconds, timeout_handler)
    timer.start()
    
    try:
        yield result
    finally:
        timer.cancel()
        if result['timed_out']:
            raise TaskTimeoutError(f"Task timed out after {seconds}s: {task_info}")


class QueueWorker:
    """
    Processes ML tasks from Redis queue.
    
    Handles task types:
    - track_analysis: Full audio analysis
    - fragment_analysis: Fragment embedding extraction
    - model_training: Preference model training
    - recommendation: Generate recommendations
    
    Features:
    - Task timeout protection (default 120s per task)
    - GPU memory cleanup between tasks
    - Automatic retry/skip for stuck tasks
    """
    
    # Redis key prefix must match Laravel's config (APP_NAME slugified + _database_)
    REDIS_PREFIX = os.getenv('REDIS_PREFIX', 'ai_music_recommendations_database_')
    QUEUE_NAME = f'{REDIS_PREFIX}ml_worker:tasks'
    RESULT_QUEUE = f'{REDIS_PREFIX}ml_worker:results'
    
    # Timeout settings (in seconds)
    TASK_TIMEOUT = int(os.getenv('TASK_TIMEOUT', 900))  # 15 minutes per task (training 300 epochs + scoring ~10k tracks)
    ANALYSIS_TIMEOUT = int(os.getenv('ANALYSIS_TIMEOUT', 60))  # 1 minute for analysis
    TRACK_ANALYSIS_BATCH_SIZE = max(1, int(os.getenv('TRACK_ANALYSIS_BATCH_SIZE', 1)))
    TRACK_ANALYSIS_BATCH_WAIT_MS = max(0, int(os.getenv('TRACK_ANALYSIS_BATCH_WAIT_MS', 0)))
    ANALYZE_SEGMENTS_HARD_TIMEOUT_REQUEUE_LIMIT = max(
        0,
        int(os.getenv('ANALYZE_SEGMENTS_HARD_TIMEOUT_REQUEUE_LIMIT', '2'))
    )
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.db_conn: Optional[psycopg2.extensions.connection] = None
        self.handlers: Dict[str, Callable] = {}
        self.batch_handlers: Dict[str, Callable] = {}
        self.running = False
        self.current_task_id: Optional[int] = None
        self.task_start_time: Optional[float] = None
        
    def connect_redis(self) -> None:
        """Connect to Redis server."""
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            password=os.getenv('REDIS_PASSWORD') or None,
            decode_responses=True
        )
        self.redis_client.ping()
        logger.info("Connected to Redis")

    @staticmethod
    def _new_redis_client() -> redis.Redis:
        return redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            password=os.getenv('REDIS_PASSWORD') or None,
            decode_responses=True
        )
        
    def connect_database(self) -> None:
        """Connect to PostgreSQL database."""
        self.db_conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'postgres'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'music_recommendations'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'secret')
        )
        self.db_conn.autocommit = True
        logger.info("Connected to PostgreSQL")
        
    def register_handler(self, task_type: str, handler: Callable) -> None:
        """Register a handler for a task type."""
        self.handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")

    def register_batch_handler(self, task_type: str, handler: Callable) -> None:
        """Register a batch handler for a task type."""
        self.batch_handlers[task_type] = handler
        logger.info(f"Registered batch handler for task type: {task_type}")
        
    def update_task_status(
        self, 
        task_id: int, 
        status: str, 
        error_message: Optional[str] = None
    ) -> None:
        """Update task status in database."""
        # Skip DB update for non-numeric task IDs (e.g., chat_search_xxx)
        if not isinstance(task_id, int) and not (isinstance(task_id, str) and task_id.isdigit()):
            logger.debug(f"Skipping DB status update for non-DB task: {task_id}")
            return
        
        with self.db_conn.cursor() as cursor:
            if status in ('completed', 'failed', 'cancelled'):
                cursor.execute(
                    """
                    UPDATE queue_tasks 
                    SET status = %s, error_message = %s, completed_at = %s
                    WHERE id = %s
                    """,
                    (status, error_message, datetime.now(), task_id)
                )
            else:
                cursor.execute(
                    """
                    UPDATE queue_tasks 
                    SET status = %s
                    WHERE id = %s
                    """,
                    (status, task_id)
                )
                
    def update_track_status(self, track_id: int, status: str) -> None:
        """Update track analysis status in database."""
        with self.db_conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE tracks 
                SET analysis_status = %s, updated_at = %s
                WHERE id = %s
                """,
                (status, datetime.now(), track_id)
            )
            
    def update_track_analysis(self, track_id: int, analysis: Dict[str, Any]) -> None:
        """Update track with analysis results."""
        import json
        
        # Convert tags to JSON strings
        genre_tags = analysis.get('genre_tags')
        mood_tags = analysis.get('mood_tags')
        genre_tags_json = json.dumps(genre_tags) if genre_tags else None
        mood_tags_json = json.dumps(mood_tags) if mood_tags else None
        
        with self.db_conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE tracks 
                SET genre = %s, genre_tags = %s, bpm = %s, key = %s, mode = %s, 
                    mood = %s, mood_tags = %s, has_vocals = %s, is_acoustic = %s,
                    loudness_integrated_lufs = %s, loudness_true_peak_db = %s,
                    loudness_gain_db = %s, loudness_target_lufs = %s,
                    embedding_path = %s, analysis_status = 'completed', updated_at = %s
                WHERE id = %s
                """,
                (
                    analysis.get('genre'),
                    genre_tags_json,
                    analysis.get('bpm'),
                    analysis.get('key'),
                    analysis.get('mode'),
                    analysis.get('mood'),
                    mood_tags_json,
                    analysis.get('has_vocals'),
                    analysis.get('is_acoustic'),
                    analysis.get('loudness_integrated_lufs'),
                    analysis.get('loudness_true_peak_db'),
                    analysis.get('loudness_gain_db'),
                    analysis.get('loudness_target_lufs'),
                    analysis.get('embedding_path'),
                    datetime.now(),
                    track_id
                )
            )

    def _update_segment_analysis_status(
        self,
        track_id: int,
        status: str,
        progress: int,
        message: str,
        **extra: Any,
    ) -> None:
        redis_prefix = os.getenv('REDIS_PREFIX', 'ai_music_recommendations_database_')
        status_key = f"{redis_prefix}segment_analysis_status:{track_id}"
        status_payload = {
            'status': status,
            'progress': progress,
            'message': message,
            'track_id': track_id,
            'updated_at': datetime.now().isoformat(),
            **extra,
        }

        redis_client = self._new_redis_client()
        try:
            redis_client.setex(status_key, 300, json.dumps(status_payload))
        finally:
            try:
                redis_client.close()
            except Exception:
                pass

    def update_stem_tuning_render_status(
        self,
        task_id: int,
        status: str,
        progress: int,
        message: str,
        *,
        level: str = 'info',
        **extra: Any,
    ) -> None:
        status_key = f"{self.REDIS_PREFIX}stem_tuning_render_status:{task_id}"
        redis_client = self._new_redis_client()
        try:
            existing_raw = redis_client.get(status_key)
            existing = json.loads(existing_raw) if existing_raw else {}
            logs = existing.get('logs') if isinstance(existing, dict) else []
            if not isinstance(logs, list):
                logs = []

            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': 'error' if level == 'error' else 'info',
                'progress': int(max(0, min(100, progress))),
                'message': message,
                'stage': extra.get('stage'),
            }

            logs.append(log_entry)
            logs = logs[-120:]

            payload = {
                'status': status,
                'progress': int(max(0, min(100, progress))),
                'message': message,
                'task_id': int(task_id),
                'updated_at': datetime.now().isoformat(),
                'logs': logs,
                **extra,
            }

            redis_client.setex(status_key, 3600, json.dumps(payload))
        finally:
            try:
                redis_client.close()
            except Exception:
                pass

    def get_stem_tuning_render_cancel_request(self, task_id: int) -> Optional[Dict[str, Any]]:
        control_key = f"{self.REDIS_PREFIX}stem_tuning_render_control:{task_id}"
        redis_client = self._new_redis_client()
        try:
            raw = redis_client.get(control_key)
            if not raw:
                return None
            decoded = json.loads(raw)
            return decoded if isinstance(decoded, dict) else None
        finally:
            try:
                redis_client.close()
            except Exception:
                pass

    def clear_stem_tuning_render_cancel_request(self, task_id: int) -> None:
        control_key = f"{self.REDIS_PREFIX}stem_tuning_render_control:{task_id}"
        redis_client = self._new_redis_client()
        try:
            redis_client.delete(control_key)
        finally:
            try:
                redis_client.close()
            except Exception:
                pass

    def _requeue_timed_out_segment_task(self, task_data: Dict[str, Any], timeout_seconds: int) -> None:
        task_id = task_data.get('task_id')
        track_id = task_data.get('track_id')
        retry_count = int(task_data.get('_hard_timeout_retry_count', 0) or 0)

        if track_id is None:
            logger.error(
                "Hard-timeout analyze_segments task missing track_id; cannot requeue task_id=%s",
                task_id,
            )
            return

        if retry_count >= self.ANALYZE_SEGMENTS_HARD_TIMEOUT_REQUEUE_LIMIT:
            self._update_segment_analysis_status(
                int(track_id),
                'failed',
                0,
                f'Segment analysis exceeded {timeout_seconds}s {retry_count + 1} times; manual intervention required.',
                task_id=task_id,
                retry_count=retry_count,
            )
            logger.error(
                "Hard timeout retry limit reached for analyze_segments task_id=%s track_id=%s retries=%s",
                task_id,
                track_id,
                retry_count,
            )
            return

        requeued_task = dict(task_data)
        requeued_task['_hard_timeout_retry_count'] = retry_count + 1
        requeued_task['_hard_timeout_requeued_at'] = datetime.now().isoformat()

        redis_client = self._new_redis_client()
        try:
            redis_client.rpush(self.QUEUE_NAME, json.dumps(requeued_task))
        finally:
            try:
                redis_client.close()
            except Exception:
                pass

        self._update_segment_analysis_status(
            int(track_id),
            'queued',
            0,
            f'Segment analysis exceeded {timeout_seconds}s; requeued and restarting worker.',
            task_id=task_id,
            retry_count=retry_count + 1,
        )
        logger.error(
            "Requeued timed-out analyze_segments task_id=%s track_id=%s retry=%s",
            task_id,
            track_id,
            retry_count + 1,
        )

    def _cleanup_process_owned_analysis_locks(self, task_data: Dict[str, Any]) -> None:
        redis_prefix = os.getenv('REDIS_PREFIX', 'ai_music_recommendations_database_')
        segment_lock_key = f"{redis_prefix}ml_worker:segment_analysis_lock"
        segment_lock_heartbeat_key = f"{redis_prefix}ml_worker:segment_analysis_lock_heartbeat"
        structure_lock_key = f"{redis_prefix}ml_worker:structure_gpu_lock"
        track_id = task_data.get('track_id')
        hostname = socket.gethostname()
        pid = os.getpid()

        redis_client = self._new_redis_client()
        try:
            if track_id is not None:
                segment_owner_prefix = f"{hostname}:{pid}:{track_id}:"
                segment_lock_owner = redis_client.get(segment_lock_key)
                if segment_lock_owner and segment_lock_owner.startswith(segment_owner_prefix):
                    removed = redis_client.delete(segment_lock_key, segment_lock_heartbeat_key)
                    logger.warning(
                        "Released process-owned segment analysis lock during hard-timeout cleanup: "
                        "key=%s heartbeat_key=%s track_id=%s owner=%s removed=%s",
                        segment_lock_key,
                        segment_lock_heartbeat_key,
                        track_id,
                        segment_lock_owner,
                        removed,
                    )

            structure_owner_prefix = f"{hostname}:{pid}:"
            structure_lock_owner = redis_client.get(structure_lock_key)
            if structure_lock_owner and structure_lock_owner.startswith(structure_owner_prefix):
                removed = redis_client.delete(structure_lock_key)
                logger.warning(
                    "Released process-owned structure GPU lock during hard-timeout cleanup: "
                    "key=%s owner=%s removed=%s",
                    structure_lock_key,
                    structure_lock_owner,
                    removed,
                )
        finally:
            try:
                redis_client.close()
            except Exception:
                pass

    def _start_hard_timeout_watchdog(
        self,
        task_type: Optional[str],
        task_data: Dict[str, Any],
        timeout_seconds: int,
    ) -> Optional[tuple[threading.Event, threading.Thread]]:
        if task_type != 'analyze_segments' or timeout_seconds <= 0:
            return None

        stop_event = threading.Event()
        task_id = task_data.get('task_id')
        track_id = task_data.get('track_id')

        def watchdog() -> None:
            if stop_event.wait(timeout_seconds):
                return

            logger.error(
                "Hard timeout reached for analyze_segments task_id=%s track_id=%s after %ss; requeueing and exiting worker process",
                task_id,
                track_id,
                timeout_seconds,
            )
            try:
                self._requeue_timed_out_segment_task(task_data, timeout_seconds)
            except Exception as exc:
                logger.exception(
                    "Failed to requeue timed-out analyze_segments task_id=%s track_id=%s: %s",
                    task_id,
                    track_id,
                    exc,
                )
            try:
                self._cleanup_process_owned_analysis_locks(task_data)
            except Exception as exc:
                logger.exception(
                    "Failed to clean up process-owned analysis locks for timed-out "
                    "analyze_segments task_id=%s track_id=%s: %s",
                    task_id,
                    track_id,
                    exc,
                )
            os._exit(124)

        thread = threading.Thread(
            target=watchdog,
            name=f"analyze-segments-hard-timeout-{task_id}",
            daemon=True,
        )
        thread.start()
        return stop_event, thread

    def process_task_with_hard_timeout(self, task_data: Dict[str, Any]) -> None:
        task_type = task_data.get('type')
        timeout_seconds = self.TASK_TIMEOUT
        hard_timeout_watchdog = self._start_hard_timeout_watchdog(task_type, task_data, timeout_seconds)

        try:
            self.process_task(task_data)
        finally:
            if hard_timeout_watchdog is not None:
                stop_event, watchdog_thread = hard_timeout_watchdog
                stop_event.set()
                watchdog_thread.join(timeout=0.1)
             
    def process_task(self, task_data: Dict[str, Any]) -> None:
        """Process a single task with timeout protection."""
        task_type = task_data.get('type')
        task_id = task_data.get('task_id')
        track_id = task_data.get('track_id')
        
        if task_type not in self.handlers:
            logger.error(f"Unknown task type: {task_type}")
            if task_id:
                self.update_task_status(task_id, 'failed', f'Unknown task type: {task_type}')
            return
        
        self.current_task_id = task_id
        self.task_start_time = time.time()
        
        try:
            # Update status to processing
            if task_id:
                self.update_task_status(task_id, 'processing')
            
            # Execute handler with timeout monitoring
            handler = self.handlers[task_type]
            
            # Use threading-based timeout check
            timeout_seconds = self.TASK_TIMEOUT
            task_info = f"type={task_type}, task_id={task_id}, track_id={track_id}"
            
            with task_timeout(timeout_seconds, task_info) as timeout_state:
                result = handler(task_data)
                # NOTE: timeout_state['timed_out'] is intentionally NOT checked here.
                # task_timeout is a soft timer — it cannot interrupt a running handler.
                # If handler() returned, the task completed successfully regardless of
                # elapsed time. Raising here would discard a valid result and mark the
                # track as failed. Truly stuck tasks never return, so the timer fires
                # inside the context manager's finally block instead.
            
            # Update status to completed
            if task_id:
                self.update_task_status(task_id, 'completed')
            
            elapsed = time.time() - self.task_start_time
            logger.info(f"Task {task_id} completed successfully in {elapsed:.1f}s")
            
        except TaskTimeoutError as e:
            logger.error(f"Task {task_id} TIMEOUT: {e}")
            if task_id:
                self.update_task_status(task_id, 'failed', f'Timeout: {str(e)}')
            if track_id:
                self.update_track_status(track_id, 'failed')
            # Clean up GPU memory after timeout
            self._cleanup_gpu_memory()

        except TaskCancelledError as e:
            logger.info(f"Task {task_id} cancelled: {e}")
            if task_id:
                self.update_task_status(task_id, 'cancelled', str(e)[:500])

        except Exception as e:
            logger.exception(f"Task {task_id} failed: {e}")
            if task_id:
                self.update_task_status(task_id, 'failed', str(e)[:500])  # Limit error message length
            if track_id:
                self.update_track_status(track_id, 'failed')
        
        finally:
            self.current_task_id = None
            self.task_start_time = None
            # Periodic GPU cleanup every task
            self._cleanup_gpu_memory()

    def _collect_track_analysis_batch(self, first_task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect a small micro-batch of track_analysis tasks from the queue head."""
        if (
            first_task.get('type') != 'track_analysis'
            or 'track_analysis' not in self.batch_handlers
            or self.TRACK_ANALYSIS_BATCH_SIZE <= 1
            or self.redis_client is None
        ):
            return [first_task]

        batch = [first_task]
        deadline = time.time() + (self.TRACK_ANALYSIS_BATCH_WAIT_MS / 1000.0)

        while len(batch) < self.TRACK_ANALYSIS_BATCH_SIZE:
            next_task_json = self.redis_client.lindex(self.QUEUE_NAME, 0)
            if next_task_json is None:
                if time.time() < deadline:
                    time.sleep(0.05)
                    continue
                break

            try:
                next_task = json.loads(next_task_json)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON at queue head while batching, discarding task: {e}")
                self.redis_client.lpop(self.QUEUE_NAME)
                continue

            if next_task.get('type') != 'track_analysis':
                break

            self.redis_client.lpop(self.QUEUE_NAME)
            batch.append(next_task)
            logger.info(f"Batched task: track_analysis (ID: {next_task.get('task_id')})")

        if len(batch) > 1:
            logger.info(
                "Collected track_analysis batch size=%s task_ids=%s",
                len(batch),
                [task.get('task_id') for task in batch],
            )

        return batch

    def process_task_batch(self, task_batch: List[Dict[str, Any]]) -> None:
        """Process a batch of tasks with timeout protection."""
        if not task_batch:
            return

        task_type = task_batch[0].get('type')
        if task_type not in self.batch_handlers:
            for task_data in task_batch:
                self.process_task(task_data)
            return

        task_ids = [task.get('task_id') for task in task_batch]
        self.current_task_id = None
        self.task_start_time = time.time()

        try:
            for task_data in task_batch:
                task_id = task_data.get('task_id')
                if task_id:
                    self.update_task_status(task_id, 'processing')

            handler = self.batch_handlers[task_type]
            timeout_seconds = self.TASK_TIMEOUT * max(1, len(task_batch))
            task_info = f"type={task_type}, task_ids={task_ids}"

            with task_timeout(timeout_seconds, task_info):
                outcomes = handler(task_batch)

            outcome_by_task_id = {
                outcome.get('task_id'): outcome
                for outcome in outcomes or []
                if outcome.get('task_id') is not None
            }

            for task_data in task_batch:
                task_id = task_data.get('task_id')
                track_id = task_data.get('track_id')
                outcome = outcome_by_task_id.get(task_id)
                if outcome and outcome.get('status') == 'completed':
                    if task_id:
                        self.update_task_status(task_id, 'completed')
                else:
                    error_message = str((outcome or {}).get('error', 'Batch handler did not return a success result'))[:500]
                    logger.error(f"Task {task_id} failed in batch: {error_message}")
                    if task_id:
                        self.update_task_status(task_id, 'failed', error_message)
                    if track_id:
                        self.update_track_status(track_id, 'failed')

            elapsed = time.time() - self.task_start_time
            logger.info(
                "Task batch completed successfully: type=%s size=%s elapsed=%.1fs",
                task_type,
                len(task_batch),
                elapsed,
            )

        except TaskTimeoutError as e:
            logger.error(f"Task batch TIMEOUT ({task_ids}): {e}")
            for task_data in task_batch:
                task_id = task_data.get('task_id')
                track_id = task_data.get('track_id')
                if task_id:
                    self.update_task_status(task_id, 'failed', f'Timeout: {str(e)}')
                if track_id:
                    self.update_track_status(track_id, 'failed')
            self._cleanup_gpu_memory()

        except Exception as e:
            logger.exception(f"Task batch failed ({task_ids}): {e}")
            for task_data in task_batch:
                task_id = task_data.get('task_id')
                track_id = task_data.get('track_id')
                if task_id:
                    self.update_task_status(task_id, 'failed', str(e)[:500])
                if track_id:
                    self.update_track_status(track_id, 'failed')

        finally:
            self.current_task_id = None
            self.task_start_time = None
            self._cleanup_gpu_memory()
    
    def _cleanup_gpu_memory(self) -> None:
        """Clean up GPU memory to prevent fragmentation and OOM."""
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                # Don't synchronize here - it can trigger OOM
        except Exception as e:
            logger.debug(f"GPU cleanup: {e}")
    
    def _deep_cleanup_gpu_memory(self) -> None:
        """Deep GPU memory cleanup when idle - releases all cached memory."""
        import gc
        
        # First, aggressive Python GC
        for _ in range(3):
            gc.collect()
        
        try:
            import torch
            if not torch.cuda.is_available():
                return
            
            # Try to empty cache without synchronize first
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            
            # Try synchronize separately - this can fail if GPU is in bad state
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            
            # Try to reset stats
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            
            # Log memory state
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"Deep GPU cleanup: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
            except Exception:
                logger.info("Deep GPU cleanup completed (stats unavailable)")
                
        except Exception as e:
            logger.warning(f"Deep GPU cleanup failed: {e}")
                
    def run(self) -> None:
        """Main worker loop with watchdog."""
        self.connect_redis()
        self.connect_database()
        
        self.running = True
        logger.info(f"Worker started, listening on queue: {self.QUEUE_NAME}")
        logger.info(f"Task timeout: {self.TASK_TIMEOUT}s, Analysis timeout: {self.ANALYSIS_TIMEOUT}s")
        
        # Track consecutive errors for circuit breaker
        consecutive_errors = 0
        max_consecutive_errors = 5
        idle_count = 0  # Track idle iterations for memory cleanup
        
        while self.running:
            try:
                # Blocking pop from queue (timeout 5 seconds)
                result = self.redis_client.blpop(self.QUEUE_NAME, timeout=5)
                
                if result is None:
                    consecutive_errors = 0  # Reset on idle
                    idle_count += 1
                    
                    # Deep cleanup after 12 idle iterations (1 minute of no tasks)
                    if idle_count >= 12:
                        self._deep_cleanup_gpu_memory()
                        idle_count = 0
                    continue
                
                idle_count = 0  # Reset idle counter when we have work
                    
                _, task_json = result
                logger.info(f"Raw task data: {repr(task_json)[:200]}")
                
                # Handle malformed JSON gracefully
                try:
                    task_data = json.loads(task_json)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in queue, skipping task: {e}")
                    continue
                
                logger.info(f"Received task: {task_data.get('type')} (ID: {task_data.get('task_id')})")
                task_batch = self._collect_track_analysis_batch(task_data)
                if len(task_batch) > 1:
                    self.process_task_batch(task_batch)
                else:
                    self.process_task_with_hard_timeout(task_data)
                consecutive_errors = 0  # Reset on success
                
            except redis.ConnectionError:
                logger.error("Redis connection lost, reconnecting...")
                consecutive_errors += 1
                time.sleep(5)
                try:
                    self.connect_redis()
                except Exception as e:
                    logger.error(f"Redis reconnect failed: {e}")
                
            except psycopg2.OperationalError as e:
                logger.error(f"Database connection lost: {e}, reconnecting...")
                consecutive_errors += 1
                time.sleep(5)
                try:
                    self.connect_database()
                except Exception as e:
                    logger.error(f"Database reconnect failed: {e}")
                
            except Exception as e:
                logger.exception(f"Error in worker loop: {e}")
                consecutive_errors += 1
                time.sleep(1)
            
            # Circuit breaker: if too many consecutive errors, pause longer
            if consecutive_errors >= max_consecutive_errors:
                logger.warning(f"Circuit breaker: {consecutive_errors} consecutive errors, pausing 30s")
                time.sleep(30)
                consecutive_errors = 0
                # Try to reconnect everything
                try:
                    self.connect_redis()
                    self.connect_database()
                except Exception as e:
                    logger.error(f"Reconnection failed: {e}")
                
    def stop(self) -> None:
        """Stop the worker."""
        self.running = False
        logger.info("Worker stopping...")
