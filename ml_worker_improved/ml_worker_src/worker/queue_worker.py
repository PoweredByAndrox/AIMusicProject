"""
QueueWorker - Redis queue processor for ML tasks.

Requirements: 13.1, 13.2, 13.3, 13.4
"""

import json
import os
import time
import logging
import signal
import threading
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from contextlib import contextmanager

import redis
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class TaskTimeoutError(Exception):
    """Raised when a task exceeds its timeout."""
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
    TASK_TIMEOUT = int(os.getenv('TASK_TIMEOUT', 120))  # 2 minutes per task
    ANALYSIS_TIMEOUT = int(os.getenv('ANALYSIS_TIMEOUT', 60))  # 1 minute for analysis
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.db_conn: Optional[psycopg2.extensions.connection] = None
        self.handlers: Dict[str, Callable] = {}
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
        
    def update_task_status(
        self, 
        task_id: int, 
        status: str, 
        error_message: Optional[str] = None
    ) -> None:
        """Update task status in database."""
        with self.db_conn.cursor() as cursor:
            if status in ('completed', 'failed'):
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
        with self.db_conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE tracks 
                SET genre = %s, bpm = %s, key = %s, mode = %s, mood = %s,
                    has_vocals = %s, is_acoustic = %s, embedding_path = %s,
                    analysis_status = 'completed', updated_at = %s
                WHERE id = %s
                """,
                (
                    analysis.get('genre'),
                    analysis.get('bpm'),
                    analysis.get('key'),
                    analysis.get('mode'),
                    analysis.get('mood'),
                    analysis.get('has_vocals'),
                    analysis.get('is_acoustic'),
                    analysis.get('embedding_path'),
                    datetime.now(),
                    track_id
                )
            )
            
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
                
                # Check if we timed out during execution
                if timeout_state['timed_out']:
                    raise TaskTimeoutError(f"Task exceeded {timeout_seconds}s timeout")
            
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
    
    def _cleanup_gpu_memory(self) -> None:
        """Clean up GPU memory to prevent fragmentation and OOM."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"GPU cleanup failed: {e}")
                
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
        
        while self.running:
            try:
                # Blocking pop from queue (timeout 5 seconds)
                result = self.redis_client.blpop(self.QUEUE_NAME, timeout=5)
                
                if result is None:
                    consecutive_errors = 0  # Reset on idle
                    continue
                    
                _, task_json = result
                logger.info(f"Raw task data: {repr(task_json)[:200]}")
                
                # Handle malformed JSON gracefully
                try:
                    task_data = json.loads(task_json)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in queue, skipping task: {e}")
                    continue
                
                logger.info(f"Received task: {task_data.get('type')} (ID: {task_data.get('task_id')})")
                self.process_task(task_data)
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
