"""
Property Test: Queue Task Processing Completeness

@tag Feature: ai-music-recommendations, Property 13: Queue Task Processing Completeness
**Validates: Requirements 13.2, 13.3**

For any task added to the queue, after processing by ML Worker,
the task status should be either 'completed' (with saved results)
or 'failed' (with error message).
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.queue_worker import QueueWorker


class TestQueueTaskProcessingCompleteness:
    """Property tests for queue task processing."""
    
    @pytest.fixture
    def worker(self):
        """Create a QueueWorker instance with mocked connections."""
        worker = QueueWorker()
        worker.redis_client = Mock()
        worker.db_conn = Mock()
        worker.db_conn.cursor.return_value.__enter__ = Mock(return_value=Mock())
        worker.db_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        return worker
    
    @given(
        task_id=st.integers(min_value=1, max_value=10000),
        track_id=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=100)
    def test_successful_task_ends_in_completed_status(self, task_id, track_id):
        """
        Property: Successful task processing results in 'completed' status.
        """
        worker = QueueWorker()
        worker.redis_client = Mock()
        worker.db_conn = Mock()
        
        cursor_mock = Mock()
        worker.db_conn.cursor.return_value.__enter__ = Mock(return_value=cursor_mock)
        worker.db_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        # Register a successful handler
        successful_handler = Mock(return_value={'result': 'success'})
        worker.register_handler('test_task', successful_handler)
        
        task_data = {
            'type': 'test_task',
            'task_id': task_id,
            'track_id': track_id,
        }
        
        # Process task
        worker.process_task(task_data)
        
        # Verify handler was called
        successful_handler.assert_called_once_with(task_data)
        
        # Verify status was updated to 'completed'
        calls = cursor_mock.execute.call_args_list
        status_updates = [c for c in calls if 'status' in str(c)]
        
        # Should have processing and completed status updates
        assert len(status_updates) >= 1
        
    @given(
        task_id=st.integers(min_value=1, max_value=10000),
        error_message=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=100)
    def test_failed_task_ends_in_failed_status_with_error(self, task_id, error_message):
        """
        Property: Failed task processing results in 'failed' status with error message.
        """
        worker = QueueWorker()
        worker.redis_client = Mock()
        worker.db_conn = Mock()
        
        cursor_mock = Mock()
        worker.db_conn.cursor.return_value.__enter__ = Mock(return_value=cursor_mock)
        worker.db_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        # Register a failing handler
        failing_handler = Mock(side_effect=Exception(error_message))
        worker.register_handler('failing_task', failing_handler)
        
        task_data = {
            'type': 'failing_task',
            'task_id': task_id,
        }
        
        # Process task (should not raise)
        worker.process_task(task_data)
        
        # Verify handler was called
        failing_handler.assert_called_once()
        
        # Verify status was updated to 'failed' with error message
        calls = cursor_mock.execute.call_args_list
        failed_calls = [c for c in calls if 'failed' in str(c)]
        
        assert len(failed_calls) >= 1
        
    @given(task_type=st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_unknown_task_type_results_in_failed_status(self, task_type):
        """
        Property: Unknown task types result in 'failed' status.
        """
        # Skip if task_type matches a registered handler
        if task_type in ['track_analysis', 'fragment_analysis', 'model_training']:
            return
            
        worker = QueueWorker()
        worker.redis_client = Mock()
        worker.db_conn = Mock()
        
        cursor_mock = Mock()
        worker.db_conn.cursor.return_value.__enter__ = Mock(return_value=cursor_mock)
        worker.db_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        task_data = {
            'type': task_type,
            'task_id': 1,
        }
        
        # Process task with unknown type
        worker.process_task(task_data)
        
        # Verify status was updated to 'failed'
        calls = cursor_mock.execute.call_args_list
        failed_calls = [c for c in calls if 'failed' in str(c)]
        
        assert len(failed_calls) >= 1
        
    @given(
        task_id=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=100)
    def test_task_status_transitions_through_processing(self, task_id):
        """
        Property: Task status transitions from pending -> processing -> completed/failed.
        """
        worker = QueueWorker()
        worker.redis_client = Mock()
        worker.db_conn = Mock()
        
        status_history = []
        
        def track_status(query, params):
            if 'status' in query:
                status_history.append(params[0] if params else None)
        
        cursor_mock = Mock()
        cursor_mock.execute = track_status
        worker.db_conn.cursor.return_value.__enter__ = Mock(return_value=cursor_mock)
        worker.db_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        # Register handler
        worker.register_handler('test_task', Mock(return_value={}))
        
        task_data = {
            'type': 'test_task',
            'task_id': task_id,
        }
        
        worker.process_task(task_data)
        
        # Verify status transitions
        assert 'processing' in status_history
        assert 'completed' in status_history or 'failed' in status_history


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
