"""
ML Worker Package.

Queue-based worker for audio analysis and ML tasks.
Requirements: 13.1, 13.2, 13.3, 13.4
"""

__all__ = ["QueueWorker", "AudioAnalyzer"]


def __getattr__(name: str):
    if name == "QueueWorker":
        from .queue_worker import QueueWorker

        return QueueWorker
    if name == "AudioAnalyzer":
        from .audio_analyzer import AudioAnalyzer

        return AudioAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
