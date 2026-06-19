from __future__ import annotations

import pytest

from tools.artifact_name_policy import validate_artifact_relative_path, validate_artifact_segment


def test_validate_artifact_relative_path_rejects_parent_traversal() -> None:
    with pytest.raises(ValueError):
        validate_artifact_relative_path("../outside.flac")

    with pytest.raises(ValueError):
        validate_artifact_relative_path("audio/../outside.flac")


def test_validate_artifact_segment_rejects_windows_reserved_names() -> None:
    with pytest.raises(ValueError):
        validate_artifact_segment("CON")

    with pytest.raises(ValueError):
        validate_artifact_segment("com1.flac")
