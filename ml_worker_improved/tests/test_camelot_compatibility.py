"""
Property tests for Camelot Wheel compatibility.

**Property 20: Совместимость треков по Camelot Wheel**
**Validates: Requirements 28.1**
"""

import pytest
from hypothesis import given, strategies as st, settings

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.dj_analyzer import (
    get_compatible_keys,
    is_valid_camelot_key,
    AIDJMatcher,
)


# Strategy for valid Camelot keys
camelot_key_strategy = st.sampled_from([
    '1A', '2A', '3A', '4A', '5A', '6A', '7A', '8A', '9A', '10A', '11A', '12A',
    '1B', '2B', '3B', '4B', '5B', '6B', '7B', '8B', '9B', '10B', '11B', '12B',
])


class TestCamelotCompatibility:
    """
    Property 20: Camelot Wheel Compatibility
    
    Validates: Requirements 28.1
    """
    
    @given(key=camelot_key_strategy)
    @settings(max_examples=50)
    def test_same_key_always_compatible(self, key: str):
        """Same key is always compatible with itself."""
        compatible = get_compatible_keys(key)
        assert key in compatible, f"Key {key} should be compatible with itself"
    
    @given(key=camelot_key_strategy)
    @settings(max_examples=50)
    def test_compatible_keys_count(self, key: str):
        """Each key should have exactly 4 compatible keys (including itself)."""
        compatible = get_compatible_keys(key)
        assert len(compatible) == 4, f"Key {key} should have 4 compatible keys, got {len(compatible)}"
    
    @given(key=camelot_key_strategy)
    @settings(max_examples=50)
    def test_adjacent_keys_compatible(self, key: str):
        """Adjacent keys (±1 position) should be compatible."""
        number = int(key[:-1])
        letter = key[-1]
        
        prev_num = 12 if number == 1 else number - 1
        next_num = 1 if number == 12 else number + 1
        
        compatible = get_compatible_keys(key)
        
        assert f"{prev_num}{letter}" in compatible, f"Previous key should be compatible"
        assert f"{next_num}{letter}" in compatible, f"Next key should be compatible"
    
    @given(key=camelot_key_strategy)
    @settings(max_examples=50)
    def test_ab_transition_compatible(self, key: str):
        """A↔B transition (same number) should be compatible."""
        number = int(key[:-1])
        letter = key[-1]
        other_letter = 'B' if letter == 'A' else 'A'
        
        compatible = get_compatible_keys(key)
        
        assert f"{number}{other_letter}" in compatible, f"A↔B transition should be compatible"
    
    @given(key=camelot_key_strategy)
    @settings(max_examples=50)
    def test_compatibility_is_symmetric(self, key: str):
        """If A is compatible with B, then B is compatible with A."""
        compatible_with_key = get_compatible_keys(key)
        
        for other_key in compatible_with_key:
            compatible_with_other = get_compatible_keys(other_key)
            assert key in compatible_with_other, \
                f"Compatibility should be symmetric: {key} -> {other_key} but not reverse"
    
    @given(key=camelot_key_strategy)
    @settings(max_examples=50)
    def test_all_compatible_keys_are_valid(self, key: str):
        """All returned compatible keys should be valid Camelot keys."""
        compatible = get_compatible_keys(key)
        
        for compat_key in compatible:
            assert is_valid_camelot_key(compat_key), \
                f"Compatible key {compat_key} should be valid"


class TestAIDJMatcherKeyScore:
    """Test key score calculation in AIDJMatcher."""
    
    def setup_method(self):
        self.matcher = AIDJMatcher()
    
    @given(key=camelot_key_strategy)
    @settings(max_examples=50)
    def test_same_key_score_is_one(self, key: str):
        """Same key should have score 1.0."""
        compatible = get_compatible_keys(key)
        score = self.matcher._calculate_key_score(key, key, compatible)
        assert score == 1.0, f"Same key score should be 1.0, got {score}"
    
    @given(key=camelot_key_strategy)
    @settings(max_examples=50)
    def test_compatible_key_score_positive(self, key: str):
        """Compatible keys should have positive score."""
        compatible = get_compatible_keys(key)
        
        for other_key in compatible:
            score = self.matcher._calculate_key_score(key, other_key, compatible)
            assert score > 0, f"Compatible key {other_key} should have positive score"
    
    @given(key=camelot_key_strategy)
    @settings(max_examples=50)
    def test_incompatible_key_score_zero(self, key: str):
        """Incompatible keys should have score 0."""
        compatible = get_compatible_keys(key)
        
        # Find an incompatible key (2 positions away)
        number = int(key[:-1])
        letter = key[-1]
        incompatible_num = ((number + 2 - 1) % 12) + 1  # +2 positions
        incompatible_key = f"{incompatible_num}{letter}"
        
        if incompatible_key not in compatible:
            score = self.matcher._calculate_key_score(key, incompatible_key, compatible)
            assert score == 0, f"Incompatible key should have score 0, got {score}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
