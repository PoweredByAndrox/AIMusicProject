"""
Property tests for BPM compatibility in DJ mixing.

**Property 21: BPM-совместимость для DJ микширования**
**Validates: Requirements 28.2, 29.4, 29.5**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.dj_analyzer import AIDJMatcher


# Strategy for realistic BPM values (60-200)
bpm_strategy = st.floats(min_value=60.0, max_value=200.0, allow_nan=False, allow_infinity=False)


class TestBPMCompatibility:
    """
    Property 21: BPM Compatibility for DJ Mixing
    
    Validates: Requirements 28.2, 29.4, 29.5
    """
    
    def setup_method(self):
        self.matcher = AIDJMatcher()
    
    @given(bpm=bpm_strategy)
    @settings(max_examples=100)
    def test_same_bpm_score_is_one(self, bpm: float):
        """Same BPM should have score 1.0."""
        score = self.matcher._calculate_bpm_score(bpm, bpm)
        assert score == 1.0, f"Same BPM score should be 1.0, got {score}"
    
    @given(bpm=bpm_strategy)
    @settings(max_examples=100)
    def test_within_tolerance_high_score(self, bpm: float):
        """BPM within 6% tolerance should have score >= 0.5."""
        # 3% difference (half of tolerance)
        other_bpm = bpm * 1.03
        score = self.matcher._calculate_bpm_score(bpm, other_bpm)
        assert score >= 0.5, f"BPM within tolerance should have score >= 0.5, got {score}"
    
    @given(bpm=bpm_strategy)
    @settings(max_examples=100)
    def test_at_tolerance_limit_score(self, bpm: float):
        """BPM at exactly 6% tolerance should have score ~0.5."""
        other_bpm = bpm * 1.06
        score = self.matcher._calculate_bpm_score(bpm, other_bpm)
        # Allow some floating point tolerance
        assert 0.45 <= score <= 0.55, f"BPM at tolerance limit should have score ~0.5, got {score}"
    
    @given(bpm=bpm_strategy)
    @settings(max_examples=100)
    def test_beyond_tolerance_low_score(self, bpm: float):
        """BPM beyond 12% should have score 0."""
        other_bpm = bpm * 1.15  # 15% difference
        score = self.matcher._calculate_bpm_score(bpm, other_bpm)
        assert score == 0.0, f"BPM beyond tolerance should have score 0, got {score}"
    
    @given(bpm=bpm_strategy)
    @settings(max_examples=100)
    def test_half_time_compatible(self, bpm: float):
        """Half-time BPM should be compatible."""
        assume(bpm >= 120)  # Ensure half-time is still in valid range
        
        half_bpm = bpm / 2
        score = self.matcher._calculate_bpm_score(bpm, half_bpm)
        assert score >= 0.5, f"Half-time BPM should be compatible, got score {score}"
    
    @given(bpm=st.floats(min_value=60.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_double_time_compatible(self, bpm: float):
        """Double-time BPM should be compatible."""
        double_bpm = bpm * 2
        score = self.matcher._calculate_bpm_score(bpm, double_bpm)
        assert score >= 0.5, f"Double-time BPM should be compatible, got score {score}"
    
    @given(bpm1=bpm_strategy, bpm2=bpm_strategy)
    @settings(max_examples=100)
    def test_score_is_symmetric(self, bpm1: float, bpm2: float):
        """BPM compatibility should be roughly symmetric."""
        score1 = self.matcher._calculate_bpm_score(bpm1, bpm2)
        score2 = self.matcher._calculate_bpm_score(bpm2, bpm1)
        
        # BPM score is not perfectly symmetric because percentage is relative to source
        # But both should be in same "compatibility zone" (both high, both low, etc.)
        # Allow larger tolerance for asymmetry
        assert abs(score1 - score2) < 0.2, \
            f"BPM score should be roughly symmetric: {score1} vs {score2}"
    
    @given(bpm1=bpm_strategy, bpm2=bpm_strategy)
    @settings(max_examples=100)
    def test_score_in_valid_range(self, bpm1: float, bpm2: float):
        """BPM score should always be between 0 and 1."""
        score = self.matcher._calculate_bpm_score(bpm1, bpm2)
        assert 0.0 <= score <= 1.0, f"Score should be in [0, 1], got {score}"
    
    @given(bpm=bpm_strategy)
    @settings(max_examples=100)
    def test_is_bpm_compatible_same(self, bpm: float):
        """Same BPM should always be compatible."""
        assert self.matcher.is_bpm_compatible(bpm, bpm), "Same BPM should be compatible"
    
    @given(bpm=bpm_strategy)
    @settings(max_examples=100)
    def test_is_bpm_compatible_within_tolerance(self, bpm: float):
        """BPM within 6% should be compatible."""
        other_bpm = bpm * 1.05  # 5% difference
        assert self.matcher.is_bpm_compatible(bpm, other_bpm), \
            f"BPM within tolerance should be compatible: {bpm} vs {other_bpm}"


class TestEnergyCompatibility:
    """Test energy level compatibility."""
    
    def setup_method(self):
        self.matcher = AIDJMatcher()
    
    @given(energy=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50)
    def test_same_energy_score_is_one(self, energy: int):
        """Same energy level should have score 1.0."""
        score = self.matcher._calculate_energy_score(energy, energy)
        assert score == 1.0, f"Same energy score should be 1.0, got {score}"
    
    @given(energy=st.integers(min_value=2, max_value=9))
    @settings(max_examples=50)
    def test_adjacent_energy_high_score(self, energy: int):
        """Adjacent energy levels (±1) should have high score."""
        score_up = self.matcher._calculate_energy_score(energy, energy + 1)
        score_down = self.matcher._calculate_energy_score(energy, energy - 1)
        
        assert score_up == 0.9, f"Energy +1 should have score 0.9, got {score_up}"
        assert score_down == 0.9, f"Energy -1 should have score 0.9, got {score_down}"
    
    @given(energy1=st.integers(min_value=1, max_value=10),
           energy2=st.integers(min_value=1, max_value=10))
    @settings(max_examples=100)
    def test_energy_score_in_valid_range(self, energy1: int, energy2: int):
        """Energy score should always be between 0 and 1."""
        score = self.matcher._calculate_energy_score(energy1, energy2)
        assert 0.0 <= score <= 1.0, f"Score should be in [0, 1], got {score}"
    
    @given(energy1=st.integers(min_value=1, max_value=10),
           energy2=st.integers(min_value=1, max_value=10))
    @settings(max_examples=100)
    def test_energy_score_is_symmetric(self, energy1: int, energy2: int):
        """Energy compatibility should be symmetric."""
        score1 = self.matcher._calculate_energy_score(energy1, energy2)
        score2 = self.matcher._calculate_energy_score(energy2, energy1)
        assert score1 == score2, f"Energy score should be symmetric: {score1} vs {score2}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
