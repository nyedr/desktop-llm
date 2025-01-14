"""Unit tests for audio configuration."""
import pytest
from app.audio.config import AudioConfig, ChunkingPreset


def test_chunking_preset_fast():
    """Test fast chunking preset."""
    preset = ChunkingPreset.fast()
    assert preset.chunk_duration == 0.5
    assert preset.overlap_ratio == 0.1
    assert preset.min_speech_duration == 0.2
    assert preset.max_silence == 0.3


def test_chunking_preset_balanced():
    """Test balanced chunking preset."""
    preset = ChunkingPreset.balanced()
    assert preset.chunk_duration == 2.0
    assert preset.overlap_ratio == 0.15
    assert preset.min_speech_duration == 0.3
    assert preset.max_silence == 0.5


def test_chunking_preset_quality():
    """Test quality chunking preset."""
    preset = ChunkingPreset.quality()
    assert preset.chunk_duration == 5.0
    assert preset.overlap_ratio == 0.2
    assert preset.min_speech_duration == 0.5
    assert preset.max_silence == 1.0


def test_audio_config_defaults():
    """Test default audio configuration."""
    config = AudioConfig()
    assert config.sample_rate == 16000
    assert config.latency_preset == "balanced"
    assert config.chunk_duration is None
    assert config.overlap_ratio is None
    assert config.max_audio_length == 30
    assert config.stack_factor == 4
    assert config.encoder_type == "wav2vec2"
    assert config.feature_layers == [6, 9, 12]


def test_audio_config_chunking():
    """Test audio config chunking configuration."""
    config = AudioConfig(latency_preset="fast")
    chunking = config.chunking_config
    assert chunking.chunk_duration == 0.5
    assert chunking.overlap_ratio == 0.1

    # Test overrides
    config = AudioConfig(
        latency_preset="fast",
        chunk_duration=1.0,
        overlap_ratio=0.2
    )
    chunking = config.chunking_config
    assert chunking.chunk_duration == 1.0
    assert chunking.overlap_ratio == 0.2


def test_audio_config_validation():
    """Test audio config validation."""
    with pytest.raises(ValueError):
        AudioConfig(latency_preset="invalid")

    with pytest.raises(ValueError):
        AudioConfig(encoder_type="invalid")

    with pytest.raises(ValueError):
        AudioConfig(overlap_ratio=2.0)  # Must be between 0 and 1


def test_audio_config_immutability():
    """Test that audio config is immutable."""
    config = AudioConfig()
    with pytest.raises(Exception):
        config.sample_rate = 44100
