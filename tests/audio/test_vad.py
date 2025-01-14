"""Unit tests for voice activity detection."""
import pytest
import torch
from app.audio.config import AudioConfig
from app.audio.vad import VoiceActivityDetector, VADResult


@pytest.fixture
def vad():
    """Create VAD instance for testing."""
    config = AudioConfig(encoder_type="wav2vec2")
    return VoiceActivityDetector(config, device=torch.device("cpu"))


@pytest.fixture
def silence():
    """Create silent audio for testing."""
    return torch.zeros(1, 16000)  # 1 second of silence


@pytest.fixture
def speech():
    """Create simulated speech for testing."""
    # Create a simple sine wave to simulate speech
    t = torch.linspace(0, 1, 16000)
    return torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # 440Hz tone


def test_vad_initialization(vad):
    """Test VAD initialization."""
    assert vad.energy_threshold == 0.1
    assert vad.speech_threshold == 0.5
    if vad.config.encoder_type == "wav2vec2":
        assert vad.vad_model is not None
    else:
        assert vad.vad_model is None


def test_vad_silence_detection(vad, silence):
    """Test VAD on silence."""
    result = vad.detect_speech(silence, use_model=False)
    assert isinstance(result, VADResult)
    assert not result.is_speech
    assert result.energy_level < vad.energy_threshold


def test_vad_speech_detection(vad, speech):
    """Test VAD on speech."""
    result = vad.detect_speech(speech, use_model=False)
    assert isinstance(result, VADResult)
    assert result.is_speech
    assert result.energy_level > vad.energy_threshold


def test_vad_model_based_detection(vad, speech):
    """Test model-based VAD."""
    if vad.vad_model is not None:
        result = vad.detect_speech(speech, use_model=True)
        assert isinstance(result, VADResult)
        assert isinstance(result.confidence, float)
        assert 0 <= result.confidence <= 1


def test_segment_type_classification(vad):
    """Test segment type classification."""
    # Test speech segment
    speech_result = VADResult(
        is_speech=True,
        confidence=0.9,
        energy_level=0.5
    )
    assert vad.get_segment_type(speech_result) == "speech"

    # Test audio segment
    audio_result = VADResult(
        is_speech=False,
        confidence=0.3,
        energy_level=0.07
    )
    assert vad.get_segment_type(audio_result) == "audio"

    # Test silence segment
    silence_result = VADResult(
        is_speech=False,
        confidence=0.1,
        energy_level=0.01
    )
    assert vad.get_segment_type(silence_result) == "silence"
