"""Unit tests for audio feature extraction."""
import pytest
import torch
import torch.nn as nn
from app.audio.config import AudioConfig
from app.audio.feature_extractor import (
    AudioFeatureExtractor,
    StackAudioFrames,
    StreamingFeatureState,
    AudioSegment
)


@pytest.fixture
def config():
    """Create audio config for testing."""
    return AudioConfig(
        encoder_type="wav2vec2",
        feature_layers=[6, 9, 12],
        stack_factor=4,
        text_hidden_size=768
    )


@pytest.fixture
def feature_extractor(config):
    """Create feature extractor for testing."""
    return AudioFeatureExtractor(config, device=torch.device("cpu"))


@pytest.fixture
def audio_input():
    """Create test audio input."""
    # Create 1 second of audio at 16kHz
    return torch.randn(1, 16000)


def test_feature_extractor_initialization(feature_extractor):
    """Test feature extractor initialization."""
    assert isinstance(feature_extractor.encoder, nn.Module)
    assert len(feature_extractor.projections) == len(
        feature_extractor.config.feature_layers) + 1
    assert isinstance(feature_extractor.frame_stacker, StackAudioFrames)
    assert feature_extractor.frame_stacker.stack_factor == 4


def test_frame_stacker():
    """Test frame stacking functionality."""
    stacker = StackAudioFrames(stack_factor=4)
    features = torch.randn(2, 100, 32)  # [batch, length, dim]
    stacked = stacker(features)

    # Check output shape
    assert stacked.shape[0] == 2  # batch size preserved
    assert stacked.shape[1] == 25  # length reduced by stack_factor
    assert stacked.shape[2] == 32 * 4  # dimension increased by stack_factor


def test_streaming_feature_state():
    """Test streaming feature state."""
    state = StreamingFeatureState(
        hidden_size=768,
        num_layers=3,
        max_buffer_size=32000
    )

    # Add features
    features = torch.randn(1, 100, 768)
    hidden_states = [torch.randn(1, 100, 768) for _ in range(3)]
    state.add_features(
        features,
        hidden_states,
        is_speech=True,
        energy_level=0.5
    )

    # Check buffer state
    assert len(state.features_buffer) == 1
    assert len(state.hidden_states_buffer) == 3
    assert state.current_segment_type == "speech"


def test_feature_extraction(feature_extractor, audio_input):
    """Test feature extraction process."""
    features, hidden_states, vad_result, _ = feature_extractor._process_chunk(
        audio_input,
        return_hidden_states=True
    )

    # Check output shapes and types
    assert isinstance(features, torch.Tensor)
    assert features.shape[-1] == feature_extractor.config.text_hidden_size
    assert len(hidden_states) == len(feature_extractor.config.feature_layers)
    for states in hidden_states:
        assert states.shape[-1] == feature_extractor.config.text_hidden_size


def test_streaming_processing(feature_extractor, audio_input):
    """Test streaming audio processing."""
    # Process first chunk
    result = feature_extractor.process_stream(audio_input)

    # Process with force flush
    result = feature_extractor.process_stream(audio_input, force_flush=True)
    if result:
        features, hidden_states, metadata, transcription = result
        assert isinstance(features, torch.Tensor)
        assert isinstance(metadata, list)
        for meta in metadata:
            assert "is_speech" in meta
            assert "energy_level" in meta
            assert "segment_type" in meta


def test_segment_processing(feature_extractor, audio_input):
    """Test processing of multiple segments."""
    segments = feature_extractor.process_segments(
        audio_inputs=[audio_input, audio_input],
        token_ids=["<|audio|>", "<|audio|>"],
        start_indices=[0, 100]
    )

    assert len(segments) == 2
    for segment in segments:
        assert isinstance(segment, AudioSegment)
        assert segment.features.shape[-1] == feature_extractor.config.text_hidden_size
        assert isinstance(segment.is_speech, bool)
        assert isinstance(segment.energy_level, float)
        assert segment.segment_type in ["speech", "audio", "silence"]
