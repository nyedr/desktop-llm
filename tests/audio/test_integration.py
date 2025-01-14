"""Integration tests for audio processing system."""
import pytest
import torch
import numpy as np
from app.audio.config import AudioConfig
from app.audio.feature_extractor import AudioFeatureExtractor


@pytest.fixture
def config():
    """Create audio config for testing."""
    return AudioConfig(
        encoder_type="wav2vec2",
        latency_preset="fast",  # Use fast preset for quicker tests
        feature_layers=[6, 9, 12],
        stack_factor=4,
        text_hidden_size=768
    )


@pytest.fixture
def audio_processor(config):
    """Create audio processor for testing."""
    return AudioFeatureExtractor(config, device=torch.device("cpu"))


@pytest.fixture
def test_waveform():
    """Create test waveform with speech-like characteristics."""
    # Create 3 seconds of audio at 16kHz
    duration = 3
    sample_rate = 16000
    t = np.linspace(0, duration, duration * sample_rate)

    # Create a mixture of frequencies to simulate speech
    frequencies = [100, 200, 400, 800]
    waveform = np.zeros_like(t)
    for freq in frequencies:
        waveform += np.sin(2 * np.pi * freq * t)

    # Add some noise
    noise = np.random.normal(0, 0.1, len(t))
    waveform = waveform + noise

    # Normalize
    waveform = waveform / np.max(np.abs(waveform))

    # Convert to torch tensor
    return torch.from_numpy(waveform).float().unsqueeze(0)


def test_end_to_end_processing(audio_processor, test_waveform):
    """Test end-to-end audio processing pipeline."""
    # Get chunk size in samples
    chunk_duration = audio_processor.config.chunking_config.chunk_duration
    chunk_samples = int(chunk_duration * audio_processor.config.sample_rate)

    # Process audio in chunks
    num_chunks = test_waveform.shape[1] // chunk_samples
    all_features = []
    all_metadata = []

    for i in range(num_chunks):
        chunk = test_waveform[:, i*chunk_samples:(i+1)*chunk_samples]
        result = audio_processor.process_stream(chunk)

        if result:
            features, hidden_states, metadata, transcription = result
            all_features.append(features)
            all_metadata.extend(metadata)

    # Force flush the last chunk
    result = audio_processor.process_stream(
        test_waveform[:, -chunk_samples:],
        force_flush=True
    )
    if result:
        features, hidden_states, metadata, transcription = result
        all_features.append(features)
        all_metadata.extend(metadata)

    # Verify results
    assert len(all_features) > 0
    assert len(all_metadata) > 0

    # Check feature dimensions
    for features in all_features:
        assert features.shape[-1] == audio_processor.config.text_hidden_size

    # Check metadata
    for meta in all_metadata:
        assert "is_speech" in meta
        assert "energy_level" in meta
        assert "segment_type" in meta
        assert "timestamp" in meta
        assert "duration" in meta


def test_streaming_state_management(audio_processor, test_waveform):
    """Test streaming state management."""
    chunk_duration = audio_processor.config.chunking_config.chunk_duration
    chunk_samples = int(chunk_duration * audio_processor.config.sample_rate)

    # Initialize streaming state
    state = audio_processor.create_streaming_state()
    assert state is not None

    # Process chunks and verify state updates
    for i in range(0, test_waveform.shape[1], chunk_samples):
        chunk = test_waveform[:, i:i+chunk_samples]
        features, hidden_states, vad_result, _ = audio_processor._process_chunk(
            chunk,
            return_hidden_states=True
        )

        # Add to state
        state.add_features(
            features,
            hidden_states,
            is_speech=vad_result.is_speech,
            energy_level=vad_result.energy_level
        )

        # Verify buffer management
        assert len(state.features_buffer) > 0
        assert state.buffer_samples <= state.max_buffer_size


def test_vad_integration(audio_processor, test_waveform):
    """Test VAD integration with feature extraction."""
    chunk_duration = audio_processor.config.chunking_config.chunk_duration
    chunk_samples = int(chunk_duration * audio_processor.config.sample_rate)

    # Process a chunk with VAD
    chunk = test_waveform[:, :chunk_samples]
    features, hidden_states, vad_result, _ = audio_processor._process_chunk(
        chunk,
        return_hidden_states=True
    )

    # Verify VAD results
    assert isinstance(vad_result.is_speech, bool)
    assert isinstance(vad_result.confidence, float)
    assert isinstance(vad_result.energy_level, float)

    # Verify feature extraction with VAD
    assert features.shape[-1] == audio_processor.config.text_hidden_size
    assert len(hidden_states) == len(audio_processor.config.feature_layers)


def test_error_handling(audio_processor):
    """Test error handling in the processing pipeline."""
    # Test with invalid input shape
    invalid_input = torch.randn(3, 1000)  # Wrong batch dimension
    with pytest.raises(ValueError):
        audio_processor.process_stream(invalid_input)

    # Test with empty input
    empty_input = torch.tensor([])
    with pytest.raises(ValueError):
        audio_processor.process_stream(empty_input)

    # Test with wrong sample rate
    wrong_rate_input = torch.randn(1, 8000)  # 8kHz instead of 16kHz
    result = audio_processor.process_stream(wrong_rate_input)
    # Should handle this gracefully by resampling or raising appropriate error
