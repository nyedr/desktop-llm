"""Pytest configuration and shared fixtures."""
import pytest
import torch
import numpy as np
from pathlib import Path
from app.audio.config import AudioConfig


@pytest.fixture(scope="session")
def test_dir():
    """Get test directory path."""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def device():
    """Get torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def sample_rate():
    """Get sample rate for testing."""
    return 16000


@pytest.fixture(scope="session")
def base_config():
    """Create base audio configuration."""
    return AudioConfig(
        encoder_type="wav2vec2",
        latency_preset="fast",  # Use fast preset for quicker tests
        feature_layers=[6, 9, 12],
        stack_factor=4,
        text_hidden_size=768
    )


@pytest.fixture
def create_audio():
    """Factory fixture to create test audio."""
    def _create_audio(duration=1.0, sample_rate=16000, frequencies=None):
        """Create test audio with specified parameters.

        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            frequencies: List of frequencies to include

        Returns:
            Tensor of shape [1, duration*sample_rate]
        """
        if frequencies is None:
            frequencies = [100, 200, 400, 800]

        t = np.linspace(0, duration, int(duration * sample_rate))
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

    return _create_audio


@pytest.fixture
def create_silence():
    """Factory fixture to create silent audio."""
    def _create_silence(duration=1.0, sample_rate=16000):
        """Create silent audio.

        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            Tensor of shape [1, duration*sample_rate]
        """
        samples = int(duration * sample_rate)
        return torch.zeros(1, samples)

    return _create_silence
