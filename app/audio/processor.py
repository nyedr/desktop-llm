"""Audio processing module for handling audio input and preprocessing."""
import torch
import torchaudio
import numpy as np
import librosa
from typing import Optional, Tuple, Union, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from .config import AudioConfig


@dataclass
class AudioFeatures:
    """Container for processed audio features."""
    values: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    token_len: Optional[int] = None
    token_start_idx: Optional[int] = None
    audio_len: Optional[torch.Tensor] = None
    is_final: bool = False  # Indicates if this is the final chunk


class StreamingAudioState:
    """State management for streaming audio processing."""

    def __init__(
        self,
        max_length: int,
        chunk_size: int,
        overlap: int = 0
    ):
        """Initialize streaming state.

        Args:
            max_length: Maximum buffer length in samples
            chunk_size: Size of each chunk in samples
            overlap: Overlap between chunks in samples
        """
        self.buffer = torch.zeros(1, 0)
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.processed_samples = 0
        self.is_final = False

    def add_audio(self, audio: torch.Tensor, is_final: bool = False) -> bool:
        """Add audio to the buffer.

        Args:
            audio: New audio samples
            is_final: Whether this is the final chunk

        Returns:
            Whether a new chunk is ready for processing
        """
        self.buffer = torch.cat([self.buffer, audio], dim=-1)
        self.is_final = is_final
        return self.has_full_chunk()

    def has_full_chunk(self) -> bool:
        """Check if there's a full chunk ready."""
        if self.is_final:
            return self.buffer.size(-1) > 0
        return self.buffer.size(-1) >= self.chunk_size

    def get_next_chunk(self) -> Optional[Tuple[torch.Tensor, bool]]:
        """Get the next chunk if available.

        Returns:
            Tuple of (chunk, is_final) or None if no chunk is ready
        """
        if not self.has_full_chunk():
            return None

        if self.is_final:
            # Process remaining audio
            chunk = self.buffer
            self.buffer = torch.zeros(1, 0)
            return chunk, True

        # Get chunk with overlap
        chunk = self.buffer[:, :self.chunk_size]
        self.buffer = self.buffer[:, self.chunk_size - self.overlap:]
        self.processed_samples += self.chunk_size - self.overlap
        return chunk, False


class AudioProcessor:
    """Handles audio loading and preprocessing with streaming support."""

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        audio_padding: str = "longest",
        encoder_ds_factor: int = 320
    ):
        """Initialize audio processor.

        Args:
            config: Audio processing configuration
            audio_padding: Padding strategy ('longest' or 'max_length')
            encoder_ds_factor: Encoder downsampling factor
        """
        self.config = config or AudioConfig()
        self.audio_padding = audio_padding
        self.encoder_ds_factor = encoder_ds_factor
        self._setup_resampler()

    def _setup_resampler(self):
        """Initialize resampler if needed."""
        self.resampler = None

    def create_streaming_state(
        self,
        chunk_duration: float = 5.0,
        overlap_duration: float = 0.5
    ) -> StreamingAudioState:
        """Create state for streaming audio processing.

        Args:
            chunk_duration: Duration of each chunk in seconds
            overlap_duration: Duration of overlap between chunks in seconds

        Returns:
            Streaming state object
        """
        chunk_size = int(chunk_duration * self.config.sample_rate)
        overlap = int(overlap_duration * self.config.sample_rate)
        max_length = self.config.max_audio_length * self.config.sample_rate

        return StreamingAudioState(
            max_length=max_length,
            chunk_size=chunk_size,
            overlap=overlap
        )

    def process_stream(
        self,
        stream_state: StreamingAudioState,
        new_audio: Union[torch.Tensor, np.ndarray],
        is_final: bool = False
    ) -> Optional[AudioFeatures]:
        """Process streaming audio input.

        Args:
            stream_state: Current streaming state
            new_audio: New audio samples
            is_final: Whether this is the final chunk

        Returns:
            Processed audio features if a chunk is ready, None otherwise
        """
        # Normalize and convert input
        if isinstance(new_audio, np.ndarray):
            new_audio = self.normalize_audio(new_audio)

        # Add to buffer
        has_chunk = stream_state.add_audio(new_audio, is_final)
        if not has_chunk:
            return None

        # Get next chunk
        chunk_data = stream_state.get_next_chunk()
        if chunk_data is None:
            return None

        chunk, is_final_chunk = chunk_data

        # Process the chunk
        features = self.process_audio(
            audio=chunk,
            sample_rate=self.config.sample_rate,
            return_attention_mask=True
        )
        features.is_final = is_final_chunk
        return features

    def load_audio(
        self,
        file_path: Union[str, Path],
        normalize: bool = True
    ) -> Tuple[torch.Tensor, int]:
        """Load and preprocess audio file.

        Args:
            file_path: Path to audio file
            normalize: Whether to normalize audio

        Returns:
            Tuple of (waveform, sample_rate)
        """
        # Load audio file
        waveform, sample_rate = torchaudio.load(str(file_path))

        # Resample if needed
        if sample_rate != self.config.sample_rate:
            if self.resampler is None or self.resampler.orig_freq != sample_rate:
                self.resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.config.sample_rate
                )
            waveform = self.resampler(waveform)
            sample_rate = self.config.sample_rate

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Normalize if requested
        if normalize:
            waveform = self.normalize_audio(waveform)

        return waveform, sample_rate

    def normalize_audio(
        self,
        waveform: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """Normalize audio to float32 in [-1, 1] range.

        Args:
            waveform: Audio waveform

        Returns:
            Normalized waveform
        """
        if isinstance(waveform, np.ndarray):
            if waveform.dtype == np.int16:
                waveform = waveform / np.float32(32768.0)
            waveform = torch.from_numpy(waveform)

        return (waveform - self.config.audio_mean) / self.config.audio_std

    def process_audio(
        self,
        audio: Union[str, Path, torch.Tensor, np.ndarray],
        sample_rate: Optional[int] = None,
        return_attention_mask: bool = True
    ) -> AudioFeatures:
        """Process audio for model input.

        Args:
            audio: Audio input (file path or tensor/array)
            sample_rate: Sample rate of input audio
            return_attention_mask: Whether to return attention mask

        Returns:
            Processed audio features
        """
        # Handle file path input
        if isinstance(audio, (str, Path)):
            waveform, sample_rate = self.load_audio(audio)
        else:
            waveform = audio
            if sample_rate is None:
                sample_rate = self.config.sample_rate

            # Normalize numpy arrays
            if isinstance(waveform, np.ndarray):
                waveform = self.normalize_audio(waveform)

            # Resample if needed
            if sample_rate != self.config.sample_rate:
                waveform = librosa.resample(
                    waveform,
                    orig_sr=sample_rate,
                    target_sr=self.config.sample_rate
                )
                if isinstance(waveform, np.ndarray):
                    waveform = torch.from_numpy(waveform)

        # Handle padding
        if self.audio_padding == "max_length":
            audio_len = 30 * self.config.sample_rate  # 30 seconds max
            if waveform.shape[-1] < audio_len:
                pad_amount = audio_len - waveform.shape[-1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        else:
            audio_len = waveform.shape[-1]

        # Calculate number of frames
        nb_encoder_frames = int(
            round(audio_len / self.encoder_ds_factor + 1e-4))
        audio_embed_frames = int(
            np.ceil(nb_encoder_frames / self.config.stack_factor))

        # Create attention mask if requested
        attention_mask = None
        if return_attention_mask:
            if self.audio_padding == "max_length":
                orig_len = min(waveform.shape[-1], audio_len)
                attention_mask = torch.ones(audio_len, dtype=torch.long)
                attention_mask[orig_len:] = 0
            else:
                attention_mask = torch.ones(audio_len, dtype=torch.long)

        return AudioFeatures(
            values=waveform.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(
                0) if attention_mask is not None else None,
            token_len=audio_embed_frames,
            audio_len=torch.tensor([waveform.shape[-1]])
        )

    def chunk_audio(
        self,
        waveform: torch.Tensor,
        chunk_size: Optional[int] = None
    ) -> list[torch.Tensor]:
        """Split audio into chunks for streaming.

        Args:
            waveform: Audio waveform
            chunk_size: Size of chunks in samples

        Returns:
            List of audio chunks
        """
        if chunk_size is None:
            chunk_size = int(self.config.chunk_length *
                             self.config.sample_rate)

        # Split into chunks
        chunks = torch.split(waveform, chunk_size, dim=-1)

        # Remove last chunk if too short
        if chunks[-1].shape[-1] < chunk_size:
            chunks = chunks[:-1]

        return list(chunks)

    def prepare_streaming_state(
        self,
        max_audio_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """Prepare state for streaming audio processing.

        Args:
            max_audio_length: Maximum audio length in samples

        Returns:
            Initial streaming state
        """
        return {
            "buffer": torch.zeros(1, 0),
            "max_length": max_audio_length or (30 * self.config.sample_rate),
            "processed_samples": 0
        }
