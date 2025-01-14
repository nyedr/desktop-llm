"""Audio processing configuration."""
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class ChunkingPreset(BaseModel):
    """Preset configurations for different latency requirements."""
    chunk_duration: float
    overlap_ratio: float
    min_speech_duration: float
    max_silence: float

    @classmethod
    def fast(cls) -> 'ChunkingPreset':
        """Fast response preset (0.5s chunks)."""
        return cls(
            chunk_duration=0.5,
            overlap_ratio=0.1,  # 50ms overlap
            min_speech_duration=0.2,
            max_silence=0.3
        )

    @classmethod
    def balanced(cls) -> 'ChunkingPreset':
        """Balanced preset (2s chunks)."""
        return cls(
            chunk_duration=2.0,
            overlap_ratio=0.15,  # 300ms overlap
            min_speech_duration=0.3,
            max_silence=0.5
        )

    @classmethod
    def quality(cls) -> 'ChunkingPreset':
        """High quality preset (5s chunks)."""
        return cls(
            chunk_duration=5.0,
            overlap_ratio=0.2,  # 1s overlap
            min_speech_duration=0.5,
            max_silence=1.0
        )


class AudioConfig(BaseModel):
    """Configuration for audio processing."""

    # Audio processing parameters
    sample_rate: int = Field(
        default=16000,
        description="Target sample rate in Hz"
    )
    latency_preset: Literal["fast", "balanced", "quality"] = Field(
        default="balanced",
        description="Preset for latency-quality tradeoff"
    )
    chunk_duration: Optional[float] = Field(
        default=None,
        description="Override chunk duration in seconds"
    )
    overlap_ratio: Optional[float] = Field(
        default=None,
        description="Override overlap ratio (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    max_audio_length: int = Field(
        default=30,
        description="Maximum audio length in seconds"
    )

    # Feature extraction parameters
    stack_factor: int = Field(
        default=4,
        description="Number of frames to stack"
    )
    audio_hidden_size: int = Field(
        default=1024,
        description="Hidden size for audio features"
    )
    text_hidden_size: int = Field(
        default=4096,
        description="Hidden size for text features"
    )
    encoder_type: Literal["wav2vec2", "whisper"] = Field(
        default="wav2vec2",
        description="Type of audio encoder to use"
    )
    encoder_name: str = Field(
        default="facebook/wav2vec2-base",
        description="Name of pretrained encoder model"
    )
    feature_layers: list[int] = Field(
        default=[6, 9, 12],
        description="Layers to extract features from"
    )
    use_specaugment: bool = Field(
        default=True,
        description="Whether to use SpecAugment for feature extraction"
    )

    # Transcription parameters
    transcribe: bool = Field(
        default=True,
        description="Whether to transcribe audio input"
    )
    transcription_model: Optional[str] = Field(
        default=None,
        description="Name of transcription model to use. If None, uses encoder_name"
    )

    # Normalization parameters
    audio_mean: float = Field(
        default=0.0,
        description="Mean for audio normalization"
    )
    audio_std: float = Field(
        default=1.0,
        description="Standard deviation for audio normalization"
    )
    norm_init: float = Field(
        default=1.0,
        description="Initialization value for normalization layers"
    )

    # Fusion parameters
    use_swiglu: bool = Field(
        default=True,
        description="Whether to use SwiGLU activation in fusion"
    )
    projection_layers: int = Field(
        default=2,
        description="Number of projection layers in fusion"
    )
    fusion_dropout: float = Field(
        default=0.1,
        description="Dropout rate in fusion layers"
    )
    use_layer_norm: bool = Field(
        default=True,
        description="Whether to use layer normalization"
    )

    # Special tokens
    audio_token: str = Field(
        default="<|audio|>",
        description="Special token for audio segments"
    )
    pad_token: str = Field(
        default="<|pad|>",
        description="Padding token"
    )
    bos_token: str = Field(
        default="<|startoftext|>",
        description="Beginning of sequence token"
    )
    eos_token: str = Field(
        default="<|endoftext|>",
        description="End of sequence token"
    )

    # Streaming parameters
    stream_chunk_size: int = Field(
        default=8000,
        description="Size of streaming chunks in samples"
    )
    stream_buffer_size: int = Field(
        default=32000,
        description="Size of streaming buffer in samples"
    )
    stream_overlap: int = Field(
        default=4000,
        description="Overlap between streaming chunks in samples"
    )
    max_context_length: Optional[int] = Field(
        default=None,
        description="Maximum context length for streaming"
    )

    @field_validator('overlap_ratio')
    def validate_overlap_ratio(cls, v):
        """Validate overlap ratio is between 0 and 1."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("overlap_ratio must be between 0 and 1")
        return v

    @property
    def chunking_config(self) -> ChunkingPreset:
        """Get chunking configuration based on preset or overrides."""
        preset = getattr(ChunkingPreset, self.latency_preset)()
        if self.chunk_duration is not None:
            preset.chunk_duration = self.chunk_duration
        if self.overlap_ratio is not None:
            preset.overlap_ratio = self.overlap_ratio
        return preset

    class Config:
        """Pydantic config."""
        frozen = True
