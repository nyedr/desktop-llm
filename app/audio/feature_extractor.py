"""Audio feature extraction module."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple, Dict
from transformers import (
    Wav2Vec2Model, WhisperModel, Wav2Vec2Config, WhisperConfig,
    Wav2Vec2Processor, Wav2Vec2ForCTC, WhisperProcessor,
    WhisperForConditionalGeneration
)
from dataclasses import dataclass
from collections import deque
import time

from .config import AudioConfig
from .vad import VoiceActivityDetector, VADResult


@dataclass
class TranscriptionResult:
    """Container for transcription results."""
    text: str
    confidence: float
    start_time: float
    end_time: float
    is_final: bool = False


@dataclass
class AudioSegment:
    """Container for audio segment information."""
    features: torch.Tensor
    token_id: str
    start_idx: int
    length: int
    is_speech: bool = False  # Indicates if segment contains speech
    energy_level: float = 0.0  # Energy level of the segment
    hidden_states: Optional[List[torch.Tensor]] = None
    # Type of segment (speech, music, silence, etc.)
    segment_type: str = "general"
    timestamp: float = 0.0  # Timestamp of the segment
    duration: float = 0.0  # Duration of the segment in seconds
    # Transcription if available
    transcription: Optional[TranscriptionResult] = None


@dataclass
class ProcessedAudioChunk:
    """Container for processed audio chunk with metadata."""
    features: torch.Tensor
    hidden_states: Optional[List[torch.Tensor]]
    is_speech: bool
    energy_level: float
    segment_type: str
    timestamp: float
    duration: float
    is_ready: bool = False  # Whether chunk is ready for fusion
    # Transcription if available
    transcription: Optional[TranscriptionResult] = None


class AudioTranscriber:
    """Handles audio transcription."""

    def __init__(
        self,
        config: AudioConfig,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    ):
        """Initialize transcriber.

        Args:
            config: Audio configuration
            device: Device to run model on
        """
        self.config = config
        self.device = device

        # Initialize transcription model and processor
        if config.encoder_type == "wav2vec2":
            self.processor = Wav2Vec2Processor.from_pretrained(
                config.encoder_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(
                config.encoder_name).to(device)
        else:  # whisper
            self.processor = WhisperProcessor.from_pretrained(
                config.encoder_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                config.encoder_name
            ).to(device)

        self.model.eval()

    def transcribe(
        self,
        audio: torch.Tensor,
        start_time: float,
        duration: float,
        return_timestamps: bool = True
    ) -> TranscriptionResult:
        """Transcribe audio chunk.

        Args:
            audio: Audio tensor
            start_time: Start time of chunk
            duration: Duration of chunk
            return_timestamps: Whether to return word-level timestamps

        Returns:
            Transcription result
        """
        with torch.no_grad():
            if self.config.encoder_type == "wav2vec2":
                # Process audio
                inputs = self.processor(
                    audio.squeeze().cpu().numpy(),
                    sampling_rate=self.config.sample_rate,
                    return_tensors="pt"
                ).to(self.device)

                # Get logits
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predictions)[0]

                # Calculate confidence
                probs = F.softmax(logits, dim=-1)
                confidence = torch.mean(torch.max(probs, dim=-1)[0]).item()

            else:  # whisper
                # Process audio
                inputs = self.processor(
                    audio.squeeze().cpu().numpy(),
                    sampling_rate=self.config.sample_rate,
                    return_tensors="pt"
                ).to(self.device)

                # Generate transcription
                outputs = self.model.generate(
                    **inputs,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=448,
                    num_beams=5
                )

                # Get text and confidence
                transcription = self.processor.batch_decode(
                    outputs.sequences,
                    skip_special_tokens=True
                )[0]
                confidence = torch.mean(
                    torch.max(outputs.scores[0], dim=-1)[0]
                ).item()

            return TranscriptionResult(
                text=transcription.strip(),
                confidence=confidence,
                start_time=start_time,
                end_time=start_time + duration,
                is_final=True
            )


class StreamingAudioProcessor:
    """Manages real-time audio processing and synchronization."""

    def __init__(
        self,
        config: AudioConfig,
        feature_extractor: 'AudioFeatureExtractor',
        chunk_duration: float = 0.5,  # Duration of each chunk in seconds
        max_silence: float = 1.0,  # Maximum silence duration before triggering fusion
        min_speech_duration: float = 0.3,  # Minimum speech duration for a valid segment
        buffer_duration: float = 5.0  # Maximum duration to buffer
    ):
        """Initialize streaming processor.

        Args:
            config: Audio configuration
            feature_extractor: Feature extractor instance
            chunk_duration: Duration of each chunk in seconds
            max_silence: Maximum silence duration before triggering fusion
            min_speech_duration: Minimum speech duration for a valid segment
            buffer_duration: Maximum duration to buffer
        """
        self.config = config
        self.feature_extractor = feature_extractor
        self.chunk_duration = chunk_duration
        self.max_silence = max_silence
        self.min_speech_duration = min_speech_duration
        self.buffer_duration = buffer_duration

        # Initialize buffers
        self.chunk_buffer = deque()
        self.current_speech_duration = 0.0
        self.last_speech_time = 0.0
        self.silence_start_time = 0.0
        self.is_in_speech = False

        # Create streaming state
        self.feature_state = feature_extractor.create_streaming_state()

        # Initialize timing
        self.start_time = time.time()
        self.last_chunk_time = self.start_time

    def process_chunk(
        self,
        audio_chunk: torch.Tensor,
        force_flush: bool = False
    ) -> Optional[ProcessedAudioChunk]:
        """Process a single audio chunk.

        Args:
            audio_chunk: Audio chunk to process
            force_flush: Whether to force processing regardless of timing

        Returns:
            Processed chunk if ready, None otherwise
        """
        current_time = time.time()
        chunk_timestamp = current_time - self.start_time

        # Process chunk with feature extractor
        features, hidden_states, is_speech, energy_level = (
            self.feature_extractor._process_chunk(audio_chunk)
        )

        # Update speech tracking
        if is_speech:
            if not self.is_in_speech:
                self.is_in_speech = True
                self.current_speech_duration = 0.0
            self.current_speech_duration += self.chunk_duration
            self.last_speech_time = current_time
            self.silence_start_time = 0.0
        else:
            if self.is_in_speech:
                if self.silence_start_time == 0.0:
                    self.silence_start_time = current_time
            self.is_in_speech = False

        # Determine if chunk is ready for fusion
        is_ready = (
            force_flush or
            (self.current_speech_duration >= self.min_speech_duration and
             not self.is_in_speech and
             current_time - self.silence_start_time >= self.max_silence)
        )

        # Create processed chunk
        chunk = ProcessedAudioChunk(
            features=features,
            hidden_states=hidden_states,
            is_speech=is_speech,
            energy_level=energy_level,
            segment_type="speech" if is_speech else "silence",
            timestamp=chunk_timestamp,
            duration=self.chunk_duration,
            is_ready=is_ready
        )

        # Add to buffer
        self.chunk_buffer.append(chunk)

        # Trim buffer if needed
        while (len(self.chunk_buffer) * self.chunk_duration > self.buffer_duration):
            self.chunk_buffer.popleft()

        self.last_chunk_time = current_time
        return chunk if is_ready else None

    def get_buffered_features(
        self,
        clear_buffer: bool = True
    ) -> Optional[Tuple[torch.Tensor, List[torch.Tensor], List[Dict[str, float]]]]:
        """Get concatenated features from buffer.

        Args:
            clear_buffer: Whether to clear the buffer after getting features

        Returns:
            Tuple of (features, hidden_states, metadata) if buffer has content,
            None otherwise
        """
        if not self.chunk_buffer:
            return None

        # Collect features and metadata
        features_list = []
        hidden_states_list = [[]
                              for _ in range(len(self.config.feature_layers))]
        metadata_list = []

        for chunk in self.chunk_buffer:
            features_list.append(chunk.features)
            if chunk.hidden_states:
                for buffer, states in zip(hidden_states_list, chunk.hidden_states):
                    buffer.append(states)

            metadata_list.append({
                "is_speech": chunk.is_speech,
                "energy_level": chunk.energy_level,
                "segment_type": chunk.segment_type,
                "timestamp": chunk.timestamp,
                "duration": chunk.duration
            })

        # Concatenate features
        features = torch.cat(features_list, dim=1)

        # Concatenate hidden states if any
        hidden_states = None
        if hidden_states_list[0]:
            hidden_states = [
                torch.cat(states, dim=1) for states in hidden_states_list
            ]

        # Clear buffer if requested
        if clear_buffer:
            self.chunk_buffer.clear()
            self.current_speech_duration = 0.0
            self.is_in_speech = False
            self.silence_start_time = 0.0

        return features, hidden_states, metadata_list

    def should_trigger_fusion(self) -> bool:
        """Check if fusion should be triggered based on current state.

        Returns:
            Whether fusion should be triggered
        """
        current_time = time.time()

        # Check various conditions for triggering fusion
        long_silence = (
            not self.is_in_speech and
            self.silence_start_time > 0 and
            current_time - self.silence_start_time >= self.max_silence
        )

        complete_speech = (
            not self.is_in_speech and
            self.current_speech_duration >= self.min_speech_duration
        )

        buffer_full = (
            len(self.chunk_buffer) * self.chunk_duration >= self.buffer_duration
        )

        return long_silence or complete_speech or buffer_full


class StreamingFeatureState:
    """State management for streaming feature extraction."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 3,
        max_buffer_size: int = 32000,
        vad_threshold: float = 0.5,
        energy_threshold: float = 0.1
    ):
        """Initialize streaming state.

        Args:
            hidden_size: Hidden size of the features
            num_layers: Number of intermediate layers to track
            max_buffer_size: Maximum size of feature buffer in samples
            vad_threshold: Threshold for voice activity detection
            energy_threshold: Threshold for energy-based segmentation
        """
        self.features_buffer = []
        self.hidden_states_buffer = [[] for _ in range(num_layers)]
        self.hidden_size = hidden_size
        self.max_buffer_size = max_buffer_size
        self.vad_threshold = vad_threshold
        self.energy_threshold = energy_threshold
        self.is_final = False
        self.current_segment_type = "silence"
        self.last_features = None
        self.buffer_samples = 0

    def add_features(
        self,
        features: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None,
        is_final: bool = False,
        is_speech: bool = False,
        energy_level: float = 0.0
    ):
        """Add features to the buffer.

        Args:
            features: New features to add
            hidden_states: Optional intermediate layer features
            is_final: Whether this is the final chunk
            is_speech: Whether chunk contains speech
            energy_level: Energy level of the chunk
        """
        # Update segment type based on speech and energy
        if is_speech and energy_level > self.energy_threshold:
            self.current_segment_type = "speech"
        elif energy_level > self.energy_threshold * 0.5:
            self.current_segment_type = "audio"
        else:
            self.current_segment_type = "silence"

        # Add features to buffer
        self.features_buffer.append(features)
        self.buffer_samples += features.shape[1]
        self.last_features = features

        # Add hidden states if available
        if hidden_states:
            for buffer, states in zip(self.hidden_states_buffer, hidden_states):
                buffer.append(states)

        # Check if buffer exceeds max size
        if self.buffer_samples > self.max_buffer_size:
            self._trim_buffer()

        self.is_final = is_final

    def _trim_buffer(self):
        """Trim buffer to max size while preserving context."""
        # Keep the last max_buffer_size samples
        total_samples = 0
        keep_idx = 0

        for i in range(len(self.features_buffer) - 1, -1, -1):
            total_samples += self.features_buffer[i].shape[1]
            if total_samples > self.max_buffer_size:
                keep_idx = i
                break

        self.features_buffer = self.features_buffer[keep_idx:]
        self.buffer_samples = sum(f.shape[1] for f in self.features_buffer)

        # Also trim hidden states buffers
        for buffer in self.hidden_states_buffer:
            buffer[:] = buffer[keep_idx:]

    def get_concatenated_features(
        self,
        min_chunk_size: int = 0
    ) -> Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        """Get concatenated features if buffer is ready.

        Args:
            min_chunk_size: Minimum chunk size to return

        Returns:
            Tuple of (concatenated features, concatenated hidden states) if ready,
            (None, None) otherwise
        """
        if not self.features_buffer:
            return None, None

        # Check if we have enough samples
        total_samples = sum(f.shape[1] for f in self.features_buffer)
        if total_samples < min_chunk_size and not self.is_final:
            return None, None

        # Concatenate features
        features = torch.cat(self.features_buffer, dim=1)

        # Concatenate hidden states if available
        hidden_states = None
        if self.hidden_states_buffer[0]:
            hidden_states = [
                torch.cat(buffer, dim=1)
                for buffer in self.hidden_states_buffer
            ]

        # Clear buffers after concatenation
        self.features_buffer = []
        self.hidden_states_buffer = [[]
                                     for _ in range(len(self.hidden_states_buffer))]
        self.buffer_samples = 0

        return features, hidden_states


class AudioFeatureExtractor(nn.Module):
    """Handles audio feature extraction and processing."""

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    ):
        """Initialize feature extractor."""
        super().__init__()
        self.config = config or AudioConfig()
        self.device = device

        # Initialize encoder
        self.encoder = self._create_encoder().to(device)

        # Initialize transcriber if needed
        self.transcriber = AudioTranscriber(
            self.config, device) if self.config.transcribe else None

        # Initialize VAD
        self.vad = VoiceActivityDetector(self.config, device)

        # Initialize projection layers
        self.projections = nn.ModuleList([
            nn.Linear(self.encoder.config.hidden_size * self.config.stack_factor,
                      self.config.text_hidden_size)
            for _ in range(len(self.config.feature_layers) + 1)
        ])

        # Initialize frame stacker
        self.frame_stacker = StackAudioFrames(self.config.stack_factor)

        # Initialize layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.config.text_hidden_size)
            for _ in range(len(self.config.feature_layers) + 1)
        ])

        # Initialize dropout
        self.dropout = nn.Dropout(self.config.fusion_dropout)

        # Get chunking config
        chunk_config = self.config.chunking_config

        # Create streaming processor
        self.streaming_processor = StreamingAudioProcessor(
            config=self.config,
            feature_extractor=self,
            chunk_duration=chunk_config.chunk_duration,
            max_silence=chunk_config.max_silence,
            min_speech_duration=chunk_config.min_speech_duration,
            buffer_duration=5.0  # 5 seconds maximum buffer
        )

    def _process_chunk(
        self,
        chunk: torch.Tensor,
        return_hidden_states: bool = True,
        transcribe: bool = True
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], VADResult, Optional[TranscriptionResult]]:
        """Process single audio chunk."""
        # Move chunk to device
        chunk = chunk.to(self.device)

        # Get current time for transcription
        current_time = time.time()
        duration = chunk.shape[1] / self.config.sample_rate

        # Detect speech
        vad_result = self.vad.detect_speech(chunk)

        # Extract features with hidden states
        encoder_outputs = self.encoder(
            chunk,
            output_hidden_states=return_hidden_states,
            return_dict=True
        )

        # Get main features and process them
        # [batch_size, seq_len, hidden_size]
        features = encoder_outputs.last_hidden_state
        # [batch_size, seq_len/stack_factor, hidden_size*stack_factor]
        features = self.frame_stacker(features)
        # Get original hidden size from encoder
        encoder_hidden_size = self.encoder.config.hidden_size
        # Project stacked features to text dimension
        features = self.projections[0](features)
        features = self.layer_norms[0](features)
        features = self.dropout(features)

        # Process intermediate features if available
        hidden_states = None
        if return_hidden_states and hasattr(encoder_outputs, "hidden_states"):
            hidden_states = []
            for i, layer_idx in enumerate(self.config.feature_layers):
                layer_features = encoder_outputs.hidden_states[layer_idx]
                layer_features = self.frame_stacker(layer_features)
                # Project intermediate features
                layer_features = self.projections[i + 1](layer_features)
                layer_features = self.layer_norms[i + 1](layer_features)
                layer_features = self.dropout(layer_features)
                hidden_states.append(layer_features)

        # Transcribe if requested and speech is detected
        transcription = None
        if transcribe and self.transcriber and vad_result.is_speech:
            transcription = self.transcriber.transcribe(
                chunk,
                start_time=current_time,
                duration=duration
            )

        return features, hidden_states, vad_result, transcription

    def _create_encoder(self) -> nn.Module:
        """Create audio encoder model.

        Returns:
            Encoder model
        """
        if self.config.encoder_type == "wav2vec2":
            model_config = Wav2Vec2Config.from_pretrained(
                self.config.encoder_name)
            model_config.output_hidden_states = True
            return Wav2Vec2Model.from_pretrained(
                self.config.encoder_name,
                config=model_config
            )
        elif self.config.encoder_type == "whisper":
            model_config = WhisperConfig.from_pretrained(
                self.config.encoder_name)
            model_config.output_hidden_states = True
            return WhisperModel.from_pretrained(
                self.config.encoder_name,
                config=model_config
            ).encoder
        else:
            raise ValueError(
                f"Unknown encoder type: {self.config.encoder_type}")

    def create_streaming_state(self) -> StreamingFeatureState:
        """Create state for streaming feature extraction.

        Returns:
            Streaming state object
        """
        return StreamingFeatureState(
            hidden_size=self.config.text_hidden_size,
            num_layers=len(self.config.feature_layers)
        )

    def process_stream(
        self,
        audio_input: torch.Tensor,
        force_flush: bool = False
    ) -> Optional[Tuple[torch.Tensor, Optional[List[torch.Tensor]], List[Dict[str, float]], Optional[TranscriptionResult]]]:
        """Process streaming audio input."""
        # Validate input shape
        if audio_input.dim() != 2 or audio_input.size(0) != 1:
            raise ValueError(
                f"Expected audio input shape [1, T], got {audio_input.shape}")

        # Process chunk through streaming processor
        chunk = self.streaming_processor.process_chunk(
            audio_input, force_flush)

        # Check if we should trigger fusion
        if chunk and chunk.is_ready or self.streaming_processor.should_trigger_fusion():
            features, hidden_states, metadata = self.streaming_processor.get_buffered_features()
            # Get latest transcription if available
            transcription = next(
                (chunk.transcription for chunk in reversed(self.streaming_processor.chunk_buffer)
                 if chunk.transcription is not None),
                None
            )
            return features, hidden_states, metadata, transcription

        return None

    def process_segments(
        self,
        audio_inputs: List[torch.Tensor],
        token_ids: List[str],
        start_indices: List[int]
    ) -> List[AudioSegment]:
        """Process multiple audio segments."""
        segments = []
        current_time = time.time()

        for audio, token_id, start_idx in zip(audio_inputs, token_ids, start_indices):
            # Validate input shape
            if audio.dim() != 2 or audio.size(0) != 1:
                raise ValueError(
                    f"Expected audio input shape [1, T], got {audio.shape}")

            # Process audio with speech detection
            features, hidden_states, vad_result, transcription = self._process_chunk(
                audio,
                return_hidden_states=True
            )

            # Calculate duration
            duration = audio.shape[1] / self.config.sample_rate

            # Determine segment type based on VAD result
            segment_type = "speech" if vad_result.is_speech and vad_result.energy_level > 0.1 else (
                "audio" if vad_result.energy_level > 0.05 else "silence"
            )

            # Create segment
            segment = AudioSegment(
                features=features,
                token_id=token_id,
                start_idx=start_idx,
                length=features.size(1),
                is_speech=vad_result.is_speech,  # Already a bool
                energy_level=vad_result.energy_level,
                hidden_states=hidden_states,
                segment_type=segment_type,
                timestamp=current_time,
                duration=duration,
                transcription=transcription
            )
            segments.append(segment)
            current_time += duration

        return segments

    def forward(
        self,
        audio_input: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], List[Dict[str, float]]]:
        """Extract features from audio input.

        Args:
            audio_input: Audio waveform or list of chunks

        Returns:
            Tuple of (features, hidden_states, metadata)
        """
        # Handle list of chunks
        if isinstance(audio_input, list):
            features_list = []
            hidden_states_list = [[]
                                  for _ in range(len(self.config.feature_layers))]
            metadata_list = []
            current_time = time.time()

            for chunk in audio_input:
                # Process chunk with speech detection
                features, hidden_states, is_speech, energy_level = self._process_chunk(
                    chunk,
                    return_hidden_states=True
                )

                # Calculate duration
                duration = chunk.shape[1] / self.config.sample_rate

                features_list.append(features)
                metadata_list.append({
                    "is_speech": is_speech,
                    "energy_level": energy_level,
                    "length": features.size(1),
                    "timestamp": current_time,
                    "duration": duration,
                    "segment_type": "speech" if is_speech else "silence"
                })

                if hidden_states:
                    for buffer, states in zip(hidden_states_list, hidden_states):
                        buffer.append(states)

                current_time += duration

            # Concatenate features
            features = torch.cat(features_list, dim=1)

            # Concatenate hidden states if any
            hidden_states = [
                torch.cat(states, dim=1) for states in hidden_states_list
            ] if hidden_states_list[0] else None

            return features, hidden_states, metadata_list

        # Process single chunk
        features, hidden_states, is_speech, energy_level = self._process_chunk(
            audio_input,
            return_hidden_states=True
        )

        duration = audio_input.shape[1] / self.config.sample_rate
        metadata = [{
            "is_speech": is_speech,
            "energy_level": energy_level,
            "length": features.size(1),
            "timestamp": time.time(),
            "duration": duration,
            "segment_type": "speech" if is_speech else "silence"
        }]

        return features, hidden_states, metadata


class StackAudioFrames(nn.Module):
    """Stack consecutive audio frames."""

    def __init__(self, stack_factor: int):
        """Initialize frame stacker.

        Args:
            stack_factor: Number of frames to stack
        """
        super().__init__()
        self.stack_factor = stack_factor

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Stack audio frames.

        Args:
            features: Audio features [B, L, D]

        Returns:
            Stacked features [B, L/stack_factor, D*stack_factor]
        """
        B, L, D = features.shape

        # Ensure length is divisible by stack_factor
        new_L = L // self.stack_factor * self.stack_factor
        features = features[:, :new_L, :]

        # Reshape and stack
        stacked = features.view(
            B, -1, self.stack_factor, D
        ).transpose(2, 3).contiguous()

        return stacked.view(B, -1, D * self.stack_factor)
