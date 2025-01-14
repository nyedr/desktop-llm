"""Module for fusing audio and text features."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .config import AudioConfig
from .feature_extractor import AudioSegment


@dataclass
class FusedFeatures:
    """Container for fused features."""
    embeddings: torch.Tensor  # Combined text and audio embeddings
    # Full attention mask for text + audio
    attention_mask: Optional[torch.Tensor] = None
    # Mask specifically for audio regions
    audio_mask: Optional[torch.Tensor] = None
    is_final: bool = False
    intermediate_states: Optional[List[torch.Tensor]] = None


class StreamingFusionState:
    """State management for streaming fusion."""

    def __init__(
        self,
        text_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_layers: int = 3,
        audio_token_id: Optional[int] = None
    ):
        """Initialize streaming state.

        Args:
            text_embeddings: Base text embeddings
            attention_mask: Attention mask for text
            num_layers: Number of intermediate layers
            audio_token_id: ID of the special audio token
        """
        self.text_embeddings = text_embeddings
        self.attention_mask = attention_mask
        self.intermediate_states = [[] for _ in range(num_layers)]
        self.current_audio_idx = 0
        self.is_final = False
        self.audio_token_id = audio_token_id
        self.audio_indices = self._find_audio_tokens() if audio_token_id is not None else []

    def _find_audio_tokens(self) -> List[int]:
        """Find positions of audio tokens in text.

        Returns:
            List of audio token positions
        """
        # Assuming text_embeddings has a token_ids attribute or similar
        # This would need to be adapted based on your actual text embedding format
        if hasattr(self.text_embeddings, 'token_ids'):
            return [i for i, tid in enumerate(self.text_embeddings.token_ids)
                    if tid == self.audio_token_id]
        return []

    def get_next_audio_position(self) -> Optional[int]:
        """Get next position for audio insertion.

        Returns:
            Next audio token position or None if no more positions
        """
        if self.current_audio_idx < len(self.audio_indices):
            pos = self.audio_indices[self.current_audio_idx]
            self.current_audio_idx += 1
            return pos
        return None

    def update_embeddings(
        self,
        audio_features: torch.Tensor,
        intermediate_features: Optional[List[torch.Tensor]],
        start_idx: int,
        length: int,
        is_final: bool = False
    ):
        """Update embeddings with new audio features.

        Args:
            audio_features: New audio features
            intermediate_features: Features from intermediate layers
            start_idx: Starting index for insertion
            length: Length of features
            is_final: Whether this is the final update
        """
        length = min(length, audio_features.shape[1])

        # Update embeddings
        self.text_embeddings[:, start_idx:start_idx +
                             length] = audio_features[:, :length]

        # Update attention mask if it exists
        if self.attention_mask is not None:
            # Create audio attention region
            audio_attention = torch.ones(
                self.attention_mask.shape[0],
                length,
                device=self.attention_mask.device
            )
            self.attention_mask[:, start_idx:start_idx +
                                length] = audio_attention

        # Update intermediate states
        if intermediate_features:
            for state_buffer, features in zip(self.intermediate_states, intermediate_features):
                state_buffer.append(features[:, :length])

        self.is_final = is_final


class AudioTextFusion(nn.Module):
    """Module for fusing audio and text features."""

    def __init__(
        self,
        config: AudioConfig
    ):
        """Initialize fusion module."""
        super().__init__()
        self.config = config

        # Frame stacking
        self.stack_factor = config.stack_factor

        # Input normalization
        audio_dim = config.audio_hidden_size * config.stack_factor
        self.input_norm = RMSNorm(audio_dim)

        # Create projection layers for each feature level
        self.projection_layers = nn.ModuleList([
            self._create_projection_block()
            for _ in range(config.projection_layers)
        ])

        # Output normalization
        self.output_norm = RMSNorm(config.text_hidden_size, eps=1e-6)

        # Initialize norms
        self.input_norm.weight.data.fill_(config.norm_init)
        self.output_norm.weight.data.fill_(config.norm_init)

        # Dropout
        self.dropout = nn.Dropout(config.fusion_dropout)

    def _create_projection_block(self) -> nn.Sequential:
        """Create a projection block with optional SwiGLU.

        Returns:
            Projection block
        """
        if self.config.use_swiglu:
            return nn.Sequential(
                nn.Linear(
                    self.config.text_hidden_size,
                    self.config.text_hidden_size * 2,
                    bias=False
                ),
                SwiGLU(),
                nn.Linear(
                    self.config.text_hidden_size,
                    self.config.text_hidden_size,
                    bias=False
                ),
                nn.LayerNorm(
                    self.config.text_hidden_size) if self.config.use_layer_norm else nn.Identity(),
                nn.Dropout(self.config.fusion_dropout)
            )
        else:
            return nn.Sequential(
                nn.Linear(
                    self.config.text_hidden_size,
                    self.config.text_hidden_size,
                    bias=False
                ),
                nn.GELU(),
                nn.LayerNorm(
                    self.config.text_hidden_size) if self.config.use_layer_norm else nn.Identity(),
                nn.Dropout(self.config.fusion_dropout)
            )

    def create_streaming_state(
        self,
        text_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        audio_token_id: Optional[int] = None
    ) -> StreamingFusionState:
        """Create state for streaming fusion."""
        return StreamingFusionState(
            text_embeddings=text_embeddings.clone(),
            attention_mask=attention_mask,
            num_layers=len(self.config.feature_layers),
            audio_token_id=audio_token_id
        )

    def create_attention_mask(
        self,
        batch_size: int,
        seq_length: int,
        audio_start_idx: int,
        audio_length: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create attention mask for combined text and audio.

        Args:
            batch_size: Batch size
            seq_length: Total sequence length
            audio_start_idx: Start index of audio
            audio_length: Length of audio sequence
            device: Device to create tensor on

        Returns:
            Tuple of (attention_mask, audio_mask)
        """
        # Create full attention mask
        attention_mask = torch.zeros(batch_size, seq_length, device=device)

        # Set text regions to 1
        attention_mask[:, :audio_start_idx] = 1
        attention_mask[:, audio_start_idx + audio_length:] = 1

        # Set audio region to 1
        attention_mask[:, audio_start_idx:audio_start_idx + audio_length] = 1

        # Create specific audio mask
        audio_mask = torch.zeros_like(attention_mask)
        audio_mask[:, audio_start_idx:audio_start_idx + audio_length] = 1

        return attention_mask, audio_mask

    def process_stream(
        self,
        stream_state: StreamingFusionState,
        audio_features: torch.Tensor,
        intermediate_features: Optional[List[torch.Tensor]],
        start_idx: int,
        length: int,
        is_final: bool = False
    ) -> FusedFeatures:
        """Process streaming audio features.

        Args:
            stream_state: Current streaming state
            audio_features: New audio features
            intermediate_features: Features from intermediate layers
            start_idx: Starting index for insertion
            length: Length of features
            is_final: Whether this is the final chunk

        Returns:
            Fused features
        """
        # Get next audio position if using special tokens
        if stream_state.audio_token_id is not None:
            next_pos = stream_state.get_next_audio_position()
            if next_pos is not None:
                start_idx = next_pos

        # Process main features
        hidden_states = self.stack_frames(audio_features)
        hidden_states = self.input_norm(hidden_states)

        # Apply projection layers
        for layer in self.projection_layers:
            hidden_states = layer(hidden_states)

        # Process intermediate features if available
        processed_intermediates = None
        if intermediate_features:
            processed_intermediates = []
            for features in intermediate_features:
                inter_states = self.stack_frames(features)
                inter_states = self.input_norm(inter_states)
                for layer in self.projection_layers:
                    inter_states = layer(inter_states)
                processed_intermediates.append(inter_states)

        # Final normalization
        hidden_states = self.output_norm(hidden_states)

        # Create attention masks
        attention_mask, audio_mask = self.create_attention_mask(
            batch_size=hidden_states.shape[0],
            seq_length=stream_state.text_embeddings.shape[1],
            audio_start_idx=start_idx,
            audio_length=length,
            device=hidden_states.device
        )

        # Update state
        stream_state.update_embeddings(
            hidden_states,
            processed_intermediates,
            start_idx,
            length,
            is_final
        )

        return FusedFeatures(
            embeddings=stream_state.text_embeddings,
            attention_mask=attention_mask,
            audio_mask=audio_mask,
            is_final=stream_state.is_final,
            intermediate_states=processed_intermediates
        )

    def process_segments(
        self,
        text_embeddings: torch.Tensor,
        segments: List[AudioSegment],
        attention_mask: Optional[torch.Tensor] = None
    ) -> FusedFeatures:
        """Process multiple audio segments.

        Args:
            text_embeddings: Base text embeddings
            segments: List of audio segments
            attention_mask: Optional attention mask

        Returns:
            Fused features
        """
        # Create output embeddings
        output_embeddings = text_embeddings.clone()
        intermediate_states = [[]
                               for _ in range(len(self.config.feature_layers))]

        # Create audio mask
        audio_mask = torch.zeros_like(
            attention_mask) if attention_mask is not None else None

        # Process each segment
        for segment in segments:
            # Process main features
            hidden_states = self.stack_frames(segment.features)
            hidden_states = self.input_norm(hidden_states)

            for layer in self.projection_layers:
                hidden_states = layer(hidden_states)

            hidden_states = self.output_norm(hidden_states)

            # Process intermediate features if available
            if segment.hidden_states:
                for i, features in enumerate(segment.hidden_states):
                    inter_states = self.stack_frames(features)
                    inter_states = self.input_norm(inter_states)

                    for layer in self.projection_layers:
                        inter_states = layer(inter_states)

                    inter_states = self.output_norm(inter_states)
                    intermediate_states[i].append(inter_states)

            # Insert into output
            length = min(segment.length, hidden_states.shape[1])
            output_embeddings[:, segment.start_idx:segment.start_idx + length] = (
                hidden_states[:, :length]
            )

            # Update audio mask
            if audio_mask is not None:
                audio_mask[:, segment.start_idx:segment.start_idx + length] = 1

        # Concatenate intermediate states if any
        processed_intermediates = None
        if any(states for states in intermediate_states):
            processed_intermediates = [
                torch.cat(states, dim=1) for states in intermediate_states
            ]

        return FusedFeatures(
            embeddings=output_embeddings,
            attention_mask=attention_mask,
            audio_mask=audio_mask,
            intermediate_states=processed_intermediates
        )

    def stack_frames(
        self,
        features: torch.Tensor,
        padding_value: float = 0.0
    ) -> torch.Tensor:
        """Stack audio frames to reduce sequence length.

        Args:
            features: Audio features [B, T, D]
            padding_value: Value to use for padding

        Returns:
            Stacked features [B, T/stack_factor, D*stack_factor]
        """
        B, T, D = features.shape

        # Pad to multiple of stack_factor
        pad_len = (self.stack_factor - (T %
                   self.stack_factor)) % self.stack_factor
        if pad_len > 0:
            features = F.pad(features, (0, 0, 0, pad_len), value=padding_value)
            T = T + pad_len

        # Reshape and stack
        features = features.view(
            B, T // self.stack_factor, self.stack_factor, D
        ).transpose(2, 3).contiguous()
        return features.view(B, T // self.stack_factor, D * self.stack_factor)

    def forward(
        self,
        audio_features: torch.Tensor,
        intermediate_features: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_start_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        """Fuse audio and text features.

        Args:
            audio_features: Audio features [B, T, D]
            intermediate_features: Optional intermediate layer features
            attention_mask: Optional attention mask
            audio_start_idx: Optional start index for audio features

        Returns:
            Tuple of (fused features, attention mask, audio mask, intermediate states)
        """
        # Stack frames
        hidden_states = self.stack_frames(audio_features)

        # Update attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask[:, ::self.stack_factor]

        # Process main features
        hidden_states = self.input_norm(hidden_states)
        for layer in self.projection_layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.output_norm(hidden_states)

        # Process intermediate features if available
        processed_intermediates = None
        if intermediate_features:
            processed_intermediates = []
            for features in intermediate_features:
                inter_states = self.stack_frames(features)
                inter_states = self.input_norm(inter_states)
                for layer in self.projection_layers:
                    inter_states = layer(inter_states)
                inter_states = self.output_norm(inter_states)
                processed_intermediates.append(inter_states)

        # Create audio mask if start index is provided
        audio_mask = None
        if audio_start_idx is not None:
            _, audio_mask = self.create_attention_mask(
                batch_size=hidden_states.shape[0],
                seq_length=attention_mask.shape[1] if attention_mask is not None else hidden_states.shape[1],
                audio_start_idx=audio_start_idx,
                audio_length=hidden_states.shape[1],
                device=hidden_states.device
            )

        return hidden_states, attention_mask, audio_mask, processed_intermediates


class RMSNorm(nn.Module):
    """RMSNorm layer for stable normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """Initialize RMSNorm.

        Args:
            hidden_size: Size of hidden dimension
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm.

        Args:
            hidden_states: Input tensor

        Returns:
            Normalized tensor
        """
        variance = hidden_states.to(torch.float32).pow(
            2).mean(-1, keepdim=True)
        hidden_states = hidden_states * \
            torch.rsqrt(variance + self.variance_epsilon)

        # Convert back to the original dtype
        return self.weight * hidden_states.to(self.weight.dtype)


class SwiGLU(nn.Module):
    """SwiGLU activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x
