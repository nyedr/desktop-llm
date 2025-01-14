"""Voice Activity Detection module."""
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC
from dataclasses import dataclass

from .config import AudioConfig


@dataclass
class VADResult:
    """Container for VAD results."""
    is_speech: bool
    confidence: float
    energy_level: float


class VoiceActivityDetector:
    """Handles voice activity detection using multiple methods."""

    def __init__(
        self,
        config: AudioConfig,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    ):
        """Initialize VAD.

        Args:
            config: Audio configuration
            device: Device to run model on
        """
        self.config = config
        self.device = device
        self.vad_model = None

        # Initialize VAD model if using wav2vec2
        if config.encoder_type == "wav2vec2":
            self.vad_model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-base"
            ).to(device)
            self.vad_model.eval()

        # Energy detection thresholds
        self.energy_threshold = 0.1
        self.speech_threshold = 0.5

    def detect_speech(
        self,
        audio: torch.Tensor,
        use_model: bool = True
    ) -> VADResult:
        """Detect speech in audio input.

        Args:
            audio: Audio input tensor
            use_model: Whether to use model-based VAD

        Returns:
            VAD result containing speech detection and confidence
        """
        # Calculate energy level
        energy_level = torch.mean(audio.pow(2)).item()

        # Use model-based VAD if available and requested
        if use_model and self.vad_model is not None:
            with torch.no_grad():
                outputs = self.vad_model(audio)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                # Assuming the last token is for non-speech
                speech_prob = 1 - probs[:, :, -1].mean()
                is_speech = speech_prob > self.speech_threshold
                return VADResult(
                    is_speech=bool(is_speech.item()),
                    confidence=speech_prob.item(),
                    energy_level=energy_level
                )

        # Fallback to energy-based detection
        is_speech = energy_level > self.energy_threshold
        return VADResult(
            is_speech=bool(is_speech),
            confidence=energy_level,
            energy_level=energy_level
        )

    def get_segment_type(
        self,
        vad_result: VADResult
    ) -> str:
        """Determine segment type based on VAD result.

        Args:
            vad_result: VAD result

        Returns:
            Segment type (speech, audio, silence)
        """
        if vad_result.is_speech and vad_result.energy_level > self.energy_threshold:
            return "speech"
        elif vad_result.energy_level > self.energy_threshold * 0.5:
            return "audio"
        return "silence"
