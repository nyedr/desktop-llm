"""Text-to-speech service using Kokoro TTS."""

import logging
from typing import AsyncGenerator, Dict, Any, Optional
import pyaudio
from openai import AsyncOpenAI
from app.utils.profiling import RequestProfile
import time

logger = logging.getLogger(__name__)


class TTSServiceError(Exception):
    """Base exception for TTS service errors."""
    pass


class TTSService:
    """Service for text-to-speech conversion using Kokoro TTS."""

    def __init__(self, base_url: str = "http://localhost:8880/v1"):
        """Initialize TTS service.

        Args:
            base_url: Base URL of the Kokoro TTS server
        """
        self.base_url = base_url
        self.client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
        self.audio_player = None
        self.chunk_size = 1024
        self.sample_rate = 24000
        self.channels = 1
        self.format = pyaudio.paInt16
        self._audio = None
        self._first_audio_played = False

    async def __aenter__(self):
        """Initialize service."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources."""
        self.cleanup()

    def _init_audio_player(self):
        """Initialize PyAudio player if not already initialized."""
        if not self.audio_player:
            try:
                if not self._audio:
                    self._audio = pyaudio.PyAudio()
                self.audio_player = self._audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    output=True,
                    start=True  # Start the stream immediately
                )
            except Exception as e:
                logger.error(f"Failed to initialize audio player: {e}")
                raise TTSServiceError(
                    f"Audio player initialization failed: {e}")

    async def get_available_voices(self) -> Dict[str, Any]:
        """Get list of available voices from Kokoro TTS.

        Returns:
            Dict containing available voices
        """
        try:
            response = await self.client.get("/audio/voices")
            return response
        except Exception as e:
            logger.error(f"Error getting voices: {e}")
            raise TTSServiceError(f"Failed to get voices: {e}")

    async def stream_speech(
        self,
        text: str,
        voice: str = "af_bella",
        play_audio: bool = False,
        response_format: str = "pcm",
        speed: float = 1.0,
        profiler: Optional[RequestProfile] = None
    ) -> AsyncGenerator[bytes, None]:
        """Stream speech from text using Kokoro TTS.

        Args:
            text: Text to convert to speech
            voice: Voice to use for synthesis
            play_audio: Whether to play audio through speakers
            response_format: Audio format (pcm, mp3, wav, etc)
            speed: Speech speed multiplier
            profiler: Optional request profiler for timing

        Yields:
            Audio data chunks
        """
        if not text.strip():
            return

        if play_audio:
            self._init_audio_player()
            self._first_audio_played = False

        try:
            async with self.client.audio.speech.with_streaming_response.create(
                model="kokoro",
                voice=voice,
                input=text,
                response_format=response_format,
                speed=speed
            ) as response:
                async for chunk in response.iter_bytes(chunk_size=self.chunk_size):
                    if chunk:
                        if play_audio and response_format == "pcm" and self.audio_player:
                            try:
                                self.audio_player.write(chunk)
                                if not self._first_audio_played and profiler:
                                    self._first_audio_played = True
                                    profiler.record_first_audio(text)
                            except Exception as e:
                                logger.error(f"Error playing audio chunk: {e}")
                        yield chunk

        except Exception as e:
            logger.error(f"Error streaming speech: {e}")
            raise TTSServiceError(f"Speech streaming failed: {e}")

    def cleanup(self):
        """Clean up audio resources."""
        if self.audio_player:
            try:
                self.audio_player.stop_stream()
                self.audio_player.close()
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
            self.audio_player = None

        if self._audio:
            try:
                self._audio.terminate()
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")
            self._audio = None
