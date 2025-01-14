"""
Audio processing package for Desktop LLM.
Provides audio processing, feature extraction, and multimodal fusion capabilities.
"""

from .processor import AudioProcessor
from .feature_extractor import AudioFeatureExtractor
from .config import AudioConfig

__all__ = ['AudioProcessor', 'AudioFeatureExtractor', 'AudioConfig']
