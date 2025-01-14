# Audio Processing System Documentation

## Overview

The Audio Processing System is a specialized component designed to handle real-time audio processing and feature extraction for large language model (LLM) interactions. Its primary purpose is to process streaming audio input, extract rich features that capture both linguistic and paralinguistic aspects of speech, and prepare this information in a format suitable for LLM consumption.

## Core Responsibilities

1. **Real-time Audio Processing**

   - Handle streaming audio input in real-time
   - Process audio chunks efficiently with minimal latency
   - Maintain state across streaming chunks
   - Extract features while preserving temporal relationships

2. **Feature Extraction**

   - Extract rich audio features from multiple encoder layers
   - Capture both linguistic and paralinguistic aspects of speech
   - Provide intermediate features for nuanced understanding
   - Handle frame stacking for efficient processing

3. **Audio-Text Integration**
   - Fuse audio features with text embeddings
   - Manage attention masks for audio-text interaction
   - Handle special audio tokens in text
   - Provide proper context for LLM understanding

## Key Features

1. **Rich Audio Feature Extraction**

   - Multi-layer feature extraction from encoder (Wav2Vec2 or Whisper)
     - Hidden states from multiple transformer layers (typically layers 6, 9, 12)
     - Capture different levels of audio understanding (phonetic, semantic, contextual)
     - Specialized processing for conversational features
   - Frame stacking for temporal context
     - Configurable stack factor (default: 4 frames)
     - Reduces sequence length while preserving temporal relationships
     - Maintains feature resolution for important audio events
   - Intermediate feature processing
     - Layer-specific projections to match text embedding dimensions
     - Normalization and dropout for better generalization
     - Separate processing paths for main and intermediate features
   - Energy level and speech detection
     - VAD (Voice Activity Detection) using Wav2Vec2
     - Energy-based detection as fallback
     - Segment classification (speech, silence, general audio)
     - Metadata tracking for each segment

2. **Streaming Capabilities**

   - Chunk-based processing
     - Configurable chunk size based on latency requirements:
       - Fast response (0.5-1s chunks): Quicker partial results, higher processing overhead
       - Balanced (2-3s chunks): Good compromise between latency and processing
       - High quality (4-5s chunks): Better feature quality, higher latency
     - Overlap ratio typically 10-20% of chunk size:
       - Short chunks (0.5s): 0.05-0.1s overlap
       - Medium chunks (2s): 0.2-0.4s overlap
       - Long chunks (5s): 0.5-1s overlap
     - Latency-quality tradeoffs:
       - Smaller chunks: Lower latency, potentially less stable features
       - Larger chunks: Better feature stability, higher latency
       - Consider downstream model requirements
   - State management across chunks
     - Streaming state objects for features and fusion
     - Buffer management with configurable sizes
     - Context preservation across chunk boundaries
   - Buffer management for continuous processing
     - Circular buffers for feature storage
     - Automatic trimming of old data
     - Configurable maximum buffer duration
   - Real-time feature extraction
     - Immediate processing of incoming chunks
     - Parallel processing of features and transcription
     - Dynamic feature concatenation

3. **Audio-Text Fusion**

   - Dynamic audio feature insertion
     - Precise positioning using special <|audio|> tokens
     - Support for multiple audio segments in text
     - Proper handling of attention flow between modalities
   - Attention mask generation
     - Bidirectional attention between text and audio
     - Separate masks for audio regions
     - Proper context window management
   - Special token handling
     - Audio token detection and tracking
     - Position management for streaming insertion
     - Context preservation around audio segments
   - Multi-layer feature integration
     - Parallel processing of main and intermediate features
     - Layer-specific projections and normalization
     - Proper alignment with text embeddings
   - Feature Processing Pipeline
     ```
     Audio Features
     └── Frame Stacking
         └── Input Normalization (RMSNorm)
             └── Projection Layers
                 └── SwiGLU/GELU Activation
                     └── Layer Normalization
                         └── Dropout
                             └── Output Normalization
     ```

4. **Configurable Processing**
   - Adjustable chunk sizes and overlap
   - Configurable model selection
   - Tunable processing parameters
   - Flexible feature extraction options

## System Pipeline

1. **Input Processing**

   ```
   Raw Audio Stream
   └── Chunking (5s chunks, 0.5s overlap)
       └── Feature Extraction
           ├── Main Features (last hidden layer)
           └── Intermediate Features (layers 6,9,12)
               └── Feature Processing
                   ├── Frame Stacking (4 frames)
                   ├── Projection (to text dimension)
                   └── Normalization
                       └── Fusion
                           ├── Text Integration
                           └── Mask Generation
                               └── LLM-Ready Format
   ```

2. **Feature Extraction Flow**

   ```
   Audio Chunk → Encoder → Multi-layer Features → Frame Stacking → Projection → Normalization
   ```

3. **Fusion Flow**
   ```
   Audio Features + Text Embeddings → Position Detection → Feature Insertion → Mask Generation → Combined Output
   ```

## File Structure and Responsibilities

### 1. `config.py`

- **Purpose**: Central configuration management
- **Responsibilities**:
  - Define processing parameters
  - Set model configurations
  - Configure feature extraction settings
  - Define special tokens
  - Set streaming parameters

### 2. `feature_extractor.py`

- **Purpose**: Core audio processing and feature extraction
- **Responsibilities**:
  - Manage audio encoders
  - Extract multi-layer features
  - Handle streaming state
  - Process audio chunks
  - Provide frame stacking
  - Manage transcription

### 3. `fusion.py`

- **Purpose**: Audio-text feature integration
- **Responsibilities**:
  - Combine audio and text features
  - Generate attention masks
  - Handle streaming fusion
  - Manage intermediate features
  - Process multiple segments

### 4. `processor.py`

- **Purpose**: High-level audio processing interface
- **Responsibilities**:
  - Coordinate feature extraction
  - Manage streaming processing
  - Handle audio normalization
  - Provide chunk management
  - Maintain processing state

## Component Interactions

1. **Initialization Flow**

   ```
   AudioConfig → AudioProcessor → FeatureExtractor → AudioTextFusion
   ```

2. **Processing Flow**

   ```
   AudioProcessor
   ├── Chunks audio
   ├── Normalizes input
   └── Calls FeatureExtractor
       ├── Extracts features
       ├── Stacks frames
       └── Calls AudioTextFusion
           ├── Fuses with text
           └── Generates masks
   ```

3. **State Management Flow**
   ```
   StreamingState
   ├── Manages audio buffers
   ├── Tracks feature states
   └── Coordinates with fusion
   ```

## Integration Points

1. **Input Integration**

   - Accepts raw audio streams
   - Handles various audio formats
   - Supports chunk-based input
   - Manages streaming buffers

2. **Output Integration**

   - Provides LLM-ready features
   - Generates appropriate masks
   - Includes metadata and context
   - Supports streaming output

3. **External System Integration**
   - Interfaces with LLM systems
   - Supports external transcription
   - Provides feature access points
   - Enables state monitoring

## Performance Considerations

1. **Latency Management**

   - Chunk size optimization:
     - Monitor end-to-end latency metrics
     - Track processing time per chunk
     - Measure feature stability across chunk sizes
     - Consider adaptive chunk sizing based on load
   - Buffer management strategies:
     - Pre-allocate buffers for known chunk sizes
     - Implement circular buffers for streaming
     - Clear old data promptly
     - Monitor memory usage
   - Processing optimizations:
     - Parallel processing of overlapping chunks
     - Batch processing when possible
     - GPU acceleration for feature extraction
     - Efficient tensor operations

2. **Resource Usage**

   - Controlled memory footprint
   - Efficient tensor operations
   - Optimized model loading
   - Careful state management

3. **Scalability**
   - Independent processing units
   - Configurable resource usage
   - Flexible deployment options
   - Modular architecture

## Future Enhancements

1. **Feature Extraction**

   - Enhanced prosodic feature extraction
   - Improved emotional detection
   - Better speaker characteristics
   - Advanced audio segmentation

2. **Processing Pipeline**

   - Optimized chunk processing
   - Enhanced streaming capabilities
   - Improved state management
   - Better resource utilization

3. **Integration Capabilities**
   - Enhanced LLM integration
   - Improved transcription options
   - Better metadata handling
   - Advanced feature fusion

## Implementation Guidelines

1. **Audio Processing**

   - Use 16kHz sampling rate for compatibility with most models
   - Configure chunk duration based on use case:
     - Real-time transcription: 0.5-1s chunks
     - Interactive conversation: 1-2s chunks
     - Rich feature extraction: 2-5s chunks
   - Overlap considerations:
     - Minimum: 10% of chunk size
     - Optimal: 15-20% for feature stability
     - Maximum: 25% (higher creates unnecessary overhead)
   - Chunk size selection factors:
     - End-to-end latency requirements
     - Available computational resources
     - Model feature stability needs
     - Network bandwidth constraints
   - Performance impact:
     - Smaller chunks: Higher CPU usage, more network calls
     - Larger chunks: Better batching, more memory usage
     - Consider monitoring and auto-scaling needs
   - Normalize audio to zero mean and unit variance
   - Apply proper padding for chunk boundaries

2. **Feature Extraction**

   - Extract features from specific layers (6, 9, 12 by default)
   - Stack 4 frames together to reduce sequence length
   - Project all features to match text embedding dimension
   - Apply layer normalization and dropout

3. **Fusion Process**

   - Use special tokens to mark audio positions
   - Generate proper attention masks for bidirectional attention
   - Maintain state for streaming fusion
   - Handle multiple audio segments properly

4. **Performance Optimization**
   - Use batch processing where possible
   - Implement proper buffer management
   - Optimize tensor operations
   - Use efficient state management

## Integration Details

### Audio Feature Integration

1. **Feature Extraction Process**

   ```python
   # Example feature extraction flow
   audio_chunk = chunk_audio(raw_audio, duration=5.0, overlap=0.5)
   features = encoder(audio_chunk)

   # Multi-layer feature extraction
   main_features = features.last_hidden_state
   intermediate_features = [
       features.hidden_states[layer]
       for layer in [6, 9, 12]
   ]

   # Feature processing
   processed_features = {
       'main': process_features(main_features),
       'intermediate': [
           process_features(feat)
           for feat in intermediate_features
       ]
   }
   ```

2. **Frame Stacking Implementation**

   ```python
   # Frame stacking process
   def stack_frames(features, stack_factor=4):
       B, L, D = features.shape
       # Ensure length is divisible by stack_factor
       new_L = L // stack_factor * stack_factor
       features = features[:, :new_L, :]

       # Reshape and stack
       return features.view(B, -1, stack_factor, D)
                     .transpose(2, 3)
                     .contiguous()
                     .view(B, -1, D * stack_factor)
   ```

### Audio-Text Fusion Process

1. **Token-based Fusion**

   ```python
   # Example fusion process
   class AudioTextFusion:
       def fuse_features(
           self,
           text_embeddings: torch.Tensor,
           audio_features: torch.Tensor,
           audio_positions: List[int]
       ):
           # Insert audio features at specified positions
           for pos, features in zip(audio_positions, audio_features):
               text_embeddings[:, pos:pos+features.size(1)] = features

           # Generate attention masks
           attention_mask = self.create_attention_mask(
               text_embeddings.shape[1],
               audio_positions,
               features.shape[1]
           )

           return text_embeddings, attention_mask
   ```

2. **Streaming Fusion State**

   ```python
   # Example streaming state
   class StreamingFusionState:
       def __init__(self):
           self.text_buffer = []
           self.audio_buffer = []
           self.position_tracker = []

       def add_chunk(
           self,
           audio_features: torch.Tensor,
           text_position: int
       ):
           # Add new chunk and update state
           self.audio_buffer.append(audio_features)
           self.position_tracker.append(text_position)

       def get_fused_output(self) -> FusedFeatures:
           # Combine and process buffers
           return FusedFeatures(
               embeddings=self.combine_features(),
               attention_mask=self.create_mask(),
               audio_mask=self.create_audio_mask()
           )
   ```

### LLM Integration

1. **Feature Preparation**

   ```python
   # Example LLM input preparation
   def prepare_llm_input(
       fused_features: FusedFeatures,
       prompt: str
   ) -> Dict:
       return {
           "input_embeddings": fused_features.embeddings,
           "attention_mask": fused_features.attention_mask,
           "audio_mask": fused_features.audio_mask,
           "prompt": prompt,
           "metadata": {
               "audio_positions": fused_features.audio_positions,
               "feature_types": fused_features.feature_types
           }
       }
   ```

2. **Streaming Output Format**
   ```python
   # Example streaming output structure
   @dataclass
   class StreamingOutput:
       features: torch.Tensor
       intermediate_features: List[torch.Tensor]
       attention_mask: torch.Tensor
       audio_mask: torch.Tensor
       metadata: Dict[str, Any]
       is_final: bool
   ```

## Usage Examples

### Basic Usage

```python
import torch
from app.audio.config import AudioConfig
from app.audio.feature_extractor import AudioFeatureExtractor

# Create configuration
config = AudioConfig(
    encoder_type="wav2vec2",
    latency_preset="balanced",  # 2s chunks with 300ms overlap
    feature_layers=[6, 9, 12],  # Extract features from these layers
    stack_factor=4  # Stack 4 frames together
)

# Initialize feature extractor
processor = AudioFeatureExtractor(config)

# Process streaming audio
def process_audio_stream(audio_stream, chunk_duration=2.0):
    """Process streaming audio input.

    Args:
        audio_stream: Iterator yielding audio chunks
        chunk_duration: Duration of each chunk in seconds
    """
    chunk_samples = int(chunk_duration * config.sample_rate)

    for audio_chunk in audio_stream:
        # Process chunk
        result = processor.process_stream(audio_chunk)

        if result:
            features, hidden_states, metadata, transcription = result

            # Features: Main audio features [batch, time, hidden_size]
            # hidden_states: List of intermediate features from specified layers
            # metadata: List of dicts with speech detection and timing info
            # transcription: Optional transcription if enabled

            # Use the features as needed...
            yield {
                "features": features,
                "hidden_states": hidden_states,
                "metadata": metadata,
                "transcription": transcription
            }

# Example with multiple segments
def process_audio_segments(audio_segments, token_ids, start_indices):
    """Process multiple audio segments.

    Args:
        audio_segments: List of audio tensors
        token_ids: List of token IDs for each segment
        start_indices: List of starting indices for each segment
    """
    segments = processor.process_segments(
        audio_inputs=audio_segments,
        token_ids=token_ids,
        start_indices=start_indices
    )

    for segment in segments:
        # Access segment information
        features = segment.features  # Audio features
        is_speech = segment.is_speech  # Speech detection result
        segment_type = segment.segment_type  # speech/audio/silence
        energy_level = segment.energy_level  # Energy level
        timestamp = segment.timestamp  # Timestamp
        duration = segment.duration  # Duration
```

### Real-time Processing Example

```python
import torch
import sounddevice as sd
import numpy as np
from queue import Queue
from threading import Thread

class AudioStreamProcessor:
    def __init__(self, config=None):
        self.config = config or AudioConfig(latency_preset="fast")
        self.processor = AudioFeatureExtractor(self.config)
        self.audio_queue = Queue()
        self.running = False

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input."""
        # Convert to torch tensor
        audio = torch.from_numpy(indata.copy()).float()
        audio = audio.mean(dim=1, keepdim=True).t()  # Convert to mono
        self.audio_queue.put(audio)

    def process_stream(self):
        """Process audio stream."""
        while self.running:
            if not self.audio_queue.empty():
                audio = self.audio_queue.get()

                # Process audio chunk
                result = self.processor.process_stream(audio)
                if result:
                    features, hidden_states, metadata, transcription = result
                    # Handle the results as needed...
                    print(f"Processed chunk: {metadata}")

    def start(self, duration=None):
        """Start audio processing.

        Args:
            duration: Optional duration to record (None for indefinite)
        """
        self.running = True

        # Start processing thread
        process_thread = Thread(target=self.process_stream)
        process_thread.start()

        # Start audio input stream
        with sd.InputStream(
            channels=1,
            samplerate=self.config.sample_rate,
            callback=self.audio_callback
        ):
            if duration:
                sd.sleep(int(duration * 1000))
            else:
                while self.running:
                    sd.sleep(100)

        # Cleanup
        self.running = False
        process_thread.join()

# Usage
processor = AudioStreamProcessor()
processor.start(duration=10)  # Process 10 seconds of audio
```

### Configuration Examples

1. **Fast Response Configuration**

```python
config = AudioConfig(
    latency_preset="fast",  # 0.5s chunks
    chunk_duration=0.5,  # Override chunk duration
    overlap_ratio=0.1,  # 50ms overlap
    encoder_type="wav2vec2"
)
```

2. **High Quality Configuration**

```python
config = AudioConfig(
    latency_preset="quality",  # 5s chunks
    feature_layers=[3, 6, 9, 12],  # More intermediate features
    use_specaugment=True,  # Enable SpecAugment
    encoder_type="whisper",  # Use Whisper model
    encoder_name="openai/whisper-small"
)
```

3. **Balanced Configuration with Transcription**

```python
config = AudioConfig(
    latency_preset="balanced",  # 2s chunks
    transcribe=True,  # Enable transcription
    max_audio_length=30,  # Maximum audio length
    stream_buffer_size=32000  # Buffer size in samples
)
```

### Error Handling Example

```python
def safe_process_audio(audio_input, processor):
    """Safely process audio with error handling."""
    try:
        # Validate input
        if audio_input.dim() != 2 or audio_input.size(0) != 1:
            raise ValueError("Expected audio input shape: [1, samples]")

        # Check sample rate
        expected_samples = int(processor.config.chunk_duration *
                             processor.config.sample_rate)
        if audio_input.size(1) != expected_samples:
            print(f"Warning: Resampling audio to {processor.config.sample_rate}Hz")
            # Implement resampling here...

        # Process audio
        result = processor.process_stream(audio_input)
        return result

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None
```

These examples demonstrate the main ways to use the audio processing system. The system is designed to be flexible and can be adapted to different use cases by adjusting the configuration and processing approach. The streaming example shows how to handle real-time audio input, while the segment processing example shows how to handle multiple audio segments with specific positions in text.

## Conclusion

The Audio Processing System provides a robust foundation for real-time audio processing and feature extraction, specifically designed for LLM integration. Its modular architecture, streaming capabilities, and rich feature extraction make it well-suited for applications requiring sophisticated audio understanding in conversational AI systems.
