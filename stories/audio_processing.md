# Ultravox Development Guide

This document outlines the detailed implementation plan for Ultravox's audio processing system, including design choices, architecture, and implementation details.

## System Architecture Overview

Ultravox is a multimodal system that combines audio understanding with language processing capabilities. The system processes both audio and text inputs to generate meaningful responses.

### Pipeline Flow

1. **Input Stage**

   ```
   Audio Input (WAV/MP3) ─┐
                         ├─> Preprocessing ─> Feature Extraction
   Text Input (String) ───┘
   ```

   - Audio files are loaded and normalized
   - Text is tokenized with special tokens
   - Both streams are prepared for parallel processing

2. **Processing Stage**

   ```
   Audio Stream:
   Raw Audio ─> Resampling (16kHz) ─> Normalization ─> Feature Extraction ─> Frame Stacking ─> Projection
                                                        (Wav2Vec2/Whisper)    (4 frames)        (to LLM dim)

   Text Stream:
   Raw Text ─> Tokenization ─> Special Token Insertion ─> Embedding ─> Position Encoding
   ```

   - Audio processing maintains temporal alignment
   - Text processing preserves semantic structure
   - Both streams are aligned to same hidden dimension

3. **Fusion Stage**

   ```
   Audio Features    Text Embeddings
         │                │
         ▼                ▼
   Projection       Token Replacement
         │                │
         └───────┬────────┘
                 │
         Combined Sequence
                 │
                 ▼
           LLM Processing
                 │
                 ▼
            Text Output
   ```

   **Detailed Node Implementation:**

   a. **Audio Feature Node**

   ```python
   class AudioFeatureProcessor(nn.Module):
       def __init__(self, config):
           super().__init__()
           self.feature_dim = config.audio_hidden_size
           self.norm = nn.LayerNorm(self.feature_dim)

       def forward(self, audio_features):
           # Shape: [batch_size, time, feature_dim]
           normalized = self.norm(audio_features)
           return normalized
   ```

   b. **Text Embedding Node**

   ```python
   class TextEmbeddingProcessor(nn.Module):
       def __init__(self, config):
           super().__init__()
           self.embed_dim = config.hidden_size
           self.dropout = nn.Dropout(config.hidden_dropout_prob)

       def forward(self, embeddings):
           # Shape: [batch_size, seq_len, hidden_size]
           return self.dropout(embeddings)
   ```

   c. **Projection Node**

   ```python
   class ProjectionNode(nn.Module):
       def __init__(self, config):
           super().__init__()
           self.projection = nn.Sequential(
               nn.Linear(config.audio_hidden_size, config.intermediate_size),
               nn.GELU(),
               nn.LayerNorm(config.intermediate_size),
               nn.Linear(config.intermediate_size, config.hidden_size),
               nn.Dropout(config.hidden_dropout_prob)
           )

       def forward(self, features):
           return self.projection(features)
   ```

   d. **Token Replacement Node**

   ```python
   class TokenReplacementNode(nn.Module):
       def __init__(self, config):
           super().__init__()
           self.audio_token_id = config.audio_token_id

       def forward(self, text_embeddings, audio_embeddings, input_ids):
           # Find audio token positions
           audio_mask = (input_ids == self.audio_token_id)

           # Create output tensor
           output = text_embeddings.clone()

           # Replace embeddings at audio token positions
           output[audio_mask] = audio_embeddings

           return output
   ```

   e. **Sequence Combiner Node**

   ```python
   class SequenceCombiner(nn.Module):
       def __init__(self, config):
           super().__init__()
           self.final_norm = nn.LayerNorm(config.hidden_size)

       def forward(self, sequence):
           # Apply final normalization
           normalized = self.final_norm(sequence)

           # Update attention mask for the combined sequence
           attention_mask = self._create_attention_mask(sequence)

           return normalized, attention_mask

       def _create_attention_mask(self, sequence):
           # Create causal attention mask
           seq_length = sequence.size(1)
           attention_mask = torch.triu(
               torch.ones(seq_length, seq_length),
               diagonal=1
           ).bool()
           return attention_mask
   ```

### Data Flow Dimensions

```
Audio Pipeline:
[1, T] ─> [1, T, 768] ─> [1, T/4, 3072] ─> [1, T/4, 4096]
(raw)    (features)      (stacked)         (projected)

Text Pipeline:
[batch_size, seq_len] ─> [batch_size, seq_len, 4096]
(token ids)              (embeddings)

Combined:
[batch_size, seq_len + T/4, 4096]
(final sequence for LLM)
```

### Processing Alignment

1. **Temporal Alignment**

   - Audio frames are processed in chunks of `AUDIO_CHUNK_LENGTH` seconds
   - Each chunk produces `AUDIO_CHUNK_LENGTH * AUDIO_SAMPLE_RATE / STACK_FACTOR` tokens
   - Text tokens are aligned with audio frames through position encodings

2. **Feature Alignment**

   - Audio features are projected to match LLM hidden size
   - Text embeddings are already in LLM dimension
   - Both modalities share the same attention space

3. **Sequence Alignment**
   ```
   Original Text:   "Transcribe this <|audio|> and explain"
   Token IDs:       [1, 2, 3, AUDIO_TOKEN_ID, 4, 5]
   With Audio:      [1, 2, 3, *audio_features*, 4, 5]
   ```
   - Audio features seamlessly replace audio tokens
   - Attention flows naturally across modalities

### Core Components

1. **Audio Processing Pipeline**

   - Input handling for various audio formats (WAV, MP3)
   - Audio preprocessing and feature extraction
   - Audio frame stacking and projection
   - Integration with pre-trained audio models

2. **Text Processing Pipeline**

   - Text tokenization and embedding
   - Special token handling (`<|audio|>`)
   - Integration with language models

3. **Multimodal Fusion System**
   - Audio-text alignment
   - Feature space projection
   - Combined sequence processing

## Configuration Constants

```python
# Model Configuration
AUDIO_SAMPLE_RATE = 16000  # Hz
AUDIO_LATENCY_BLOCK_SIZE = 50  # Number of frames per block
STACK_FACTOR = 4  # Number of frames to stack
AUDIO_HIDDEN_SIZE = 1024  # Hidden size for audio features
TEXT_HIDDEN_SIZE = 4096  # Hidden size for text features
MAX_AUDIO_LENGTH = 30  # Maximum audio length in seconds
AUDIO_CHUNK_LENGTH = 5  # Length of audio chunks in seconds

# Special Tokens
AUDIO_TOKEN = "<|audio|>"
PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|startoftext|>"
EOS_TOKEN = "<|endoftext|>"

# Processing Parameters
AUDIO_MEAN = 0.0  # For normalization
AUDIO_STD = 1.0   # For normalization
MAX_TEXT_LENGTH = 512  # Maximum text sequence length
```

## Implementation Details

### 1. Audio Input Processing

#### Design Choices

- **Sample Rate**: 16kHz standard for audio processing
- **Input Formats**: Support for WAV and MP3 through standard audio libraries
- **Preprocessing**: Automatic resampling and normalization

#### Implementation Steps

1. Create audio input handler:

   ```python
   # ultravox/tools/infer_tool.py

   class AudioProcessor:
       def __init__(self, target_sample_rate=AUDIO_SAMPLE_RATE):
           self.target_sr = target_sample_rate

       def load_audio(self, file_path):
           """Load and preprocess audio file."""
           import torchaudio

           waveform, sample_rate = torchaudio.load(file_path)
           if sample_rate != self.target_sr:
               resampler = torchaudio.transforms.Resample(
                   sample_rate, self.target_sr
               )
               waveform = resampler(waveform)

           # Convert to mono if stereo
           if waveform.shape[0] > 1:
               waveform = torch.mean(waveform, dim=0, keepdim=True)

           return self.normalize_audio(waveform)

       def normalize_audio(self, waveform):
           """Normalize audio to zero mean and unit variance."""
           return (waveform - AUDIO_MEAN) / AUDIO_STD
   ```

   - Implement file validation
   - Add format conversion if needed
   - Setup audio preprocessing pipeline

2. Setup audio preprocessing:

   ```python
   # ultravox/model/ultravox_processing.py

   class UltravoxProcessor:
       def __init__(self, config):
           self.config = config
           self.audio_processor = AudioProcessor()
           self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

       def preprocess_audio(self, audio_path):
           """Process audio for model input."""
           waveform = self.audio_processor.load_audio(audio_path)

           # Split into chunks if needed
           if self.config.use_chunking:
               chunk_size = int(AUDIO_CHUNK_LENGTH * AUDIO_SAMPLE_RATE)
               chunks = torch.split(waveform, chunk_size, dim=-1)
               return [self.process_chunk(c) for c in chunks]
           return self.process_chunk(waveform)
   ```

   - Implement resampling logic
   - Add normalization functions
   - Setup feature extraction pipeline

### 2. Audio Feature Extraction

#### Design Choices

- **Primary Models**:
  - Wav2Vec2 for speech feature extraction
  - Whisper as an alternative encoder
- **Feature Dimensionality**: Based on model architecture

#### Implementation Steps

1. Audio encoder integration:

   ```python
   # ultravox/model/ultravox_model.py

   class AudioTower(nn.Module):
       def __init__(self, config):
           super().__init__()
           if config.audio_encoder_type == "wav2vec2":
               self.encoder = Wav2Vec2Model.from_pretrained(
                   config.audio_encoder_name
               )
           elif config.audio_encoder_type == "whisper":
               self.encoder = WhisperModel.from_pretrained(
                   config.audio_encoder_name
               ).encoder

           self.proj = nn.Linear(
               self.encoder.config.hidden_size,
               config.hidden_size
           )

       def forward(self, audio_input):
           features = self.encoder(audio_input).last_hidden_state
           return self.proj(features)
   ```

   - Implement `_create_audio_tower`
   - Setup model configuration
   - Add feature extraction pipeline

2. Audio frame processing:

   ```python
   # ultravox/model/ultravox_model.py

   class StackAudioFrames(nn.Module):
       def __init__(self, stack_factor):
           super().__init__()
           self.stack_factor = stack_factor

       def forward(self, features):
           B, L, D = features.shape
           # Ensure length is divisible by stack_factor
           new_L = L // self.stack_factor * self.stack_factor
           features = features[:, :new_L, :]

           # Reshape and stack
           stacked = features.view(
               B, -1, self.stack_factor, D
           ).transpose(2, 3).contiguous()
           return stacked.view(B, -1, D * self.stack_factor)
   ```

   - Implement `StackAudioFrames`
   - Add frame stacking logic
   - Setup projection layers

### 3. Text Processing

#### Design Choices

- **Tokenization**: Based on LLM tokenizer
- **Special Tokens**: `<|audio|>` for audio context
- **Embedding Dimension**: Matched with LLM requirements

#### Implementation Steps

1. Text preprocessing:

   ```python
   # ultravox/model/ultravox_processing.py

   class TextProcessor:
       def __init__(self, tokenizer):
           self.tokenizer = tokenizer
           self.tokenizer.add_special_tokens({
               "additional_special_tokens": [AUDIO_TOKEN]
           })

       def process_text(self, text, audio_present=True):
           """Process text input with optional audio token."""
           if audio_present and AUDIO_TOKEN not in text:
               # Add audio token at the start if not present
               text = f"{AUDIO_TOKEN} {text}"

           tokens = self.tokenizer(
               text,
               max_length=MAX_TEXT_LENGTH,
               padding="max_length",
               truncation=True,
               return_tensors="pt"
           )
           return tokens
   ```

   - Setup tokenizer
   - Implement special token handling
   - Add embedding logic

2. LLM integration:
   ```python
   # ultravox/model/ultravox_model.py
   ```
   - Implement `_create_language_model`
   - Setup model configuration
   - Add inference pipeline

### 4. Multimodal Fusion

#### Design Choices

- **Fusion Strategy**: Token replacement with projected audio
- **Alignment**: Linear projection layers
- **Sequence Handling**: Combined processing

#### Implementation Steps

1. Projector implementation:

   ```python
   # ultravox/model/ultravox_model.py

   class UltravoxProjector(nn.Module):
       def __init__(self, config):
           super().__init__()
           self.audio_proj = nn.Sequential(
               nn.Linear(
                   config.audio_hidden_size * config.stack_factor,
                   config.hidden_size
               ),
               nn.LayerNorm(config.hidden_size),
               nn.GELU(),
               nn.Linear(config.hidden_size, config.hidden_size)
           )

       def forward(self, audio_features):
           return self.audio_proj(audio_features)
   ```

   - Create `UltravoxProjector`
   - Implement projection layers
   - Add dimension alignment

2. Fusion logic:

   ```python
   # ultravox/model/ultravox_model.py

   class UltravoxModel(nn.Module):
       def forward(self, input_ids, audio_features=None, attention_mask=None):
           # Get base embeddings
           embeddings = self.get_input_embeddings()(input_ids)

           if audio_features is not None:
               # Project audio features
               audio_embeds = self.audio_projector(audio_features)

               # Find audio token positions
               audio_pos = (input_ids == self.tokenizer.convert_tokens_to_ids(AUDIO_TOKEN))

               # Replace audio token embeddings
               embeddings[audio_pos] = audio_embeds

           # Process through LLM
           outputs = self.language_model(
               inputs_embeds=embeddings,
               attention_mask=attention_mask,
               return_dict=True
           )
           return outputs
   ```

   - Implement forward pass
   - Add sequence combination
   - Setup attention mechanisms

## Testing and Validation

### Unit Tests

1. Audio processing tests

   - Input handling
   - Preprocessing
   - Feature extraction

2. Text processing tests

   - Tokenization
   - Special token handling
   - Embedding generation

3. Fusion tests
   - Projection accuracy
   - Sequence alignment
   - Output generation

### Integration Tests

1. End-to-end pipeline tests
2. Performance benchmarks
3. Memory usage analysis

## Deployment

### Requirements

#### Core Dependencies

```python
# requirements.txt
torch>=1.8.0
transformers>=4.30.0
numpy>=1.21.0
tqdm>=4.65.0
```

#### Audio Processing Libraries

```python
# audio_requirements.txt
torchaudio>=0.8.0  # Audio loading and processing
librosa>=0.9.2     # Advanced audio processing features
soundfile>=0.10.3  # Audio file I/O
scipy>=1.7.0       # Signal processing utilities
pyaudioanalysis>=0.3.14  # Audio feature extraction
```

#### Audio Codec Support

```python
# codec_requirements.txt
pydub>=0.25.1      # Audio format conversion
ffmpeg-python>=0.2.0  # FFmpeg bindings for audio processing
```

#### Feature Requirements

- **torchaudio**: Primary audio processing library

  - Audio loading/saving (WAV, MP3, FLAC)
  - Resampling and normalization
  - Spectrogram generation

- **librosa**: Advanced audio processing

  - Feature extraction (MFCCs, mel spectrograms)
  - Time-frequency transforms
  - Audio signal processing

- **soundfile**: Audio file handling

  - Multi-format support
  - Streaming capabilities
  - Metadata handling

- **ffmpeg-python**: Audio format support
  - Codec management
  - Format conversion
  - Stream processing

## Performance Considerations

### Scalability

1. Batch processing
2. Parallel processing
3. Resource management

## Future Improvements

### Potential Enhancements

1. Additional audio encoders
2. Improved fusion strategies
3. Streaming capabilities
4. Model optimization

## References

### Code References

- Audio Processing: `ultravox/model/ultravox_model.py`
- Text Processing: `ultravox/model/ultravox_processing.py`
- Inference: `ultravox/inference/infer.py`
- Tools: `ultravox/tools/`
