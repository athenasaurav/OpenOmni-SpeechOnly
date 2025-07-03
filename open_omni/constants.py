"""
Optimized Constants for Speech-Only OpenOmni
- Removed image-related constants for cleaner codebase
- Added speech-specific optimizations
"""

# Speech-only tokens (removed image tokens for optimization)
DEFAULT_SPEECH_TOKEN = "<speech>"
DEFAULT_SPEECH_PATCH_TOKEN = "<sp_patch>"
DEFAULT_SP_START_TOKEN = "<sp_start>"
DEFAULT_SP_END_TOKEN = "<sp_end>"

# Legacy image tokens (kept for compatibility but not used)
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# Speech processing constants
SPEECH_TOKEN_INDEX = -200
SPEECH_PATCH_TOKEN_INDEX = -201
SP_START_TOKEN_INDEX = -202
SP_END_TOKEN_INDEX = -203

# Legacy image constants (kept for compatibility)
IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100

# Speech-specific configurations
DEFAULT_SPEECH_SAMPLE_RATE = 16000
DEFAULT_SPEECH_CHUNK_LENGTH = 30  # seconds
DEFAULT_SPEECH_OVERLAP = 0.5  # seconds

# Optimization flags
ENABLE_SPEECH_STREAMING = True
ENABLE_MEMORY_OPTIMIZATION = True
ENABLE_FAST_INFERENCE = True

# Model configurations for speech-only mode
SPEECH_ENCODER_HIDDEN_SIZE = 1024
SPEECH_PROJECTOR_HIDDEN_SIZE = 4096
SPEECH_GENERATOR_HIDDEN_SIZE = 4096

# Performance optimization constants
MAX_SPEECH_LENGTH = 30.0  # Maximum speech input length in seconds
SPEECH_BATCH_SIZE = 1  # Optimized for low latency
SPEECH_NUM_WORKERS = 2  # Reduced for memory efficiency

# Streaming configurations
SPEECH_STREAM_CHUNK_SIZE = 1024
SPEECH_STREAM_OVERLAP = 256
SPEECH_STREAM_TIMEOUT = 5.0

# Memory optimization settings
ENABLE_GRADIENT_CHECKPOINTING = True
ENABLE_MIXED_PRECISION = True
ENABLE_TORCH_COMPILE = False  # Disable for compatibility

# Speech unit generation settings
SPEECH_UNIT_VOCAB_SIZE = 1024
SPEECH_UNIT_SAMPLE_RATE = 50  # Hz
SPEECH_UNIT_HOP_LENGTH = 320

# Vocoder settings
VOCODER_SAMPLE_RATE = 22050
VOCODER_HOP_LENGTH = 256
VOCODER_WIN_LENGTH = 1024

# Default model paths (speech-only)
DEFAULT_SPEECH_ENCODER_PATH = "openai/whisper-large-v3"
DEFAULT_SPEECH_PROJECTOR_TYPE = "mlp2x_gelu"
DEFAULT_SPEECH_GENERATOR_TYPE = "ctc"
DEFAULT_VOCODER_PATH = "microsoft/speecht5_hifigan"

# Conversation templates for speech-only mode
SPEECH_CONVERSATION_TEMPLATE = "speech_v1"
DEFAULT_CONVERSATION_VERSION = "v1"

# API endpoints for speech-only mode
SPEECH_GENERATION_ENDPOINT = "/generate_speech"
SPEECH_STREAMING_ENDPOINT = "/stream_speech"
HEALTH_CHECK_ENDPOINT = "/health"

# Error messages
SPEECH_INPUT_ERROR = "Invalid speech input format"
MODEL_LOADING_ERROR = "Failed to load speech-only model"
GENERATION_ERROR = "Speech generation failed"
STREAMING_ERROR = "Speech streaming failed"

# Success messages
MODEL_LOADED_SUCCESS = "Speech-only model loaded successfully"
GENERATION_SUCCESS = "Speech generated successfully"
STREAMING_SUCCESS = "Speech streaming started successfully"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "openomni_speech_only.log"

# Feature flags for optimization
FEATURES = {
    "vision_processing": False,  # Disabled for speech-only mode
    "speech_processing": True,
    "streaming": True,
    "memory_optimization": True,
    "fast_inference": True,
    "gradient_checkpointing": True,
    "mixed_precision": True,
    "torch_compile": False,
    "chunked_processing": True,
    "parallel_processing": True,
}

# Version information
VERSION = "1.0.0-speech-only"
BUILD_DATE = "2024-07-02"
OPTIMIZATION_LEVEL = "maximum"

