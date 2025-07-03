# Speech-Only Model Imports
# Vision-related models completely removed for optimization

import os

# Only include speech-to-speech models
AVAILABLE_MODELS = {
    "llava_s2s_llama": "LlavaS2SLlamaForCausalLM, LlavaS2SLlamaConfig",
    "llava_s2s_qwen": "LlavaS2SQwenForCausalLM, LlavaS2SQwenConfig",
}

# Import only speech-to-speech models
for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except Exception as e:
        print(f"Warning: Failed to import {model_name}: {e}")
        # Continue without failing - some models might not be available

# Export only speech models
__all__ = [
    'LlavaS2SLlamaForCausalLM', 'LlavaS2SLlamaConfig',
    'LlavaS2SQwenForCausalLM', 'LlavaS2SQwenConfig'
]

