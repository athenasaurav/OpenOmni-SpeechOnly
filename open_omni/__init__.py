# Speech-Only OpenOmni
# Vision components removed for optimization

# Only import what's absolutely necessary for speech-to-speech operation
# Lazy imports to avoid loading unnecessary modules

__version__ = "1.0.0-speech-only"
__all__ = []

# Lazy import function for speech models
def get_speech_model(model_name):
    """Lazy import speech models to avoid loading vision dependencies"""
    if model_name == "llava_s2s_qwen":
        from .model.language_model.llava_s2s_qwen import LlavaS2SQwenForCausalLM
        return LlavaS2SQwenForCausalLM
    elif model_name == "llava_s2s_llama":
        from .model.language_model.llava_s2s_llama import LlavaS2SLlamaForCausalLM
        return LlavaS2SLlamaForCausalLM
    else:
        raise ValueError(f"Unknown speech model: {model_name}")

