# VISION COMPONENTS REMOVED FOR SPEECH-ONLY OPTIMIZATION
# This module is kept for compatibility but contains no functional code

def build_vision_resampler(config, delay_load=False, **kwargs):
    """
    Vision resampler builder - DISABLED for speech-only optimization
    This function is kept for compatibility but always returns None
    """
    raise NotImplementedError(
        "Vision resampler functionality has been removed in the speech-only version. "
        "This model only supports speech-to-speech processing."
    )

# Export empty list to prevent imports
__all__ = []

