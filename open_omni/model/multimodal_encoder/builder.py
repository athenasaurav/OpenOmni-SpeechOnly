# VISION COMPONENTS REMOVED FOR SPEECH-ONLY OPTIMIZATION
# This module is kept for compatibility but contains no functional code

def build_vision_tower(vision_tower_cfg, **kwargs):
    """
    Vision tower builder - DISABLED for speech-only optimization
    This function is kept for compatibility but always returns None
    """
    raise NotImplementedError(
        "Vision tower functionality has been removed in the speech-only version. "
        "This model only supports speech-to-speech processing."
    )

# Stub classes for compatibility
class CLIPVisionTower:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Vision components removed in speech-only version")

class CLIPVisionTowerS2:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Vision components removed in speech-only version")

# Export empty list to prevent imports
__all__ = []

