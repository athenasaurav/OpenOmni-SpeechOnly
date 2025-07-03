"""
Speech-Only Multimodal Utilities
- Removed all image/vision processing functions
- Optimized purely for speech-to-speech processing
- No image/video processing capabilities
"""

import torch
from transformers import StoppingCriteria
from open_omni.constants import SPEECH_TOKEN_INDEX


def tokenizer_speech_tokens(prompt, tokenizer, speech_token_index=SPEECH_TOKEN_INDEX, return_tensors=None):
    """
    Speech-only version of tokenizer function
    - Removed all image token processing
    - Only handles speech tokens for optimized performance
    """
    # Split the prompt into chunks based on <speech> placeholders
    prompt_chunks = []
    speech_indices = []
    last_position = 0

    # Find all speech token positions
    start = 0
    while True:
        start = prompt.find("<speech>", start)
        if start == -1:
            break
        # Add text chunk before speech token
        if start > last_position:
            chunk_ids = tokenizer(prompt[last_position:start]).input_ids
            if len(chunk_ids) > 0:
                prompt_chunks.append(chunk_ids)
        
        # Add speech token index
        speech_indices.append(len(prompt_chunks))
        prompt_chunks.append([speech_token_index])
        
        last_position = start + len("<speech>")
        start = last_position

    # Add the remaining part of the prompt
    if last_position < len(prompt):
        remaining_chunk = tokenizer(prompt[last_position:]).input_ids
        if len(remaining_chunk) > 0:
            prompt_chunks.append(remaining_chunk)

    # Flatten the chunks
    input_ids = []
    for chunk in prompt_chunks:
        if isinstance(chunk, list):
            input_ids.extend(chunk)
        else:
            input_ids.extend(chunk.tolist() if hasattr(chunk, 'tolist') else chunk)

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        elif return_tensors == "np":
            import numpy as np
            return np.array(input_ids)
    
    return input_ids


def expand_speech_tokens(input_ids, speech_lengths, speech_token_index=SPEECH_TOKEN_INDEX):
    """
    Expand speech tokens to match speech feature lengths
    - Speech-only optimization
    - No image token handling
    """
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
    
    if len(speech_lengths) == 0:
        return input_ids
    
    # Find speech token positions
    speech_positions = (input_ids == speech_token_index).nonzero(as_tuple=True)[0]
    
    if len(speech_positions) != len(speech_lengths):
        raise ValueError(f"Number of speech tokens ({len(speech_positions)}) doesn't match speech lengths ({len(speech_lengths)})")
    
    # Expand tokens
    new_input_ids = []
    last_pos = 0
    
    for i, pos in enumerate(speech_positions):
        # Add tokens before speech token
        new_input_ids.extend(input_ids[last_pos:pos].tolist())
        
        # Add expanded speech tokens
        speech_length = speech_lengths[i]
        new_input_ids.extend([speech_token_index] * speech_length)
        
        last_pos = pos + 1
    
    # Add remaining tokens
    new_input_ids.extend(input_ids[last_pos:].tolist())
    
    return torch.tensor(new_input_ids, dtype=input_ids.dtype)


class KeywordsStoppingCriteria(StoppingCriteria):
    """Stopping criteria for speech generation"""
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)


def process_speech_for_generation(speech_features, model_config):
    """
    Process speech features for generation
    - Optimized for speech-only processing
    - No vision component handling
    """
    if speech_features is None:
        return None
    
    # Ensure proper tensor format
    if not isinstance(speech_features, torch.Tensor):
        speech_features = torch.tensor(speech_features)
    
    # Add batch dimension if needed
    if speech_features.dim() == 2:
        speech_features = speech_features.unsqueeze(0)
    
    # Memory optimization: use FP16 if available
    if hasattr(model_config, 'torch_dtype') and model_config.torch_dtype == torch.float16:
        speech_features = speech_features.half()
    
    return speech_features


def get_speech_token_count(prompt):
    """
    Count speech tokens in prompt
    - Speech-only utility function
    """
    return prompt.count("<speech>")


def validate_speech_input(speech_data):
    """
    Validate speech input data
    - Ensures proper format for speech processing
    """
    if speech_data is None:
        return False
    
    if isinstance(speech_data, (list, tuple)):
        return len(speech_data) > 0
    
    if isinstance(speech_data, torch.Tensor):
        return speech_data.numel() > 0
    
    return False


# Speech-only conversation utilities
def prepare_speech_conversation(prompt, speech_data=None):
    """
    Prepare conversation with speech input
    - Speech-only optimization
    - No image handling
    """
    if speech_data is not None and validate_speech_input(speech_data):
        # Add speech token to prompt if speech data is provided
        if "<speech>" not in prompt:
            prompt = "<speech>\n" + prompt
    
    return prompt


def cleanup_speech_memory():
    """
    Clean up speech processing memory
    - Optimized memory management for speech-only mode
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Export speech-only functions
__all__ = [
    'tokenizer_speech_tokens',
    'expand_speech_tokens', 
    'KeywordsStoppingCriteria',
    'process_speech_for_generation',
    'get_speech_token_count',
    'validate_speech_input',
    'prepare_speech_conversation',
    'cleanup_speech_memory'
]

