#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# OPTIMIZED FOR SPEECH-TO-SPEECH ONLY
# Removed all vision components for reduced latency and memory usage

import os
import warnings
import shutil

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

# Import specific models instead of wildcard import to avoid loading all models at startup
def get_model_class(model_name):
    """Lazy import of model classes to avoid loading all models at startup"""
    if model_name == "llava_s2s_qwen":
        from open_omni.model.language_model.llava_s2s_qwen import LlavaS2SQwenForCausalLM
        return LlavaS2SQwenForCausalLM
    elif model_name == "llava_qwen":
        from open_omni.model.language_model.llava_qwen import LlavaQwenForCausalLM
        return LlavaQwenForCausalLM
    elif model_name == "llava_s2s_llama":
        from open_omni.model.language_model.llava_s2s_llama import LlavaS2SLlamaForCausalLM
        return LlavaS2SLlamaForCausalLM
    elif model_name == "llava_llama":
        from open_omni.model.language_model.llava_llama import LlavaLlamaForCausalLM
        return LlavaLlamaForCausalLM
    else:
        raise ValueError(f"Unknown model name: {model_name}")

from open_omni.constants import DEFAULT_SPEECH_TOKEN
from open_omni.model.speech_encoder.builder import build_speech_encoder
from open_omni.model.speech_projector.builder import build_speech_projector
from open_omni.utils import rank0_print


def load_pretrained_model_speech_only(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", attn_implementation="flash_attention_2", customized_config=None, overwrite_config=None, **kwargs):
    """
    Optimized model loading for speech-only variants
    - Streamlined loading process for faster initialization
    - Memory optimizations throughout the pipeline
    """
    
    kwargs["device_map"] = device_map

    # Quantization setup with optimizations
    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    if customized_config is not None:
        kwargs["config"] = customized_config

    # Check for multimodal flag (but we'll optimize for speech-only)
    if "multimodal" in kwargs:
        if kwargs["multimodal"] is True:
            is_multimodal = True
            kwargs.pop("multimodal")
        else:
            is_multimodal = False
    else:
        is_multimodal = False

    rank0_print("Loading optimized speech-only model...")

    # Optimized model loading for speech-only variants
    if ("llava" in model_name.lower() or "longva" in model_name.lower() or 
        "openomni" in model_name.lower() or "omni" in model_name.lower() or is_multimodal):
        
        # Handle LoRA models
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn("LoRA model detected but no base model specified. This may cause issues.")
        
        if model_base is not None:
            # Loading with base model
            rank0_print(f"Loading speech-only model with base: {model_base}")
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            rank0_print("Loading speech-only LLaVA from base model...")

            # Streamlined model loading based on architecture
            if "qwen" in model_name.lower():
                if 's2s' in model_name.lower():
                    from open_omni.model.language_model.llava_s2s_qwen import LlavaS2SQwenConfig
                    llava_s2s_cfg = LlavaS2SQwenConfig.from_pretrained(model_path)
                    if overwrite_config is not None:
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_s2s_cfg, k, v)
                    ModelClass = get_model_class("llava_s2s_qwen")
                    model = ModelClass.from_pretrained(
                        model_base, low_cpu_mem_usage=True, 
                        attn_implementation=attn_implementation, config=llava_s2s_cfg, **kwargs
                    )
                else:
                    from open_omni.model.language_model.llava_qwen import LlavaQwenConfig
                    llava_cfg = LlavaQwenConfig.from_pretrained(model_path)
                    if overwrite_config is not None:
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_cfg, k, v)
                    ModelClass = get_model_class("llava_qwen")
                    model = ModelClass.from_pretrained(
                        model_base, low_cpu_mem_usage=True, 
                        attn_implementation=attn_implementation, config=llava_cfg, **kwargs
                    )
            else:
                # Default to Llama-based models
                if 's2s' in model_name.lower():
                    from open_omni.model.language_model.llava_s2s_llama import LlavaS2SLlamaConfig
                    llava_s2s_cfg = LlavaS2SLlamaConfig.from_pretrained(model_path)
                    if overwrite_config is not None:
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_s2s_cfg, k, v)
                    ModelClass = get_model_class("llava_s2s_llama")
                    model = ModelClass.from_pretrained(
                        model_base, low_cpu_mem_usage=True, 
                        attn_implementation=attn_implementation, config=llava_s2s_cfg, **kwargs
                    )
                else:
                    from open_omni.model.language_model.llava_llama import LlavaConfig
                    llava_cfg = LlavaConfig.from_pretrained(model_path)
                    if overwrite_config is not None:
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_cfg, k, v)
                    ModelClass = get_model_class("llava_llama")
                    model = ModelClass.from_pretrained(
                        model_base, low_cpu_mem_usage=True, 
                        attn_implementation=attn_implementation, config=llava_cfg, **kwargs
                    )
        else:
            # Standard model loading for non-multimodal models
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            rank0_print("Loading speech-only LLaVA from base model...")

            # Handle token embeddings
            token_num, token_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype)
                )
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype)
                )

            # Load LoRA weights
            rank0_print("Loading additional speech-only weights...")
            if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(
                    os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu"
                )
                non_lora_trainables = {
                    (k[11:] if k.startswith("base_model.") else k): v
                    for k, v in non_lora_trainables.items()
                }
                if any(k.startswith("model.model.") for k in non_lora_trainables):
                    non_lora_trainables = {
                        (k[6:] if k.startswith("model.") else k): v
                        for k, v in non_lora_trainables.items()
                    }
                model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            rank0_print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, model_path)
            rank0_print("Merging LoRA weights...")
            model = model.merge_and_unload()
            rank0_print("Model is loaded...")

        else:
            # Direct model loading (most common case)
            rank0_print(f"Loading speech-only model: {model_path}")
            
            if "qwen" in model_name.lower():
                if 's2s' in model_name.lower():
                    from open_omni.model.language_model.llava_s2s_qwen import LlavaS2SQwenConfig
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    if overwrite_config is not None:
                        llava_s2s_cfg = LlavaS2SQwenConfig.from_pretrained(model_path)
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_s2s_cfg, k, v)
                        ModelClass = get_model_class("llava_s2s_qwen")
                        model = ModelClass.from_pretrained(
                            model_path, low_cpu_mem_usage=True, 
                            attn_implementation=attn_implementation, config=llava_s2s_cfg, **kwargs
                        )
                    else:
                        ModelClass = get_model_class("llava_s2s_qwen")
                        model = ModelClass.from_pretrained(
                            model_path, low_cpu_mem_usage=True, 
                            attn_implementation=attn_implementation, **kwargs
                        )
                else:
                    from open_omni.model.language_model.llava_qwen import LlavaQwenConfig
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    if overwrite_config is not None:
                        llava_cfg = LlavaQwenConfig.from_pretrained(model_path)
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_cfg, k, v)
                        ModelClass = get_model_class("llava_qwen")
                        model = ModelClass.from_pretrained(
                            model_path, low_cpu_mem_usage=True, 
                            attn_implementation=attn_implementation, config=llava_cfg, **kwargs
                        )
                    else:
                        ModelClass = get_model_class("llava_qwen")
                        model = ModelClass.from_pretrained(
                            model_path, low_cpu_mem_usage=True, 
                            attn_implementation=attn_implementation, **kwargs
                        )
            else:
                # Default to Llama-based models
                if 's2s' in model_name.lower():
                    from open_omni.model.language_model.llava_s2s_llama import LlavaS2SLlamaConfig
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    if overwrite_config is not None:
                        llava_s2s_cfg = LlavaS2SLlamaConfig.from_pretrained(model_path)
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_s2s_cfg, k, v)
                        ModelClass = get_model_class("llava_s2s_llama")
                        model = ModelClass.from_pretrained(
                            model_path, low_cpu_mem_usage=True, 
                            attn_implementation=attn_implementation, config=llava_s2s_cfg, **kwargs
                        )
                    else:
                        ModelClass = get_model_class("llava_s2s_llama")
                        model = ModelClass.from_pretrained(
                            model_path, low_cpu_mem_usage=True, 
                            attn_implementation=attn_implementation, **kwargs
                        )
                else:
                    from open_omni.model.language_model.llava_llama import LlavaConfig
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    if overwrite_config is not None:
                        llava_cfg = LlavaConfig.from_pretrained(model_path)
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_cfg, k, v)
                        ModelClass = get_model_class("llava_llama")
                        model = ModelClass.from_pretrained(
                            model_path, low_cpu_mem_usage=True, 
                            attn_implementation=attn_implementation, config=llava_cfg, **kwargs
                        )
                    else:
                        ModelClass = get_model_class("llava_llama")
                        model = ModelClass.from_pretrained(
                            model_path, low_cpu_mem_usage=True, 
                            attn_implementation=attn_implementation, **kwargs
                        )
    else:
        # Standard model loading for non-multimodal models
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )

    # Speech encoder optimization
    speech_encoder = model.get_speech_encoder()
    if speech_encoder is not None:
        # FIXED: Removed is_loaded check since WhisperWrappedEncoder doesn't have this attribute
        # The speech encoder is already loaded when created, so we can directly optimize it
        speech_encoder.half()
        speech_encoder.eval()
        for param in speech_encoder.parameters():
            param.requires_grad = False

    # Memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Removed image_processor initialization for speech-only optimization
    image_processor = None

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", attn_implementation="flash_attention_2", customized_config=None, overwrite_config=None, **kwargs):
    """
    Main entry point - delegates to optimized speech-only loader
    """
    return load_pretrained_model_speech_only(
        model_path, model_base, model_name, load_8bit, load_4bit, 
        device_map, attn_implementation, customized_config, overwrite_config, **kwargs
    )

