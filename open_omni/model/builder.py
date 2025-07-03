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

"""
Optimized Speech-Only Model Builder
- Removed vision processing for reduced latency
- Streamlined speech-only model loading
- Memory optimizations for speech pipeline
"""

import os
import warnings
import shutil
import gc

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from open_omni.model import *
from open_omni.constants import DEFAULT_SPEECH_TOKEN
from open_omni.utils import rank0_print
from open_omni.model.speech_encoder.builder import build_speech_encoder
from open_omni.model.speech_projector.builder import build_speech_projector


def load_pretrained_model_speech_only(model_path, model_base, model_name, load_8bit=False, load_4bit=False, 
                                     device_map="auto", attn_implementation="flash_attention_2", 
                                     customized_config=None, overwrite_config=None, **kwargs):
    """
    Optimized speech-only model loading
    - Removed vision components for reduced memory usage
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
        if "qwen2" in model_path.lower() and attn_implementation == "eager":
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            kwargs["torch_dtype"] = torch.float16

    if customized_config is not None:
        kwargs["config"] = customized_config

    # Check for multimodal flag (but we'll optimize for speech-only)
    if "multimodal" in kwargs:
        if kwargs["multimodal"] is True:
            is_multimodal = True
            kwargs.pop("multimodal")
    else:
        is_multimodal = False

    rank0_print("Loading optimized speech-only model...")

    # Optimized model loading for speech-only variants
    if ("llava" in model_name.lower() or "longva" in model_name.lower() or 
        "openomni" in model_name.lower() or "omni" in model_name.lower() or is_multimodal):
        
        # Handle LoRA models
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. "
                "If you are loading a LoRA model, please provide the `model_base` argument."
            )
        
        if "lora" in model_name.lower() and model_base is not None:
            # LoRA loading with speech-only optimizations
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            rank0_print("Loading speech-only LLaVA from base model...")
            
            # Streamlined model loading based on architecture
            if "qwen" in model_name.lower():
                if 's2s' in model_name.lower():
                    from open_omni.model.language_model.llava_s2s_qwen import LlavaS2SQwenConfig
                    lora_cfg_pretrained = LlavaS2SQwenConfig.from_pretrained(model_path)
                    model = LlavaS2SQwenForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, 
                        attn_implementation=attn_implementation, **kwargs
                    )
                else:
                    from open_omni.model.language_model.llava_qwen import LlavaQwenConfig
                    lora_cfg_pretrained = LlavaQwenConfig.from_pretrained(model_path)
                    model = LlavaQwenForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, 
                        attn_implementation=attn_implementation, **kwargs
                    )
            else:
                # Default to Llama-based models
                if "s2s" in model_name.lower():
                    from open_omni.model.language_model.llava_s2s_llama import LlavaS2SLlamaConfig
                    lora_cfg_pretrained = LlavaS2SLlamaConfig.from_pretrained(model_path)
                    model = LlavaS2SLlamaForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, 
                        attn_implementation=attn_implementation, **kwargs
                    )
                else:
                    from open_omni.model.language_model.llava_llama import LlavaConfig
                    lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
                    model = LlavaLlamaForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, 
                        attn_implementation=attn_implementation, **kwargs
                    )

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
            rank0_print("Loading additional speech-only LLaVA weights...")
            if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(
                    os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu"
                )
            else:
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
                    return torch.load(cache_file, map_location="cpu")
                non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")
            
            # Clean up key names
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
            rank0_print("Speech-only model loaded successfully...")

        elif model_base is not None:
            # Base model with projector loading
            rank0_print(f"Loading speech-only LLaVA from base model {model_base}...")
            
            if "qwen" in model_name.lower():
                if 's2s' in model_name.lower():
                    from open_omni.model.language_model.llava_s2s_qwen import LlavaS2SQwenConfig
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                    cfg_pretrained = LlavaS2SQwenConfig.from_pretrained(model_path)
                    model = LlavaS2SQwenForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, 
                        attn_implementation=attn_implementation, **kwargs
                    )
                else:
                    from open_omni.model.language_model.llava_qwen import LlavaQwenConfig
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                    cfg_pretrained = LlavaQwenConfig.from_pretrained(model_path)
                    model = LlavaQwenForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, 
                        attn_implementation=attn_implementation, **kwargs
                    )
            else:
                # Default to Llama-based models
                if "s2s" in model_name.lower():
                    from open_omni.model.language_model.llava_s2s_llama import LlavaS2SLlamaConfig
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                    cfg_pretrained = LlavaS2SLlamaConfig.from_pretrained(model_path)
                    model = LlavaS2SLlamaForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, 
                        attn_implementation=attn_implementation, **kwargs
                    )
                else:
                    from open_omni.model.language_model.llava_llama import LlavaConfig
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                    cfg_pretrained = LlavaConfig.from_pretrained(model_path)
                    model = LlavaLlamaForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, 
                        attn_implementation=attn_implementation, **kwargs
                    )

            # Load multimodal projector weights
            mm_projector_weights = torch.load(
                os.path.join(model_path, "mm_projector.bin"), map_location="cpu"
            )
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)

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
                        model = LlavaS2SQwenForCausalLM.from_pretrained(
                            model_path, low_cpu_mem_usage=True, 
                            attn_implementation=attn_implementation, config=llava_s2s_cfg, **kwargs
                        )
                    else:
                        model = LlavaS2SQwenForCausalLM.from_pretrained(
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
                        model = LlavaQwenForCausalLM.from_pretrained(
                            model_path, low_cpu_mem_usage=True, 
                            attn_implementation=attn_implementation, config=llava_cfg, **kwargs
                        )
                    else:
                        model = LlavaQwenForCausalLM.from_pretrained(
                            model_path, low_cpu_mem_usage=True, 
                            attn_implementation=attn_implementation, **kwargs
                        )
            else:
                # Default to Llama-based models
                if "s2s" in model_name.lower():
                    from open_omni.model.language_model.llava_s2s_llama import LlavaS2SLlamaConfig
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    if customized_config is None:
                        llava_cfg = LlavaS2SLlamaConfig.from_pretrained(model_path)
                    else:
                        llava_cfg = customized_config

                    if overwrite_config is not None:
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_cfg, k, v)

                    model = LlavaS2SLlamaForCausalLM.from_pretrained(
                        model_path, low_cpu_mem_usage=True, 
                        attn_implementation=attn_implementation, config=llava_cfg, **kwargs
                    )
                else:
                    from open_omni.model.language_model.llava_llama import LlavaConfig
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    if customized_config is None:
                        llava_cfg = LlavaConfig.from_pretrained(model_path)
                        if "v1.5" in model_name.lower():
                            llava_cfg.delay_load = True
                    else:
                        llava_cfg = customized_config

                    if overwrite_config is not None:
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_cfg, k, v)

                    model = LlavaLlamaForCausalLM.from_pretrained(
                        model_path, low_cpu_mem_usage=True, 
                        attn_implementation=attn_implementation, config=llava_cfg, **kwargs
                    )

    else:
        # Load standard language model (non-multimodal)
        if model_base is not None:
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto"
            )
            rank0_print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            rank0_print(f"Merging weights")
            model = model.merge_and_unload()
            rank0_print("Convert to FP16...")
            model.to(torch.float16)
        else:
            if "mpt" in model_name.lower().replace("prompt", ""):
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )

    rank0_print(f"Speech-Only Model Class: {model.__class__.__name__}")
    
    # Removed image_processor initialization for speech-only optimization
    image_processor = None

    # Optimized multimodal setup for speech-only
    if ("llava" in model_name.lower() or "longva" in model_name.lower() or 
        "openomni" in model_name.lower() or "omni" in model_name.lower() or is_multimodal):
        
        # Handle speech tokens only (removed image tokens for optimization)
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        
        # Removed image patch token handling for speech-only optimization
        # if mm_use_im_patch_token:
        #     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        
        # Add speech token if needed
        if hasattr(model.config, 'speech_token') and model.config.speech_token:
            tokenizer.add_tokens([DEFAULT_SPEECH_TOKEN], special_tokens=True)
        
        model.resize_token_embeddings(len(tokenizer))

        # REMOVED: Vision tower loading for speech-only optimization
        # This eliminates the major memory and latency bottleneck
        
        # Optimized speech encoder loading
        if getattr(model.config, "speech_encoder_type", None) is not None:
            rank0_print("Loading optimized speech encoder...")
            model.get_model().speech_encoder = build_speech_encoder(model.config)
            model.get_model().speech_encoder.to(device=device_map, dtype=torch.float16)
            model.get_model().speech_projector.to(device=device_map, dtype=torch.float16)
            
            # Optimize speech encoder for inference
            model.get_model().speech_encoder.eval()
            for param in model.get_model().speech_encoder.parameters():
                param.requires_grad = False
            
            rank0_print("Speech encoder optimized for inference")

    # Determine context length
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    # Final optimizations
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    rank0_print("Speech-only model loading completed with optimizations")
    
    return tokenizer, model, image_processor, context_len


# Alias for backward compatibility
load_pretrained_model = load_pretrained_model_speech_only

