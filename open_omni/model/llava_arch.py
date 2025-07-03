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

from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn

# REMOVED: Vision tower imports
# from .multimodal_encoder.builder import build_vision_tower
# from .multimodal_resampler.builder import build_vision_resampler
# from .multimodal_projector.builder import build_vision_projector

from .speech_encoder.builder import build_speech_encoder
from .speech_projector.builder import build_speech_projector
from .speech_generator.builder import build_speech_generator

from open_omni.constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN
from open_omni.utils import rank0_print, lengths_to_padding_mask
import random


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        # REMOVED: Vision encoder initialization
        # Keeping only speech components for optimized performance
        
        # speech encoder     
        if hasattr(config, "speech_encoder"):
            self.speech_encoder = build_speech_encoder(config)
            self.speech_projector = build_speech_projector(config)
        
        # speech generator
        if hasattr(config, "speech_generator_type"):
            self.speech_generator = build_speech_generator(config)

    # REMOVED: Vision tower methods
    # def get_vision_tower(self):
    # def initialize_vision_modules(self, model_args, fsdp=None):
    
    # speech encoder
    def get_speech_encoder(self):
        speech_encoder = getattr(self, 'speech_encoder', None)
        if type(speech_encoder) is list:
            speech_encoder = speech_encoder[0]
        return speech_encoder
    
    def initialize_speech_modules(self, model_args, fsdp=None):
        self.config.speech_encoder = getattr(model_args, "speech_encoder", None)
        self.config.speech_encoder_type = getattr(model_args, "speech_encoder_type", None)
        self.config.speech_projector_type = getattr(model_args, 'speech_projector_type', 'linear')
        self.config.speech_encoder_ds_rate = getattr(model_args, 'speech_encoder_ds_rate', 5)
        self.config.speech_encoder_hidden_size = getattr(model_args, 'speech_encoder_hidden_size', 1280)

        if self.get_speech_encoder() is None:
            speech_encoder = build_speech_encoder(self.config)
            if fsdp is not None and len(fsdp) > 0:
                self.speech_encoder = [speech_encoder]
            else:
                self.speech_encoder = speech_encoder

        if getattr(self, 'speech_projector', None) is None:
            self.speech_projector = build_speech_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.speech_projector.parameters():
                p.requires_grad = True

        if model_args.pretrain_speech_projector is not None:
            pretrain_speech_projector_weights = torch.load(model_args.pretrain_speech_projector, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.speech_projector.load_state_dict(get_w(pretrain_speech_projector_weights, 'speech_projector'))
    
    def get_speech_generator(self):
        speech_generator = getattr(self, 'speech_generator', None)
        if type(speech_generator) is list:
            speech_generator = speech_generator[0]
        return speech_generator
    
    def initialize_speech_generator(self, model_args):
        self.config.speech_generator_type = getattr(model_args, 'speech_generator_type', 'ctc')
        self.config.ctc_decoder_config = getattr(model_args, 'ctc_decoder_config', '(4,4096,32,11008)')
        self.config.ctc_upsample_factor = getattr(model_args, 'ctc_upsample_factor', 1)
        self.config.ctc_loss_weight = getattr(model_args, 'ctc_loss_weight', 1.0)
        self.config.unit_vocab_size = getattr(model_args, 'unit_vocab_size', 1000)
        self.tune_speech_generator_only = getattr(model_args, 'tune_speech_generator_only', False)
        if getattr(self, "speech_generator", None) is None:
            self.speech_generator = build_speech_generator(self.config)


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    # REMOVED: Vision tower methods
    # def get_vision_tower(self):
    # def get_2dPool(self, image_feature):
    # def encode_images(self, images):
    # def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
    
    # speech proposer
    def get_speech_encoder(self):
        return self.get_model().get_speech_encoder()
    
    def get_speech_projector(self):
        return self.get_model().speech_projector

    def get_speech_generator(self):
        return self.get_model().get_speech_generator()

    def encode_speech(self, speech, speech_lengths):
        """Optimized speech encoding with FP16 support"""
        speech_encoder_type = self.config.speech_encoder_type
        speech_encoder = self.get_speech_encoder()
        
        if "whisper" in speech_encoder_type.lower():
            # Optimized Whisper encoding with memory efficiency
            with torch.cuda.amp.autocast(enabled=True):
                encoder_outs = speech_encoder(speech.permute(0, 2, 1))
            speech_lengths = (speech_lengths + 1) // 2
        else:
            raise ValueError(f'Unknown speech encoder: {speech_encoder}')
            
        speech_projector_type = self.config.speech_projector_type
        speech_projector = self.get_speech_projector()
        
        if speech_projector_type == "linear":
            with torch.cuda.amp.autocast(enabled=True):
                encoder_outs = speech_projector(encoder_outs)
            speech_lengths = speech_lengths // speech_projector.k
        else:
            raise ValueError(f'Unknown speech projector: {speech_projector_type}')
            
        speech_features = [encoder_outs[i, :speech_lengths[i]] for i in range(len(encoder_outs))]
        return speech_features
    
    def prepare_inputs_labels_for_speech_only(self, input_ids, position_ids, attention_mask, past_key_values, labels, speeches=None, speech_lengths=None):
        """
        Optimized speech-only input preparation
        Removed all vision processing for maximum performance
        """
        
        # Early return if no speech input
        if speeches is None:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        # Process speech features
        speech_features = self.encode_speech(speeches, speech_lengths)
        
        # Initialize tensors
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
            
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Remove padding using attention_mask
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        
        # Process each batch item
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_speeches = (cur_input_ids == SPEECH_TOKEN_INDEX).sum()
            
            if num_speeches == 0:
                # No speech tokens, just embed text
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                continue
            
            # Find speech token positions
            speech_token_indices = torch.where(cur_input_ids == SPEECH_TOKEN_INDEX)[0].tolist()
            
            # Split input around speech tokens
            cur_input_ids_no_speech = []
            cur_labels = labels[batch_idx]
            cur_labels_no_speech = []
            prev_index = -1

            for index in speech_token_indices + [cur_input_ids.shape[0]]:
                cur_input_ids_no_speech.append(cur_input_ids[prev_index + 1: index])
                cur_labels_no_speech.append(cur_labels[prev_index + 1: index])
                prev_index = index

            # Embed text segments
            split_sizes = [x.shape[0] for x in cur_labels_no_speech]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_no_speech))
            cur_input_embeds_no_speech = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            # Reconstruct with speech features
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_speech_idx = 0

            for i, index in enumerate(speech_token_indices):
                # Add text segment
                cur_new_input_embeds.append(cur_input_embeds_no_speech[i])
                cur_new_labels.append(cur_labels_no_speech[i])
                
                # Add speech features
                cur_speech_features = speech_features[cur_speech_idx].to(cur_input_embeds.dtype)
                cur_speech_idx += 1
                cur_new_input_embeds.append(cur_speech_features)
                cur_new_labels.append(torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, 
                                               device=cur_labels.device, dtype=cur_labels.dtype))

            # Add final text segment
            cur_new_input_embeds.append(cur_input_embeds_no_speech[-1])
            cur_new_labels.append(cur_labels_no_speech[-1])

            # Concatenate all segments
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate to max length
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Pad sequences
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, 
                                     dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), 
                              dtype=cur_new_embed.dtype, device=cur_new_embed.device), 
                    cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed, 
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), 
                              dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # Handle None values
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
            
        # Optimized position handling
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
            
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    # REMOVED: Legacy multimodal methods
    # def prepare_inputs_labels_for_multimodal(self, ...):
    # def prepare_inputs_labels_for_multimodal_av(self, ...):
    # def initialize_vision_tokenizer(self, ...):

    def initialize_speech_tokenizer(self, model_args, tokenizer):
        """Initialize speech-specific tokenizer"""
        if getattr(model_args, 'use_speech_token', True):
            num_new_tokens = tokenizer.add_tokens([DEFAULT_SPEECH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

