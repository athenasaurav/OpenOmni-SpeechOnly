"""
Speech-Only Optimization Configuration
This file contains all the optimization settings for the speech-only version of OpenOmni
"""

import torch
import os
from typing import Dict, Any

class SpeechOnlyConfig:
    """Configuration class for speech-only optimizations"""
    
    def __init__(self):
        # Performance optimizations
        self.enable_memory_optimization = True
        self.enable_fast_inference = True
        self.enable_streaming = True
        self.enable_chunked_processing = True
        
        # Model optimizations
        self.use_fp16 = True
        self.use_gradient_checkpointing = True
        self.freeze_speech_encoder = True
        self.optimize_attention = True
        
        # Memory settings
        self.max_memory_usage = "6GB"  # Reduced from 12GB+ in original
        self.batch_size = 1  # Optimized for low latency
        self.max_sequence_length = 2048
        
        # Speech processing settings
        self.speech_sample_rate = 16000
        self.speech_chunk_length = 30.0  # seconds
        self.speech_overlap = 0.5  # seconds
        self.enable_vad = True  # Voice Activity Detection
        
        # Streaming settings
        self.stream_chunk_size = 1024
        self.stream_overlap = 256
        self.stream_timeout = 5.0
        self.enable_real_time_processing = True
        
        # Generation settings
        self.max_new_tokens = 512
        self.temperature = 0.7
        self.top_p = 0.9
        self.do_sample = True
        self.num_beams = 1  # Greedy decoding for speed
        
        # Vocoder settings
        self.vocoder_sample_rate = 22050
        self.vocoder_hop_length = 256
        self.enable_vocoder_optimization = True
        
        # Hardware optimizations
        self.use_cuda = torch.cuda.is_available()
        self.device_map = "auto"
        self.torch_dtype = torch.float16
        
        # Removed features (for optimization)
        self.enable_vision = False
        self.enable_image_processing = False
        self.enable_video_processing = False
        
    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get model loading kwargs with optimizations"""
        return {
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "low_cpu_mem_usage": True,
            "attn_implementation": "flash_attention_2",
        }
    
    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get generation kwargs with optimizations"""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "pad_token_id": 0,  # Will be set properly during model loading
        }
    
    def get_speech_kwargs(self) -> Dict[str, Any]:
        """Get speech processing kwargs"""
        return {
            "sample_rate": self.speech_sample_rate,
            "chunk_length": self.speech_chunk_length,
            "overlap": self.speech_overlap,
            "enable_vad": self.enable_vad,
        }
    
    def get_streaming_kwargs(self) -> Dict[str, Any]:
        """Get streaming kwargs"""
        return {
            "chunk_size": self.stream_chunk_size,
            "overlap": self.stream_overlap,
            "timeout": self.stream_timeout,
            "enable_real_time": self.enable_real_time_processing,
        }
    
    def apply_optimizations(self, model):
        """Apply optimizations to the loaded model"""
        if self.enable_memory_optimization:
            # Enable gradient checkpointing
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            # Freeze speech encoder if enabled
            if self.freeze_speech_encoder and hasattr(model, 'get_model'):
                speech_encoder = getattr(model.get_model(), 'speech_encoder', None)
                if speech_encoder:
                    speech_encoder.eval()
                    for param in speech_encoder.parameters():
                        param.requires_grad = False
        
        # Set model to eval mode for inference
        model.eval()
        
        # Optimize for inference
        if self.enable_fast_inference:
            for param in model.parameters():
                param.requires_grad = False
        
        return model
    
    def setup_environment(self):
        """Setup environment variables for optimization"""
        if self.enable_memory_optimization:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        
        if self.enable_fast_inference:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def get_memory_stats(self):
        """Get current memory usage statistics"""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,  # GB
            }
        return {"message": "CUDA not available"}
    
    def optimize_for_deployment(self):
        """Additional optimizations for deployment"""
        # Disable debugging features
        torch.autograd.set_grad_enabled(False)
        
        # Set optimal thread counts
        torch.set_num_threads(4)  # Reduced for memory efficiency
        
        # Enable optimized kernels
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True


# Global configuration instance
speech_config = SpeechOnlyConfig()

# Convenience functions
def get_optimized_model_kwargs():
    return speech_config.get_model_kwargs()

def get_optimized_generation_kwargs():
    return speech_config.get_generation_kwargs()

def apply_speech_optimizations(model):
    return speech_config.apply_optimizations(model)

def setup_optimized_environment():
    speech_config.setup_environment()

def get_memory_usage():
    return speech_config.get_memory_stats()

# Performance monitoring
class PerformanceMonitor:
    """Monitor performance metrics for speech-only mode"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_latency": 0.0,
            "total_processing_time": 0.0,
        }
    
    def record_request(self, success: bool, latency: float):
        """Record a request with its outcome and latency"""
        self.metrics["total_requests"] += 1
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        self.metrics["total_processing_time"] += latency
        self.metrics["average_latency"] = (
            self.metrics["total_processing_time"] / self.metrics["total_requests"]
        )
    
    def get_stats(self):
        """Get current performance statistics"""
        success_rate = (
            self.metrics["successful_requests"] / max(self.metrics["total_requests"], 1) * 100
        )
        return {
            **self.metrics,
            "success_rate": success_rate,
            "memory_usage": get_memory_usage(),
        }
    
    def reset_stats(self):
        """Reset all statistics"""
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_latency": 0.0,
            "total_processing_time": 0.0,
        }

# Global performance monitor
performance_monitor = PerformanceMonitor()

