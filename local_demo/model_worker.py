"""
Optimized Speech-Only Model Worker
Removed video processing for reduced latency and memory usage
"""
import argparse
import asyncio
import json
import time
import threading
import uuid
import datetime
import gc

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
import whisper
import numpy as np
from functools import partial

from transformers import PreTrainedTokenizer

from open_omni.constants import WORKER_HEART_BEAT_INTERVAL
from open_omni.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from open_omni.model.builder import load_pretrained_model
from open_omni.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN
from open_omni.mm_utils_speech import tokenizer_speech_tokens, KeywordsStoppingCriteria
from transformers import TextIteratorStreamer
from threading import Thread


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker_speech_only", f"model_worker_speech_only_{worker_id}.log")
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


def load_speech_optimized(audio, input_type, mel_size, speech_normalize):
    """
    Optimized speech loading with memory management
    """
    speech = np.array(audio, dtype=np.float32)
    
    with torch.no_grad():
        if input_type == "raw":
            speech = torch.from_numpy(speech)
            if speech_normalize:
                speech = torch.nn.functional.layer_norm(speech, speech.shape)
        elif input_type == "mel":
            speech = whisper.pad_or_trim(speech)
            speech = whisper.log_mel_spectrogram(speech, n_mels=mel_size).permute(1, 0)
    
    return speech


def build_unit_tokenizer(vocab_size):
    """Optimized unit tokenizer builder"""
    import os
    from transformers import BertTokenizer
    
    vocab_file = f"unit_vocab_{worker_id}.txt"
    try:
        with open(vocab_file, "w") as f:
            for i in range(vocab_size + 1):
                f.write(str(i) + "\n")
        tokenizer = BertTokenizer(vocab_file=vocab_file)
    finally:
        if os.path.exists(vocab_file):
            os.remove(vocab_file)
    
    return tokenizer


class SpeechOnlyModelWorker:
    """
    Optimized Speech-Only Model Worker
    - Removed video processing capabilities
    - Optimized memory usage
    - Reduced latency through streamlined pipeline
    """
    
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_base, model_name,
                 load_8bit, load_4bit, device, input_type, mel_size, 
                 attn_implementation="flash_attention_2"):
        
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.device = device
        self.model_name = model_name
        self.input_type = input_type
        self.mel_size = mel_size
        
        logger.info("Loading optimized speech-only model...")
        
        # Load model with optimizations (image_processor will be None for speech-only)
        self.tokenizer, self.model, _, self.context_len = load_pretrained_model(
            model_path, model_base, model_name, 
            load_8bit=load_8bit, load_4bit=load_4bit, 
            device_map=self.device, 
            attn_implementation=attn_implementation
        )
        
        # Optimize model for inference
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Enable memory optimizations
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        self.unit_tokenizer = build_unit_tokenizer(self.model.config.unit_vocab_size)
        
        logger.info("Speech-only model loaded successfully")
        
        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,), daemon=True)
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register speech-only worker to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @staticmethod
    def utc_now_str():
        return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"

    @torch.inference_mode()
    def generate_stream_speech_only(self, params):
        """
        Optimized speech-only generation with reduced latency
        - Removed video processing
        - Optimized memory management
        - Streamlined speech pipeline
        """
        tokenizer, model = self.tokenizer, self.model

        t_request_received = datetime.datetime.utcnow()
        logger.info(f"TTFB-LOG: [speech_worker] Request received at {self.utc_now_str()}")
        ttfb_text = None
        ttfb_units = None

        prompt = params["prompt"]
        ori_prompt = prompt
        
        # Optimized audio processing - removed video handling
        audio = params.get("audio", None)
        if audio is not None and len(audio) > 0:
            speech = load_speech_optimized(audio, self.input_type, self.mel_size, 
                                         self.model.config.speech_normalize)
            speech_length = torch.LongTensor([speech.shape[0]]).unsqueeze(0).to(self.device)
            speech_tensor = speech.unsqueeze(0).to(self.device, dtype=torch.float16)
            mm_args = {"speeches": speech_tensor, "speech_lengths": speech_length}
        else:
            speech = None
            mm_args = {}
        
        # Removed video processing entirely for speech-only optimization
        # Original video processing code removed to reduce latency
        
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False

        # Optimized tokenization - speech-only processing
        input_ids = tokenizer_speech_tokens(
            prompt, tokenizer, SPEECH_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(self.device)
        
        # Optimized streamers with reduced timeout for faster response
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10)
        streamer_unit = TextIteratorStreamer(self.unit_tokenizer, skip_prompt=False, skip_special_tokens=True, timeout=10)

        if max_new_tokens < 1:
            yield json.dumps({
                "text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", 
                "error_code": 0
            }).encode() + b"\0"
            return

        # Optimized generation with memory management
        generation_kwargs = dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            streamer_unit=streamer_unit,
            streaming_unit_gen=True,
            use_cache=True,
            **mm_args
        )
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ori_prompt
        
        try:
            for new_text in streamer:
                generated_text += new_text
                generated_unit = " ".join(map(str, streamer_unit.token_cache))
                
                # TTFB tracking for performance optimization
                if ttfb_text is None and new_text:
                    ttfb_text = datetime.datetime.utcnow()
                    logger.info(f"TTFB-LOG: [speech_worker] First text at {self.utc_now_str()} "
                              f"(delta: {(ttfb_text-t_request_received).total_seconds():.3f}s)")
                
                if ttfb_units is None and streamer_unit.token_cache:
                    ttfb_units = datetime.datetime.utcnow()
                    logger.info(f"TTFB-LOG: [speech_worker] First units at {self.utc_now_str()} "
                              f"(delta: {(ttfb_units-t_request_received).total_seconds():.3f}s)")
                
                if stop_str and generated_text.endswith(stop_str):
                    generated_text = generated_text[:-len(stop_str)]
                
                yield json.dumps({
                    "text": generated_text, 
                    "unit": generated_unit, 
                    "error_code": 0
                }).encode() + b"\0"
        
        finally:
            # Memory cleanup
            if 'speech_tensor' in locals():
                del speech_tensor
            if 'input_ids' in locals():
                del input_ids
            torch.cuda.empty_cache()
            gc.collect()

    def generate_stream_gate(self, params):
        """Enhanced error handling for speech-only generation"""
        try:
            for x in self.generate_stream_speech_only(params):
                yield x
        except ValueError as e:
            logger.error(f"ValueError in speech generation: {e}")
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            logger.error(f"CUDA error in speech generation: {e}")
            torch.cuda.empty_cache()
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            logger.error(f"Unknown error in speech generation: {e}")
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--input-type", type=str, default="mel")
    parser.add_argument("--mel-size", type=int, default=128)
    args = parser.parse_args()
    logger.info(f"Speech-only worker args: {args}")

    worker = SpeechOnlyModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.no_register,
        args.model_path,
        args.model_base,
        args.model_name,
        args.load_8bit,
        args.load_4bit,
        args.device,
        args.input_type,
        args.mel_size,
        attn_implementation=args.attn_implementation
    )
    
    logger.info("Starting optimized speech-only model worker...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

