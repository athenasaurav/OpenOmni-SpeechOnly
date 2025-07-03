# OpenOmni-Zen: Complete Setup Guide (Updated & Tested)

**Status:** ✅ **WORKING CONFIGURATION**  
**Last Updated:** July 2, 2025  
**Repository:** [OpenOmni-Zen](https://github.com/athenasaurav/OpenOmni.git)

> **Note:** This guide is based on a successful working setup and addresses common issues found in the original documentation.

## 🎯 Overview

OpenOmni-Zen is a fully open-source implementation of a GPT-4o-like speech-to-speech video understanding model. This guide provides step-by-step instructions from git clone to a working Gradio interface with public sharing.

**Key Features:**
- Real-time speech-to-speech synthesis
- Multimodal understanding (speech + vision + text)
- Streaming responses
- Web-based interface with public sharing

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 4090/A6000 minimum, H100 recommended)
- **RAM**: 32GB+ system RAM
- **Storage**: 150GB+ free space
- **Internet**: Stable connection for model downloads

### Software Requirements
- **OS**: Linux (Ubuntu 20.04+) or Windows WSL2
- **CUDA**: 11.8 or compatible
- **Python**: 3.10
- **Git LFS**: For large model files

## 🚀 Step 1: Repository Setup

### 1.1 Clone Repository
```bash
git clone https://github.com/athenasaurav/OpenOmni.git
cd OpenOmni
```

### 1.2 Create Environment
```bash
conda create -n omni python=3.10 -y
conda activate omni
```

### 1.3 Install Core Dependencies
```bash
# Install PyTorch with CUDA 11.8
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

# Install package
pip install -e ".[train]"

# Install Flash Attention
pip install flash-attn==2.6.3 --no-build-isolation --no-cache-dir

## If Flash Attention doesnt install like this then try 
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install remaining requirements
pip install -r requirements.txt

# Install Fairseq
pip install fairseq

#If Fairseq installation fails then try
pip install pip==24
pip install fairseq

# Upgrade Gradio (IMPORTANT!)
pip install --upgrade gradio
```

## 📦 Step 2: Model Downloads

### 2.1 Create Directory Structure
```bash
# Create main checkpoints directory in project root
mkdir -p checkpoints
cd checkpoints
```

### 2.2 Download Backbone Models

#### Language Model (Choose One)
```bash
# Option A: Qwen2-7B-Instruct (Recommended)
git lfs clone https://huggingface.co/Qwen/Qwen2-7B-Instruct
 
# Option B: Llama-3.1-8B-Instruct For this 
git clone https://<your_username>:<your_token>@huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
```

#### Vision Encoder
```bash
git lfs clone https://huggingface.co/openai/clip-vit-large-patch14-336
```

#### Speech Encoder
```bash
mkdir -p whisper
cd whisper
wget https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt
cd ..
```

#### Vocoder
Download the HiFi-GAN vocoder:
```bash
mkdir -p vocoder
cd vocoder
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json
# The exact download links may need to be obtained from the project maintainers
cd ..
```

### 2.3 Download OpenOmni-Zen Checkpoint
```bash
# Download the main pretrained model
git lfs clone https://huggingface.co/ColorfulAI/OpenOmni-7B-Qwen2-Omni
```

### 2.4 Verify Directory Structure
Your `checkpoints/` directory should look like:
```
checkpoints/
├── Qwen2-7B-Instruct/              # or Meta-Llama-3.1-8B-Instruct/
├── clip-vit-large-patch14-336/
├── whisper/
│   └── large-v3.pt
├── vocoder/
│   ├── config.json
│   └── g_00500000
└── OpenOmni-7B-Qwen2-Omni/
```

## ⚙️ Step 3: Launch System

### 3.1 Terminal 1: Controller
```bash
cd OpenOmni
conda activate omni

python -m local_demo.controller --host 0.0.0.0 --port 10000
```

### 3.2 Terminal 2: Model Worker
```bash
cd OpenOmni
conda activate omni

python -m local_demo.model_worker \
    --host 0.0.0.0 \
    --controller http://localhost:10000 \
    --port 40000 \
    --worker http://localhost:40000 \
    --model-path checkpoints/OpenOmni-7B-Qwen2-Omni \
    --model-name llava_s2s_qwen
```

### 3.3 Terminal 3: Gradio Web Server
```bash
cd OpenOmni
conda activate omni

python -m local_demo.gradio_web_server \
    --controller-url http://localhost:10000 \
    --port 8000 \
    --model-list-mode reload \
    --vocoder checkpoints/vocoder/g_00500000 \
    --vocoder-cfg checkpoints/vocoder/config.json \
    --share
```

## 🌐 Step 4: Access Interface

After successful launch, you'll see:
```
Running on local URL:  http://localhost:8000
Running on public URL: https://xxxxxxxxxxxxxxxx.gradio.live
```

- **Local Access**: http://localhost:8000
- **Public Access**: Use the gradio.live URL (expires in 72 hours)

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. Model Loading Errors
```bash
# Check if models exist
ls -la checkpoints/OpenOmni-7B-Qwen2-Omni/

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

#### 2. HuggingFace Access Attempts
**Problem**: System tries to download from HuggingFace instead of using local models.
**Solution**: Ensure you specify `--model-path` and `--model-name` parameters correctly.

#### 3. Vision Tower Errors
**Problem**: `ValueError: Unknown vision tower: checkpoints/clip-vit-large-patch14-336`
**Solution**: Ensure CLIP model is properly downloaded with LFS:
```bash
cd checkpoints/clip-vit-large-patch14-336
git lfs pull
ls -lh pytorch_model.bin  # Should be ~1.6GB, not 135 bytes
```

#### 4. Port Conflicts
```bash
# Check if ports are in use
netstat -tulpn | grep :10000
netstat -tulpn | grep :8000
netstat -tulpn | grep :40000

# Kill processes if needed
sudo kill -9 $(lsof -t -i:10000)
```

#### 5. Memory Issues
```bash
# Monitor GPU memory
nvidia-smi

# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Model File Size Verification
Ensure your model files are properly downloaded:
```bash
# CLIP model should be ~1.6GB
ls -lh checkpoints/clip-vit-large-patch14-336/pytorch_model.bin

# Whisper should be ~2.9GB
ls -lh checkpoints/whisper/large-v3.pt

# OpenOmni model should be several GB
du -sh checkpoints/OpenOmni-7B-Qwen2-Omni/
```

## 🎮 Testing the System

### Basic Functionality Test
1. **Upload Image/Video**: Use the file upload interface
2. **Record Audio**: Click microphone and ask about the uploaded content
3. **Submit**: Click submit and wait for response
4. **Verify Streaming**: Check that text and audio stream progressively

### Speech-to-Speech Test
1. Upload an image
2. Record speech: "What do you see in this image?"
3. Verify you get both text and audio responses
4. Test streaming by asking follow-up questions

## 📝 Key Differences from Original README

### What We Fixed:
1. **Model Directory**: Models must be in root `checkpoints/`, not `open_omni/checkpoints/`
2. **Required Parameters**: `--model-path` and `--model-name` are mandatory
3. **Gradio Upgrade**: Must upgrade gradio for compatibility
4. **Working Model Source**: Use `ColorfulAI/OpenOmni-7B-Qwen2-Omni` from HuggingFace
5. **Correct Port Configuration**: Specific ports that work together
6. **Share Parameter**: Built-in `--share` flag instead of code modification

### Command Structure Clarification:
The original README's demo commands were incomplete. The working commands require:
- Explicit model path specification
- Correct port assignments
- Proper vocoder configuration
- Model name parameter

## 🔒 Security & Production Notes

### Public Sharing Considerations:
- **Temporary Links**: Gradio share links expire in 72 hours
- **Public Access**: Anyone with the link can use your model
- **Rate Limiting**: Consider implementing for production use
- **Data Privacy**: Be cautious with sensitive inputs

### Production Deployment:
- Use dedicated servers with proper security
- Implement authentication and rate limiting
- Consider Docker containerization
- Monitor resource usage and costs

## 📊 Performance Expectations

### Current Model Status:
⚠️ **Important**: The available checkpoints are undertrained and may exhibit unpredictable behavior. This is acknowledged in the original repository.

### Expected Performance:
- **Response Time**: 5-15 seconds depending on input complexity
- **GPU Memory**: 12-16GB VRAM usage
- **Streaming**: Progressive text and audio generation
- **Quality**: Proof-of-concept level, not production-ready

## 🛠️ Advanced Configuration

### For Llama Models:
If using Llama instead of Qwen, modify the template in `local_demo/gradio_web_server.py`:
```python
# Line 115: Change template_name from 'qwen_1_5' to 'llava_llama_3'
```

### Custom Vocoder:
```bash
# Use different vocoder if needed
--vocoder your/vocoder/model
--vocoder-cfg your/vocoder/config.json
```

### Memory Optimization:
```bash
# Reduce memory usage
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## 🎯 Success Indicators

You know the setup is working when:
- ✅ All three terminals show "ready" status
- ✅ Gradio interface loads without errors
- ✅ You can upload images/videos
- ✅ Audio recording works
- ✅ Responses include both text and speech
- ✅ Streaming works progressively
- ✅ Public share link is accessible

## 📞 Support & Resources

- **Repository**: [OpenOmni-Zen GitHub](https://github.com/OmniMMI/OpenOmni-Zen)
- **Model Source**: [ColorfulAI/OpenOmni-7B-Qwen2-Omni](https://huggingface.co/ColorfulAI/OpenOmni-7B-Qwen2-Omni)
- **Issues**: Report problems on the GitHub repository
- **Community**: Check discussions for additional tips and solutions

---

**Note**: This guide represents a working configuration as of July 2025. The OpenOmni-Zen project is actively developed, so commands and requirements may change. Always refer to the official repository for the latest updates.

