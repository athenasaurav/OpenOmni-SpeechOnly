#!/bin/bash
#SBATCH --job-name=PT
#SBATCH --partition=HGX,DGX
#SBATCH --account=research
#SBATCH --qos=lv1
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=./slurm_logs/pretrain-llama3-7b-bliplaion.out
#SBATCH --error=./slurm_logs/pretrain-llama3-7b-bliplaion.error.out


export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=4
# export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO


export NUM_GPUS=4
MASTER_PORT=$(expr $RANDOM + 1000)
export PORT=$MASTER_PORT

export PYTHONPATH=$(pwd)
echo $PYTHONPATH

# LLM_VERSION="checkpoints/Qwen2.5-7B-Instruct"
LLM_VERSION="checkpoints/Meta-Llama-3.1-8B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="checkpoints/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION=plain

BASE_RUN_NAME="bliplaion-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

module add cuda11.8

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${PORT}" \
    open_omni/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path inputs/texts/blip_laion_cc_sbu_558k.json \
    --image_folder inputs/images/blip_laion_cc_sbu_558k \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (336, 1008), (336, 1344), (336, 1680), (336, 2016), (336, 2352), (336, 2688), (336, 3024), (336, 3360), (336, 3696), (336, 4032), (336, 4368), (336, 4704), (336, 5040), (336, 5376), (336, 5712), (336, 6048), (336, 6384), (336, 6720), (336, 7056), (336, 7392), (336, 7728), (336, 8064), (336, 8400), (336, 8736), (336, 9072), (336, 9408), (336, 9744), (336, 10080), (336, 10416), (336, 10752), (336, 11088), (336, 11424), (336, 11760), (336, 12096), (336, 12432), (336, 12768), (336, 13104), (336, 13440), (336, 13776), (336, 14112), (336, 14448), (336, 14784), (336, 15120), (336, 15456), (336, 15792), (336, 16128), (336, 16464), (672, 336), (672, 672), (672, 1008), (672, 1344), (672, 1680), (672, 2016), (672, 2352), (672, 2688), (672, 3024), (672, 3360), (672, 3696), (672, 4032), (672, 4368), (672, 4704), (672, 5040), (672, 5376), (672, 5712), (672, 6048), (672, 6384), (672, 6720), (672, 7056), (672, 7392), (672, 7728), (672, 8064), (1008, 336), (1008, 672), (1008, 1008), (1008, 1344), (1008, 1680), (1008, 2016), (1008, 2352), (1008, 2688), (1008, 3024), (1008, 3360), (1008, 3696), (1008, 4032), (1008, 4368), (1008, 4704), (1008, 5040), (1008, 5376), (1344, 336), (1344, 672), (1344, 1008), (1344, 1344), (1344, 1680), (1344, 2016), (1344, 2352), (1344, 2688), (1344, 3024), (1344, 3360), (1344, 3696), (1344, 4032), (1680, 336), (1680, 672), (1680, 1008), (1680, 1344), (1680, 1680), (1680, 2016), (1680, 2352), (1680, 2688), (1680, 3024), (2016, 336), (2016, 672), (2016, 1008), (2016, 1344), (2016, 1680), (2016, 2016), (2016, 2352), (2016, 2688), (2352, 336), (2352, 672), (2352, 1008), (2352, 1344), (2352, 1680), (2352, 2016), (2352, 2352), (2688, 336), (2688, 672), (2688, 1008), (2688, 1344), (2688, 1680), (2688, 2016), (3024, 336), (3024, 672), (3024, 1008), (3024, 1344), (3024, 1680), (3360, 336), (3360, 672), (3360, 1008), (3360, 1344), (3696, 336), (3696, 672), (3696, 1008), (3696, 1344), (4032, 336), (4032, 672), (4032, 1008), (4032, 1344), (4368, 336), (4368, 672), (4368, 1008), (4704, 336), (4704, 672), (4704, 1008), (5040, 336), (5040, 672), (5040, 1008), (5376, 336), (5376, 672), (5376, 1008), (5712, 336), (5712, 672), (6048, 336), (6048, 672), (6384, 336), (6384, 672), (6720, 336), (6720, 672), (7056, 336), (7056, 672), (7392, 336), (7392, 672), (7728, 336), (7728, 672), (8064, 336), (8064, 672), (8400, 336), (8736, 336), (9072, 336), (9408, 336), (9744, 336), (10080, 336), (10416, 336), (10752, 336), (11088, 336), (11424, 336), (11760, 336), (12096, 336), (12432, 336), (12768, 336), (13104, 336), (13440, 336), (13776, 336), (14112, 336), (14448, 336), (14784, 336), (15120, 336), (15456, 336), (15792, 336), (16128, 336), (16464, 336)]" \
    --mm_patch_merge_type unires \
    --bf16 True \
    --output_dir checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name $BASE_RUN_NAME \
    # --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn