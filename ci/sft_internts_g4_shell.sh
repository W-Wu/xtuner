
set -ex
cd /mnt/shared-storage-user/brainllm-share/wuwen/times_v2/xtuner

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export XTUNER_TOKENIZE_WORKERS=16
export XTUNER_USE_FA3=1
export XTUNER_DECORD_VIDEO_THREADS=16
export XTUNER_SKIP_EMPTY_THINK=1

export GLOBAL_BATCH_SIZE=8
export WORK_DIR="/mnt/shared-storage-gpfs2/speechllm-share/wuwen/times_v2_exp_xtuner/qwen3vl_sft_ts_all_en_zh"
export PYTHONPATH="$(pwd)"

CONFIG_PATH="ci/sft_interns1_g4_ts_config.py"

current_time=$(date "+%m%d%H%M")
if [ ! -d "$WORK_DIR" ]; then
  mkdir -p "$WORK_DIR"
fi

SCRIPT_NAME=$(basename "$0")
cp "$0" "${WORK_DIR}/${SCRIPT_NAME}"

# -m debugpy --connect 5680
torchrun \
    --nnodes=$NODE_COUNT \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$PROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    xtuner/v1/train/cli/sft.py \
    --config $CONFIG_PATH \
    2>&1 | tee -a "${WORK_DIR}/training_log_${current_time}.txt"
