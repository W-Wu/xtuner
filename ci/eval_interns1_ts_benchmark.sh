#!/bin/bash

# 时序数据推理评估脚本

source /home/wuwen/miniconda3/etc/profile.d/conda.sh
conda activate time
echo "Current Python environment:"
which python

infer=true
eval=true


MODEL_PATH="/mnt/shared-storage-gpfs2/speechllm-share/wuwen/times_v2_exp_xtuner/qwen3vl_sft_ts_4task/20260113005136/hf-1650"
TEST_DATA_FOLDER=/mnt/shared-storage-user/brainllm-share/wuwen/times_v2/scp/test_data
infer_folder=${MODEL_PATH}_benchmark/eval_results
eval_output=$infer_folder-eval_results.csv
BATCH_SIZE=1
NUM_WORKERS=8

if [ ! -d "$infer_folder" ]; then
  mkdir -p "$infer_folder"
fi

echo "Starting time series LLM evaluation..."
echo "Data file: $TEST_DATA_FOLDER"
echo "Output file: $infer_folder"
echo "Model path: $MODEL_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Number of workers: $NUM_WORKERS"

if [ "$infer" = true ]; then
    python \
        /mnt/shared-storage-user/brainllm-share/wuwen/times_v2/xtuner/ci/eval_interns1_g4_ts.py \
        --data_file_folder "$TEST_DATA_FOLDER" \
        --output_folder "$infer_folder" \
        --model_path "$MODEL_PATH" \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS 
        2>&1 | tee -a "${infer_folder}/eval_log.txt"
    echo "Inference completed! Results saved to $infer_folder"
fi

# if [ "$eval" = true ]; then
python /mnt/shared-storage-user/brainllm-share/wuwen/times_v2/xtuner/ci/time_series_llm_eval-compute-metric-benchmark.py "$infer_folder" --output "$eval_output"
echo "Evaluation completed! Results saved to $eval_output"
# fi

