import os
import torch
from transformers import AutoModelForCausalLM,AutoProcessor
import json
from collections import defaultdict
from tqdm import tqdm
import argparse
import sys

from transformers import logging
logging.set_verbosity(logging.ERROR)


def load_data_and_create_task_datasets(jsonl_file):
    """加载数据并按task创建dataset字典"""
    task_groups = defaultdict(list)
    
    # 加载jsonl数据
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            
            scene = item.get("scene", "unknown")
            # Extract required fields
            
            ts_path = item["ts_path"]
            sr = item["sr"]
            conversations = item["conversations"]

            # Build messages format
            messages = []

            # user message
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "time_series",
                        "data": ts_path,
                        "sampling_rate": sr,
                    },
                    {"type": "text", "text": conversations[0]["value"]},
                ],
            }
            messages.append(message)

            # Build final entry
            new_entry = {
                "id": item["id"],                # keep id unchanged
                'uid': item['uid'],
                'dataset_name': item['dataset_name'],
                'task': item['task'],
                'scene': item['scene'],
                'ts_path': item['ts_path'],
                "messages": messages,
                'input_text': item['conversations'][0]['value'],  # question,
                'ground_truth': item['conversations'][1]['value'],   # ground truth,
                'gt_result': item.get('gt_result',None),
                'num_tokens': item.get('num_ts_tokens', 0),
                "sr": item.get('sr',None)
            }

            # 按task分组
            task_groups[scene].append(new_entry)
    
    return task_groups



def save_results(results, output_file):
    """保存结果到jsonl文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"Results saved to {output_file}")



def main():
    parser = argparse.ArgumentParser(description="Time Series LLM Evaluation with DDP")
    parser.add_argument("--model_path", type=str, 
                       default="/mnt/shared-storage-gpfs2/speechllm-share/wuwen/interns1_1_ts/InternS1_1_1T_A22_1217",
                       help="Path to the model")
    parser.add_argument("--data_file_folder", type=str, required=True,
                       help="Path to the JSONL data file folder")
    parser.add_argument("--output_folder", type=str, required=True,
                       help="Folder to save the results")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                       help="Max new tokens for generation")
    parser.add_argument("--num_workers", type=int, default=16,
                       help="Number of workers for DataLoader")
    
    args = parser.parse_args()
    model_path=args.model_path
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            # attn_implementation="flash_attention_2",  #时序暂不支持flash_attn，load加这行会报错
            trust_remote_code=True
        )

    if os.path.isfile(args.data_file_folder):
        data_files = [args.data_file_folder]  # 单个文件
    elif os.path.isdir(args.data_file_folder):
        data_files = [os.path.join(args.data_file_folder, x) 
                    for x in os.listdir(args.data_file_folder)]
    else:
        raise ValueError(f"{args.data_file_folder} is not a file or directory")

    
    for data_file in data_files:
        # Load dataset and create task datasets
        print(f"Loading dataset from {data_file}...")
        
        task_groups = load_data_and_create_task_datasets(data_file)

        total_samples = sum(len(dataset) for dataset in task_groups.values())
        print(f"Loaded {total_samples} samples")
        print(f"Found {len(task_groups)} unique tasks")
        
        # Run inference
        print("Starting inference...")
        print(model.dtype)

        results=[]
        for scene, samples in task_groups.items():
            print(f"Processing {scene}")
            for sample in tqdm(samples, total=len(samples),desc=scene,
                                file=sys.stdout,
                                disable=False,
                                leave=True,
                                mininterval=5,
                                dynamic_ncols=False,):
                messages=sample['messages']
                # messages = [
                #     {
                #         "role": "user",
                #         "content": [
                #             {"type": "time_series", "data": f"{model_path}/0092638_seism.npy", "sampling_rate": 100},
                #             {"type": "text", "text": "Please determine whether an Earthquake event has occurred in the provided time-series data. If so, please specify the starting time point indices of the P-wave and S-wave in the event."},
                #         ],
                #     }
                # ]
                time_series_inputs = processor.time_series_preprocessor(messages)
                multimodal_inputs = processor.apply_chat_template(messages, 
                                                                  add_generation_prompt=True, 
                                                                  tokenize=True, 
                                                                  return_dict=True, 
                                                                  return_tensors="pt", 
                                                                  enable_thinking=False, 
                                                                  **time_series_inputs).to(model.device, dtype=torch.bfloat16)               
                with torch.inference_mode():
                    multimodal_generated_ids = model.generate(
                        **multimodal_inputs,
                        max_new_tokens=200,
                        do_sample=False,
                        temperature=1.0,
                    )
                multimodal_generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(multimodal_inputs.input_ids, multimodal_generated_ids)
                ]

                decoded_output = processor.batch_decode(
                    multimodal_generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

                results.append({
                    'id': sample['id'],
                    'uid': sample['uid'],
                    'dataset_name': sample['dataset_name'],
                    'ts_path': sample['ts_path'],
                    'input_text': sample['input_text'],
                    'task': sample['task'],
                    'scene': sample['scene'],
                    'generated_text': decoded_output[0].strip(),
                    'ground_truth': sample['ground_truth'],
                    'gt_result': sample['gt_result'],
                    'num_tokens': sample['num_tokens'],
                })
        base_name = os.path.basename(data_file)
        output_file = os.path.join(args.output_folder, base_name)

        # Save results
        save_results(results, output_file)
            
if __name__ == "__main__":
    main()
