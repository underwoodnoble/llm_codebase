import datasets
import os
import psutil
import timeit
from src.utils.general_utils import read_json_or_jsonl_data


mem_before  = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

dataset = datasets.load_dataset('json', data_files=["/apdcephfs_cq10/share_1567347/share_info/nobelhu/code/llm_codebase/mydata/T/response_training_data/t_response_training_data_v0.13_step2_inference_preview.jsonl"], streaming=False)
#dataset = read_json_or_jsonl_data("/apdcephfs_cq10/share_1567347/share_info/nobelhu/code/llm_codebase/mydata/adversarial_game/pengyucheng/keyword_freq30k_game_results/keyword_freq30k_game_results.json")
mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

print(mem_after - mem_before, "MB")
print(dataset)