import torch
import json
from tqdm import tqdm


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def read_json_or_jsonl_data(data_path: str):
    if data_path.endswith('json'):
        with open(data_path, 'r') as f:
            data_list = json.load(f)
    elif data_path.endswith('jsonl'):
        with open(data_path, 'r') as f:
            lines = f.read().strip().split('\n')
            data_list = [json.loads(l) for l in lines]
    else:
        raise ValueError("The data file must end with json or jsonl.")
    
    print_rank_0(f">>> totally load {len(data_list)} data from {data_path}.")
    return data_list

    
def load_data_from_paths(data_paths: str):
    total_data_list = []
    i = 0
    for data_path in data_paths:
        data_list = read_json_or_jsonl_data(data_path)
        for data in tqdm(data_list):
            data['id'] = i
            i += 1
            total_data_list.append(data)
    print_rank_0(f">>> total load {len(total_data_list)} data.")
    return total_data_list

    