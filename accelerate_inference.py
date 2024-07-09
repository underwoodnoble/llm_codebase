import json
from tqdm import tqdm
from argparse import ArgumentParser
from typing import List

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer

from src.models.RewardModel import QwenRewardModel


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        '--task_type', type=str, choices=['rm', 'llm']
    )
    parser.add_argument(
        '--model_type', type=str
    )
    parser.add_argument(
        '--model_name_or_path', type=str
    )
    parser.add_argument(
        '--model_max_length', type=int
    )
    parser.add_argument(
        '--input_file_path', type=str
    )
    parser.add_argument(
        '--batch_size', type=int
    )
    parser.add_argument(
        '--save_path', type=str
    )
    parser.add_argument(
        '--dtype', type=str, default='fp16'
    )
    return parser


def load_json_or_jsonl(file_path: str):
    with open(file_path, 'r') as f:
        if file_path.endswith('json'):
            dataset = json.load(f)
        elif file_path.endswith('jsonl'):
            dataset = [json.loads(line) for line in f.readlines()]
    
    print(f'Finish load {len(dataset)} pieces of data from {file_path}.')
    return dataset


def reward_model_inference(dataset: List, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, batch_size: int, save_path: str):
    for data in dataset:
        texts = [
            data['prompt'] + data['answers'][i] for i in range(len(data['answers']))
        ]
        rewards = []
        for i in range(0, len(texts), batch_size):
            encoding = tokenizer(texts[i:i+batch_size], return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                rewards.extend(
                    model(input_ids=encoding['input_ids'].to(model.device)).rm_logits.flatten().tolist()
                )
        new_data = {
            "prompt": data['prompt'],
            "answers": data['answers'],
            "rewards": rewards
        }
        with open(save_path, 'a') as f:
            f.write(json.dumps(new_data, ensure_ascii=False) + '\n')

    

def main(args):
    dataset = load_json_or_jsonl(args.input_file_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.model_max_length = args.model_max_length

    TYPE_MAP = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32
    }
    if args.task_type == 'rm':
        if args.model_type == 'qwen':
            model = QwenRewardModel.from_pretrained(args.model_name_or_path, device_map='auto', torch_dtype=TYPE_MAP[args.dtype])
        
        model.eval()
        reward_model_inference(dataset, tokenizer, model, args.batch_size, args.save_path)
    else:
        pass
    

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)