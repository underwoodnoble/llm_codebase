import json
import yaml
from tqdm import tqdm
from argparse import ArgumentParser
from typing import List

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModelForCausalLM, GenerationConfig

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
        '--generation_config_path', type=str
    )
    parser.add_argument(
        '--save_path', type=str
    )
    parser.add_argument(
        '--dtype', type=str, default='fp16'
    )
    parser.add_argument(
        '--add_special_tokens', type=bool, default=False
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


@torch.no_grad()
def reward_model_inference(dataset: List, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, batch_size: int, save_path: str, add_special_tokens=False):
    for data in dataset:
        if 'prompt' in data:
            texts = [
                data['prompt'] + data['answers'][i] for i in range(len(data['answers']))
            ]
        elif 'text' in data:
            texts = data['text']
        elif 'texts' in data:
            texts = data['texts']
        else:
            print(data)
            raise ValueError("wrong data!")

        rewards = []
        for i in range(0, len(texts), batch_size):
            encoding = tokenizer(texts[i:i+batch_size], return_tensors='pt', padding=True, truncation=True, add_special_tokens=add_special_tokens)
            rewards.extend(
                model(input_ids=encoding['input_ids'].to(model.device)).rm_logits.flatten().tolist()
            )
        
        data['rewards'] = rewards
        with open(save_path, 'a') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


@torch.no_grad()
def llm_inference(dataset: List, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, batch_size: int, generation_config: GenerationConfig, save_path: str, add_special_tokens=False):
    bos_token = tokenizer.bos_token if add_special_tokens and tokenizer.bos_token is not None else ""
    if 'prompt' in dataset[0]:
        prompts = [bos_token + data['prompt'] for data in dataset]
    elif 'query' in dataset[0]:
        prompts = [bos_token + data['query'] for data in dataset]
    elif 'input' in dataset[0]:
        prompts = [bos_token + data['input'] for data in dataset]
    else: 
        raise ValueError('wrong data!')
    

    for i in range(0, len(prompts), batch_size):
        encoding = tokenizer(prompts[i:i+batch_size], add_special_tokens=False, padding=True, truncation=True, return_tensors='pt')
        ret = model.generate(encoding['input_ids'].to(model.device), attention_mask=encoding['attention_mask'].to(model.device), generation_config=generation_config)
        responses = tokenizer.batch_decode(ret, skip_special_tokens=True)
        print(responses)
        with open(save_path, 'a') as f:
            for j, prompt in enumerate(prompts[i:i+batch_size]):
                prompt_responses = responses[j*generation_config.num_return_sequences:(j+1)*generation_config.num_return_sequences]
                f.write(json.dumps({
                    'prompt': prompt,
                    "responses": prompt_responses
                }, ensure_ascii=False) + '\n')


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
        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'left'
        if args.model_type == 'qwen':
            model = QwenRewardModel.from_pretrained(args.model_name_or_path, device_map='auto', torch_dtype=TYPE_MAP[args.dtype])
        
        model.eval()
        reward_model_inference(dataset, tokenizer, model, args.batch_size, args.save_path, args.add_special_tokens)
    elif args.task_type == 'llm':
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
        with open(args.generation_config_path, 'r') as f:
            generation_config = GenerationConfig.from_dict(yaml.safe_load(f))
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto', torch_dtype=TYPE_MAP[args.dtype])
        model.eval()
        llm_inference(dataset, tokenizer, model, args.batch_size, generation_config, args.save_path, args.add_special_tokens)
    

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)