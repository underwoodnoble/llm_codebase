import json
import yaml
from tqdm import tqdm
from argparse import ArgumentParser
from typing import List
from pathlib import Path

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
        '--prompt_field_name', type=str, default='prompt'
    )
    parser.add_argument(
        '--answers_field_name', type=str, default='answers', help='Only for reward model inference'
    )
    parser.add_argument(
        '--output_field_name', type=str, default='output'
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
def reward_model_inference(dataset: List, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, args):
    all_texts = []
    for data in dataset:
        all_texts.append([
            data[args.prompt_field_name] + answer for answer in data[args.answers_field_name]
        ])

    rewards = []
    for i in range(0, len(all_texts), args.batch_size):
        if args.add_special_tokens:
            bos_token = tokenizer.bos_token if tokenizer.bos_token else ""
            eos_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.pad_token

        texts = []
        for j in range(args.batch_size):
            texts.extend(bos_token + text + eos_token for text in all_texts[i + j])

        encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, add_special_tokens=False)
        rewards = model(input_ids=encoding['input_ids'].to(model.device)).rm_logits.flatten().tolist()

        num_of_text_per_example = len(rewards) // args.batch_size
        for j in range(args.batch_size):
            data = dataset[i + j]
            data['rewards'] = rewards[j*num_of_text_per_example:(j+1)*num_of_text_per_example]
            with open(args.save_path, 'a') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')


@torch.no_grad()
def llm_inference(dataset: List, tokenizer: PreTrainedTokenizer, model: PreTrainedModel,generation_config: GenerationConfig, args):
    bos_token = tokenizer.bos_token if args.add_special_tokens and tokenizer.bos_token is not None else ""
    prompts = [bos_token + data[args.prompt_field_name] for data in dataset]
    

    for i in range(0, len(prompts), args.batch_size):
        encoding = tokenizer(prompts[i:i+args.batch_size], add_special_tokens=False, padding=True, truncation=True, return_tensors='pt')
        ret = model.generate(encoding['input_ids'].to(model.device), attention_mask=encoding['attention_mask'].to(model.device), generation_config=generation_config)
        responses = tokenizer.batch_decode(ret, skip_special_tokens=True)
        print(responses)
        with open(args.save_path, 'a') as f:
            for j, prompt in enumerate(prompts[i:i+args.batch_size]):
                prompt_responses = responses[j*generation_config.num_return_sequences:(j+1)*generation_config.num_return_sequences]
                f.write(json.dumps({
                    'prompt': prompt,
                    "responses": prompt_responses
                }, ensure_ascii=False) + '\n')


def main(args):
    dataset = load_json_or_jsonl(args.input_file_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.model_max_length = args.model_max_length

    # resume inference
    if Path(args.save_path).exists():
        with open(args.save_path, 'r') as f:
            length = len(f.readlines())
            dataset = dataset[length:]

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
        reward_model_inference(dataset, tokenizer, model, args)
    elif args.task_type == 'llm':
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
        with open(args.generation_config_path, 'r') as f:
            generation_config = GenerationConfig.from_dict(yaml.safe_load(f))
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto', torch_dtype=TYPE_MAP[args.dtype])
        model.eval()
        llm_inference(dataset, tokenizer, model, generation_config, args)
    

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)