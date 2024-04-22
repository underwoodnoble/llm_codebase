from src.arguments import InferenceArguments
from deepspeed import init_inference
from src.utils import print_object_on_main_process, is_main_process
from src.inference_utils import loadDataset, loadModelAndTokenizer, default_collator, get_inference_dict
from transformers import HfArgumentParser, PreTrainedModel, PreTrainedTokenizer, GenerationConfig
from torch.utils.data import DistributedSampler, DataLoader
import torch
import os
import json
from tqdm import tqdm


def reward_inference(data_loader: DataLoader, tokenizer: PreTrainedTokenizer, model: PreTrainedModel,
                    save_name: str, args: InferenceArguments):
    results = []
    progress_bar = tqdm(range(len(data_loader)), disable=not is_main_process(), desc=save_name)
    for batch in tqdm(data_loader):
        progress_bar.update(1)
        texts = []
        for prompt, answers in zip(batch['prompt'], batch["answers"]):
            texts.extend([prompt + answer for answer in answers])
        inputs = tokenizer(texts, padding=True, truncation=True)
        input_ids = torch.tensor(inputs['input_ids']).to(model.device)
        attention_mask = torch.tensor(inputs['attention_mask']).to(model.device)
        with torch.no_grad():
            rewards: torch.FloatTensor = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)['rm_logits']
        rewards = rewards.flatten().view(data_loader.batch_size, -1).tolist()
        
        results.extend([
            {
                "texts": [prompt + answer for answer in answers],
                "scores": scores
            } for prompt, answers, scores in zip(batch['prompt'], batch['answers'], rewards)
            ])
        
        if len(results) >= args.save_steps:
            with open(os.path.join(args.save_dir, save_name + '.jsonl'), 'a') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            results = []
    
    if len(results) > 0:
        with open(os.path.join(args.save_dir, save_name + '.jsonl'), 'a') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')


def llm_inference(data_loader: DataLoader, tokenizer: PreTrainedTokenizer, model: PreTrainedModel,
                save_name: str, args: InferenceArguments):
    results = []
    num_return_sequences=4
    generation_config = GenerationConfig(
        temperature=1.2,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        max_new_tokens=512,
        top_k=40,
        top_p=0.85,
        repetition_penalty=1.05
    )

    for batch in data_loader:
        batch_prompts = batch['prompt']
        encoding = tokenizer(batch_prompts, truncation=True, padding=True)
        encoding = {k: torch.tensor(v).to(model.device) for k, v in encoding.items()}
        print_object_on_main_process("encoding", encoding)
        with torch.no_grad():
            generated_ids = model.generate(**encoding, generation_config=generation_config)
            generated_ids = generated_ids[:, encoding['input_ids'].shape[1]:]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        for j in range(len(batch_prompts)):
            results.append({
                "prompt": batch_prompts[j],
                "answers": generated_texts[j*num_return_sequences:(j+1)*num_return_sequences]
            })
        
        if len(results) >= args.save_steps:
            with open(os.path.join(args.output_dir, save_name + '.jsonl'), 'a') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            results = []
    
    with open(os.path.join(args.output_dir, save_name + '.jsonl'), 'a') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

def main(args: InferenceArguments):
    datasets = loadDataset(args)
    samplers = [DistributedSampler(dataset, shuffle=False) for dataset in datasets]
    data_loaders = [
        DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=default_collator
        ) for dataset, sampler in zip(datasets, samplers)
    ]
    tokenizer, model = loadModelAndTokenizer(args)
    model.eval()
    print_object_on_main_process("tokenizer", tokenizer)

    inference_dict = get_inference_dict(args)
    ds_engine = init_inference(
        model=model,
        **inference_dict
    )
    model = ds_engine.module
    print_object_on_main_process("model", model)

    for data_loader, save_name in zip(data_loaders, args.save_names):
        if args.task_type == 'reward_model_inference':
            reward_inference(data_loader, tokenizer, model, save_name, args)
        elif args.task_type == 'llm_inference':
            llm_inference(data_loader, tokenizer, model, save_name, args)
        else:
            raise ValueError(f"Do not support task type '{args.task_type}'")


if __name__ == '__main__':
    parser = HfArgumentParser(InferenceArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)