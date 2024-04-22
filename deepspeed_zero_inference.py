from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, TrainingArguments, PreTrainedModel, PreTrainedTokenizer, GenerationConfig
from transformers.deepspeed import HfDeepSpeedConfig
from accelerate import PartialState
from accelerate.utils import gather_object
from tqdm import tqdm
import torch
from typing import List, Optional
import json
from dataclasses import dataclass, field
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import deepspeed
import gc


@dataclass
class CustomArguments(TrainingArguments):
    model_name_or_path: Optional[str] = field(default=None)
    data_path: Optional[str] = field(default=None)
    save_name: Optional[str] = field(default=None)

def load_data(data_path):
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    return dataset

def load_ds_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def llm_inference(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, prompts):
    new_dataset = []
    num_return_sequences = 4
    generation_config = GenerationConfig(
        temperature=1.2,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        max_new_tokens=64,
        top_k=40,
        top_p=0.85,
        repetition_penalty=1.05
    )
    encoding = tokenizer(prompts, truncation=True, padding=True)
    encoding = {k: torch.tensor(v).to(model.device) for k, v in encoding.items()}
    print(encoding)
    with torch.no_grad():
        generated_ids = model.generate(**encoding, generation_config=generation_config)
        generated_ids = generated_ids[:, encoding['input_ids'].shape[1]:]
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        print(generated_texts)
    for j in range(len(prompts)):
        new_dataset.append({
            "prompt": prompts[j],
            "answers": generated_texts[j:j+num_return_sequences]
        })
    return new_dataset


def collator(items):
    return [
        data['prompt'] for data in items
    ]

def main(args):
    distributed_state = PartialState()
    dataset = Dataset.from_list(load_data(args.data_path))
    sampler = DistributedSampler(dataset, shuffle=False)
    data_loader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=collator,
        batch_size=2,
        sampler=sampler
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, truncation_side='left', padding_side='left', trust_remote_code=True)

    ds_config = load_ds_config(args.deepspeed)
    dschf = HfDeepSpeedConfig(ds_config)
    torch.cuda.empty_cache()
    gc.collect()

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.half, trust_remote_code=True)
    ret = deepspeed.initialize(
        model=model,
        config=ds_config
    )
    ds_engine = ret[0]
    model = ds_engine.module
    model.eval()

    if distributed_state.is_main_process:
        print(model)

    for prompts in data_loader:
        llm_inference(tokenizer, model, prompts)

if __name__ == '__main__':
    parser = HfArgumentParser(CustomArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)