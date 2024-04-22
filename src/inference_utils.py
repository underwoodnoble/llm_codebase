from .arguments import InferenceArguments
from typing import List, Dict, Any, Union
from .utils import read_json_or_jsonl_data
from datasets import Dataset
from src.models.RewardModel import QwenRewardModel
from transformers import (PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig,
LlamaConfig, LlamaForCausalLM, LlamaTokenizer, Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer)
import deepspeed
import torch
import os
import json


def default_collator(batch: List[Dict[str, Any]]):
    keys = batch[0].keys()
    ret = {
        key: [] for key in keys
    }
    for data in batch:
        for key in keys:
            ret[key].append(data[key])

    return ret


def loadDataset(args: InferenceArguments) -> List[Dataset]:
    '''load multiple dataset into a list.
    '''
    dataset = []
    for data_path in args.data_paths:
        data_list = read_json_or_jsonl_data(data_path)
        dataset.append(Dataset.from_list(data_list))

    return dataset


def loadModelAndTokenizer(args: InferenceArguments):
    MODEL_CLASS_MAP: Dict[str, Dict[str, Dict[str, Union[PreTrainedTokenizer, PreTrainedModel]]]] = {
        "RM": {
            "qwen": {
                "tokenizer": Qwen2Tokenizer,
                "model": QwenRewardModel,
                "config": Qwen2Config
            }
        },
        "CasualLM": {
            "llama": {
                "tokenizer": LlamaTokenizer,
                "model": LlamaForCausalLM,
                "config": LlamaConfig
            },
            "auto": {
                "tokenizer": AutoTokenizer,
                "model": AutoModelForCausalLM,
                "config": AutoConfig
            }
        }
    }
    if args.task_type == 'reward_model_inference':
        FIRST_MAP = "RM"
    elif args.task_type == 'llm_inference':
        FIRST_MAP = "CasualLM"
    
    tokenizer = MODEL_CLASS_MAP[FIRST_MAP][args.model_type]["tokenizer"].from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        truncation_side='left',
        padding_side='left')
    if args.checkpoint_type == 'deepspeed':
        config = MODEL_CLASS_MAP[FIRST_MAP][args.model_type]["config"].from_pretrained(args.model_name_or_path)
        with deepspeed.OnDevice(dtype=torch.half, device='meta'):
            model = MODEL_CLASS_MAP[FIRST_MAP][args.model_type]["model"](config=config)
    elif args.checkpoint_type == 'huggingface':
        model = MODEL_CLASS_MAP[FIRST_MAP][args.model_type]['model'].from_pretrained(args.model_name_or_path, torch_dtype=torch.half)
    else:
        raise ValueError(f"Do not support checkpoint type {args.checkpoint_type}")

    return tokenizer, model


def get_inference_dict(args: InferenceArguments):
    ret = {
        "tensor_parallel": {"tp_size": args.tp_size},
        "dtype": torch.half,
        "replace_with_kernel_inject": False
    }
    if args.checkpoint_type == 'deepspeed':
        checkpoint_dict = {
            "type": "ds_model",
            "checkpoints": [path for path in os.listdir(args.model_name_or_path) if path.startswith("pytorch_model-")],
            "version": 1.0
        }
        ret.update({'base_dir': args.model_name_or_path, "checkpoint": checkpoint_dict})
    return ret

    
def load_ds_config(args: InferenceArguments):
    with open(args.deepspeed, 'r') as f:
        ds_config = json.load(f)
    return ds_config