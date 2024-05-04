import json
from typing import List, Dict, Callable, Tuple, Type

import torch
import torch.distributed as dist

from src.algorithms.base import BaseTrainer
from src.collator import sft_data_collator, alol_data_collator
from src.algorithms import SFTTrainer, ALOLTrainer


def is_main_process():
    if dist.is_initialized():
        if dist.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True


def print_rank_0(message, end='\n', color='green') -> None:
    PREFIX_MAP = {
        "default": "\033[38m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "pink": "\033[35m",
        "cyan": "\033[36m"
    }
    prefix = PREFIX_MAP[color]
    postfix="\033[0m"
    if is_main_process():
        print(prefix + repr(message) + postfix, flush=True, end=end)


def print_object_on_main_process(name: str, obj: object, split_line_color="yellow", object_color="pink") -> None:
    print_rank_0(">"*30 + name, color=split_line_color)
    print_rank_0(obj, color=object_color)
    print_rank_0(">"*30, color=split_line_color)


def read_json_or_jsonl_data(data_path: str) -> List:
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

    
def get_collator_and_trainer(algorithm) -> Tuple[Callable[[Dict[str, any]], Dict[str, torch.Tensor]], Type[BaseTrainer]]:
    MAP: Dict[str, Tuple[Callable[[Dict[str, any]], Dict[str, torch.Tensor]], Type[BaseTrainer]]] = {
        "sft": (sft_data_collator, SFTTrainer),
        "alol": (alol_data_collator, ALOLTrainer)
    }

    return MAP[algorithm]