from typing import List
import random

import numpy as np
import torch
import torch.distributed as dist


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

    
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed_all(seed)
