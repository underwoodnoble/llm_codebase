import json
from typing import List, Dict
from .general_utils import print_rank_0, print_object_on_main_process
from ..arguments import GenericDataArguments
import os
import datasets


def load_dataset(data_args: GenericDataArguments, algorithm):
    def gather_data_path(data_paths: str, data_dirs: List[str]) -> List[str]:
        all_data_paths = []
        if data_paths is not None:
            all_data_paths.extend(data_paths)
        if data_dirs is not None:
            for data_dir in data_dirs:
                all_data_paths.extend([path for path in os.listdir(data_dir)
                                    if path.endswith('.json') or path.endswith('.jsonl')])
        return all_data_paths


    have_train = data_args.data_paths or data_args.data_dirs
    have_eval = data_args.eval_data_paths or data_args.eval_data_dirs
    data_path_dict = {}
    if have_train:
        data_path_dict['train'] = gather_data_path(data_args.data_paths, data_args.data_dirs)
    if have_eval:
        data_path_dict['eval'] = gather_data_path(data_args.eval_data_paths, data_args.eval_data_dirs)

    print_object_on_main_process("data path dict", data_path_dict)
    # Create Dataset
    dataset = datasets.load_dataset('json', data_files=data_path_dict, streaming=data_args.streaming)

    training_set = dataset['train'] if have_train else None
    eval_set = dataset['eval'] if have_eval else None

    
    # transform data
    TRANSFORM_MAP = {
        "sft": None,
        "rm": None
    }

    return training_set, eval_set
