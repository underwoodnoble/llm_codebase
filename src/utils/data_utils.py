from pathlib import Path
from typing import List
import json

import datasets

from .general_utils import print_rank_0
from ..arguments import GenericDataArguments
from src.algorithms.sft import sft_transform
from src.algorithms.alol import alol_transform


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


def load_dataset(data_args: GenericDataArguments, algorithm):
    def get_data_files(data_paths: List[Path]) -> List[str]:
        all_data_paths = []
        for data_path in data_paths:
            if data_path.is_dir():
                all_data_paths.extend(get_data_files(list(data_path.iterdir())))
            else:
                if data_path.suffix in ['.json', '.jsonl']:
                    all_data_paths.append(str(data_path))
        return all_data_paths

    def get_datasets(data_files: List[str]) -> List[datasets.Dataset]:
        return [
            datasets.load_dataset('json', data_files=data_file, streaming=data_args.streaming, split='train')
            for data_file in data_files
        ]

    # transform method
    TRANSFORM_MAP = {
        "sft": sft_transform(data_args),
        "rm": None,
        "alol": alol_transform(data_args)
    }

    if data_args.data_paths is not None:
        data_files = get_data_files(data_args.data_paths)
        train_dataset = get_datasets(data_files)
        train_dataset = datasets.concatenate_datasets(
            [
                ds.map(TRANSFORM_MAP[algorithm])
                for ds in train_dataset
            ]
        )
    else:
        train_dataset = None

    if data_args.eval_data_paths is not None:
        data_files = get_data_files(data_args.eval_data_paths)
        eval_dataset = get_datasets(data_files)
        if data_args.eval_dataset_merge_mode == 'merge':
            eval_dataset = datasets.concatenate_datasets(
                [
                    ds.map(TRANSFORM_MAP[algorithm])
                    for ds in eval_dataset
                ]
            )
        else:
            eval_dataset = {
                Path(data_file).stem: ds.map(TRANSFORM_MAP[algorithm]) for data_file, ds in zip(data_files, eval_dataset)
            }

    return train_dataset, eval_dataset
