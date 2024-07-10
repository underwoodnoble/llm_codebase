from pathlib import Path
from typing import List
import json

import datasets

from .general_utils import print_rank_0
from ..arguments import GenericDataArguments
from src.algorithms import sft_transform, offline_ppo_transform, rm_transform, dpo_transform


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
                if data_path.suffix == '.json':
                    all_data_paths.append(str(data_path))
                else:
                    raise ValueError("Only support json format dataset.")
        return all_data_paths

    def get_datasets(data_files: List[str]) -> List[datasets.Dataset]:
        return [
            datasets.load_dataset('json', data_files=data_file, streaming=data_args.streaming, split='train')
            for data_file in data_files
        ]

    # transform method
    TRANSFORM_MAP = {
        "sft": sft_transform(data_args),
        "rm": rm_transform(data_args),
        "offline_ppo": offline_ppo_transform(data_args),
        "dpo": dpo_transform(data_args)
    }

    if data_args.data_paths is not None:
        data_files = get_data_files(data_args.data_paths)
        train_dataset = get_datasets(data_files)

        # todo: In datasets 2.20.0, map will keep the original columns, we need to remove them.
        # Attention: make sure that all of the dataset have the same type on the corresponding field.
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
    else:
        eval_dataset = None

    return train_dataset, eval_dataset
