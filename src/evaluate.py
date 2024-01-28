from argparse import ArgumentParser
from typing import List
from utils import getTestDataset, loadTestTokenizerAndModel
from accelerate import PartialState
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from datasets import Dataset
from test import compute_ppl
import json
from tqdm import tqdm


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--model_type', type=str, choices=['llama'])
    parser.add_argument('--task_type', type=str, choices=['ppl'])
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--cache_size', type=int, default=8)
    parser.add_argument('--data_prompt_name', type=str, default='prompt')
    parser.add_argument('--data_answer_name', type=str, default='answer')
    parser.add_argument('--model_max_length', type=int, default=512)
    parser.add_argument('--debug_mode', type=bool, default=False)
    args = parser.parse_args()
    return args


def main(args):
    distributed_state = PartialState()
    dataset = getTestDataset(args)
    tokenizer, model = loadTestTokenizerAndModel(args)
    model.to(distributed_state.device)

    for i in range(0, len(dataset), args.cache_size):
        total_ret = []
        with distributed_state.split_between_processes(dataset[i:i+args.cache_size]) as sub_dataset:
            n = distributed_state.process_index
            with tqdm(total=len(sub_dataset), desc=f'rank: {n+1}') as pbar:
                sub_dataset = Dataset.from_list(sub_dataset)
                data_loader = DataLoader(sub_dataset, batch_size=args.batch_size)
                for x in data_loader:
                    ret = compute_ppl(x, model=model, tokenizer=tokenizer)
                    total_ret.extend(ret)
                    pbar.update(1)

        total_ret = gather_object(total_ret)
        if distributed_state.is_main_process:
            with open(args.save_path, 'a+') as f:
                for ret in total_ret:
                    f.write(json.dumps(ret) + '\n')


if __name__ == '__main__':
    args = get_args()
    main(args)
