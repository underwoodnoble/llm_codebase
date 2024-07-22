import os
import subprocess
import json

from typing import List
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from transformers import TrainingArguments, HfArgumentParser


def print_rank0(string: str):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(string)
    else:
        print(string)
    
@dataclass
class InferenceArguments(TrainingArguments):
    model_type: str = field(default='other')
    model_name_or_path: str = field(default=None)
    input_files: List[str] = field(default=None)
    batch_size_per_process: int = field(default=None)

    task_type: str = field(default='rm')
    model_max_length: int = field(default=512)
    prompt_field_name: str = field(default='prompt')
    answers_field_name: str = field(default='answers')
    output_field_name: str = field(default='output')
    generation_config_path: str = field(default=None)
    dtype: str = field(default='fp16')
    add_special_tokens: bool = field(default=False)
    remove_intermediate_result: bool = field(default=False)
    output_files: List[str] = field(default_factory=lambda :[])
    output_file_name: str = field(default='output.json')


    def post_initalize(self):
        if len(self.input_files) > 1 or dist.get_world_size() > 1:
            dataset = self.merge_file()
            self.input_files = self.split(dataset)
    

    def merge_file(self):
        all_dataset = []
        for file_path in self.input_files:
            with open(file_path, 'r') as f:
                if file_path.endswith('json'):
                    dataset = json.load(f)
                else:
                    dataset = [json.loads(line) for line in f.readlines()]
                all_dataset.extend(dataset)
        return all_dataset

    def split(self, dataset):
        if  dist.get_rank() == 0 and (not os.path.exists(self.output_dir)):
            os.makedirs(self.output_dir)
        chunck_size = (len(dataset) + 1) // dist.get_world_size()

        new_input_files = []
        for i, offset in enumerate(range(0, len(dataset), chunck_size)):
            file_path = Path(self.output_dir) / f'input_rank{i}.json'
            if dist.get_rank() == 0: # Only wirte on main processes
                with open(file_path, 'w') as f:
                    json.dump(dataset[offset:offset+chunck_size], f, ensure_ascii=False, indent=2)
            new_input_files.append(str(file_path))
        return new_input_files        

    def generate_cuda_devices_list(self):
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            cuda_visible_devices = [int(device) for device in os.getenv('CUDA_VISIBLE_DEVICES').split(',')]
        else:
            cuda_visible_devices = list(range(torch.cuda.device_count()))

        gpus_per_process = len(cuda_visible_devices) // int(os.getenv('LOCAL_WORLD_SIZE'))
        return [
            ','.join(
                [str(j) for j in cuda_visible_devices[i*gpus_per_process:(i+1)*gpus_per_process]]
                ) for i in range(int(os.getenv('LOCAL_WORLD_SIZE')))]

    def save_final_results(self):
        if dist.get_rank() == 0:
            dataset = self.merge_file()
            with open(self.output_dir + f'/{self.output_file_name}', 'w') as f:
                if self.output_file_name.endswith('json'):
                    json.dump(dataset, f)
                else:
                    for data in dataset:
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            if self.remove_intermediate_result:
                for file_path in self.input_files:
                    if 'input_rank' in str(file_path):
                        os.remove(file_path)
                for file_path in self.output_files:
                    os.remove(file_path)


def main(args: InferenceArguments):
    args.post_initalize()
    cuda_devices_list = args.generate_cuda_devices_list()
    print(">>>cuda_devices_list: ", cuda_devices_list)

    processes = []
    for i, cuda_devices in enumerate(cuda_devices_list):
        rank = int(os.getenv('RANK'))
        local_rank = int(os.getenv('LOCAL_RANK'))
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = cuda_devices

        save_path = args.output_dir + f'/rank{rank}.jsonl'
        args.output_files.append(save_path)

        if local_rank == i:
            cmd_args = [
                f'--task_type {args.task_type}',
                f'--model_type {args.model_type}',
                f'--model_name_or_path {args.model_name_or_path}',
                f'--model_max_length {args.model_max_length}',
                f'--input_file_path {args.input_files[rank]}',
                f'--prompt_field_name {args.prompt_field_name}',
                f'--answers_field_name {args.answers_field_name}',
                f'--output_field_name {args.output_field_name}',
                f'--batch_size {args.batch_size_per_process}',
                f'--generation_config {args.generation_config_path}',
                f'--save_path {save_path}',
                f'--dtype {args.dtype}',
                f'--add_special_tokens {args.add_special_tokens}'
            ]
            cmd_list = ['python', str((Path(__file__).parent / 'accelerate_inference.py'))] + cmd_args
            cmd = ' '.join(cmd_list)
            print(cmd)
            process = subprocess.Popen(cmd, env=env, shell=True)
            processes.append(process)
    
    for process in processes:
        process.wait()
    torch.cuda.synchronize()
    args.save_final_results()


if __name__ == '__main__':
    parser = HfArgumentParser((InferenceArguments,))
    args = parser.parse_args_into_dataclasses()[0]
    main(args)