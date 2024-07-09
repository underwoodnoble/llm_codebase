import os
import subprocess
import json
import yaml

from argparse import ArgumentParser
from pydantic import BaseModel, Field, ConfigDict
from typing import List
from pathlib import Path


class InferenceConfig(BaseModel):
    model_config  = ConfigDict(protected_namespaces=())

    processes: int
    gpus: int
    model_type: str
    model_name_or_path: str
    input_files: List[str]
    save_dir: str

    task_type: str = Field(default='rm')
    batch_size_per_process: int = Field(default=1)
    model_max_length: int = Field(default=512)
    dtype: str = Field(default='fp16')

    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            config = InferenceConfig(**config)

        if len(config.input_files) > 1 or config.processes > 1:
            dataset = InferenceConfig.merge_file(config.input_files)
            config.input_files = InferenceConfig.split(dataset, config.processes, config.save_dir)
        
        if config.gpus % config.processes != 0:
            raise ValueError('The number of Gpus must be an integer multiple of the number of processes') 

        return config

    def generate_cuda_devices_list(self):
        gpus_per_process = self.gpus // self.processes
        return [
            ','.join([
                str(j) for j in
                range(i*gpus_per_process, (i+1)*gpus_per_process)])
            for i in range(self.processes)
        ]

    @staticmethod
    def merge_file(file_paths):
        all_dataset = []
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                if file_path.endswith('json'):
                    dataset = json.load(f)
                else:
                    dataset = [json.loads(line) for line in f.readlines()]
                all_dataset.extend(dataset)
        return all_dataset
    
    @staticmethod
    def split(dataset, chuncks, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        chunck_size = (len(dataset) + 1) // chuncks
        new_input_files = []
        for i, offset in enumerate(range(0, len(dataset), chunck_size)):
            file_path = Path(save_dir) / f'input_rank_{i}.json'
            with open(file_path, 'w') as f:
                json.dump(dataset[offset:offset+chunck_size], f, ensure_ascii=False, indent=2)
            new_input_files.append(str(file_path))
        return new_input_files
    

args_template = """
    --task_type {task_type} 
    --model_type {model_type} 
    --model_name_or_path {model_name_or_path} 
    --input_file_path {input_file_path} 
    --batch_size {batch_size} 
    --save_path {save_path} 
    --dtype {dtype}
    """

def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        '--config_path', type=str
    )
    return parser


def main(args):
    processes = []
    config = InferenceConfig.load_config(args.config_path)
    print(config)
    cuda_devices_list = config.generate_cuda_devices_list()

    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)

    for i, cuda_devices in enumerate(cuda_devices_list):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = cuda_devices

        save_path = config.save_dir + f'/rank{i}.json'
        cmd_args = [
            f'--task_type {config.task_type}',
            f'--model_type {config.model_type}',
            f'--model_name_or_path {config.model_name_or_path}',
            f'--model_max_length {config.model_max_length}',
            f'--input_file_path {config.input_files[i]}',
            f'--batch_size {config.batch_size_per_process}',
            f'--save_path {save_path}',
            f'--dtype {config.dtype}'
        ]
        cmd_list = ['python', str((Path(__file__).parent / 'accelerate_inference.py'))] + cmd_args
        cmd = ' '.join(cmd_list)
        print(cmd)
        process = subprocess.Popen(cmd, env=env, shell=True)
        processes.append(process)
    
    for process in processes:
        process.wait()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)