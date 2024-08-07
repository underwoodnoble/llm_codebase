import argparse
import sys
from transformers import HfArgumentParser
from ..arguments import (SFTTrainingArguments, SFTDataArguments, RMDataArguments, RMTrainingArguments,
    OfflinePPODataArguments, OfflinePPOTrainingArguments, DPODataArguments, DPOTrainingArguments)
from .general_utils import print_object_on_main_process


def get_args():
    # create top parser
    parser = argparse.ArgumentParser(description="LLM codebase")

    # each algorithm has a subparser
    subparsers = parser.add_subparsers(dest='algorithm', help="Select training algorithm")
    subparsers.add_parser('sft', help="Using SFT parser")
    subparsers.add_parser('rm', help="Using RM parser")
    subparsers.add_parser('offline_ppo', help="Using offline ppo parser")
    subparsers.add_parser('dpo', help="Using DPO parser")

    supported_algorithms = list(subparsers.choices.keys())
    # Some distributed training frameworks add additional argumes.
    for argument in sys.argv[1:]:
        if argument in supported_algorithms:
            algorithm_args = parser.parse_args([argument])
            sys.argv.remove(argument)
            break

    if not hasattr(algorithm_args, 'algorithm'):
        parser.print_help()
        sys.exit(1)
    
    # parser arguments according to the selected algorithm
    if algorithm_args.algorithm == 'sft':
        subparser = HfArgumentParser((SFTTrainingArguments, SFTDataArguments))
    elif algorithm_args.algorithm == 'rm':
        subparser = HfArgumentParser((RMTrainingArguments, RMDataArguments))
    elif algorithm_args.algorithm == 'offline_ppo':
        subparser = HfArgumentParser((OfflinePPOTrainingArguments, OfflinePPODataArguments))
    elif algorithm_args.algorithm == 'dpo':
        subparser = HfArgumentParser((DPOTrainingArguments, DPODataArguments))

    return algorithm_args.algorithm, subparser.parse_args_into_dataclasses(sys.argv[1:])