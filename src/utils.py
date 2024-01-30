import torch
import json
from tqdm import tqdm
from transformers import (LlamaTokenizer, LlamaPreTrainedModel, BertForSequenceClassification, BertConfig, 
BertTokenizer, AutoConfig, LlamaForCausalLM, AutoModelForSequenceClassification, AutoTokenizer)
from model.RewardModel import LlamaRewardModel
from arguments import CustomArguments
from datasets import Dataset
import os
from typing import List


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def read_json_or_jsonl_data(data_path: str):
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

    
def load_data_from_paths(data_paths: List[str]):
    total_data_list = []
    i = 0
    for data_path in data_paths:
        data_list = read_json_or_jsonl_data(data_path)
        for data in tqdm(data_list):
            data['id'] = i
            i += 1
            total_data_list.append(data)
    print_rank_0(f">>> total load {len(total_data_list)} data.")
    return total_data_list

    
def set_llama_special_token(tokenizer: LlamaTokenizer, model: LlamaPreTrainedModel):
    DEFAULT_PAD_TOKEN = "<pad>"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"
    
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def getDataset(args: CustomArguments, type='train'):
    if type == 'train':
        if args.data_paths is not None:
            data_paths = args.data_paths
        else:
            data_paths = [os.path.join(args.data_dir, path) for path in os.listdir(args.data_dir)]
    else:
        if args.eval_data_paths is not None:
            data_paths = args.eval_data_paths
        else:
            data_paths = [os.path.join(args.eval_data_dir, path) for path in os.listdir(args.eval_data_dir)]       
    data_list = load_data_from_paths(data_paths)

    if args.task_type in ['reward', "offline_rejection_sampling", "offline_RRHF"]:
        # transform to the format: {"texts": ["text1", "text2"], "scores": [s1, s2]}
        if args.preference_data_text_name != 'texts' or args.preference_data_score_name != 'scores':
            new_data_list = []
            for data in data_list:
                new_data = {
                    "texts": data[args.preference_data_text_name],
                    "scores": data[args.preference_data_score_name]
                }
                new_data_list.append(new_data)
            data_list = new_data_list
    elif args.task_type == 'sft':
        if args.sft_data_prompt_name != 'prompt' or args.sft_data_answer_name != 'answer':
            new_data_list = []
            for data in data_list:
                new_data = {
                    "prompt": data[args.sft_data_prompt_name],
                    "answer": data[args.sft_data_answer_name]
                }
                new_data_list.append(new_data)
            data_list = new_data_list
    elif args.task_type == 'contrastive_learning':
        if args.contrastive_data_prompt_name != 'prompt' or args.contrastive_data_answer_name != 'answer' or args.contrastive_data_score_name != 'score':
            new_data_list = []
            for data in data_list:
                new_data = {
                    "prompt": data[args.contrastive_data_prompt_name],
                    "answer": data[args.contrastive_data_answer_name],
                    "score": data[args.contrastive_data_score_name]
                }
    elif args.task_type == 'classification':
        new_data_list = []
        labels = []
        args.id2label = {}
        args.label2id = {}
        for data in data_list:
            new_data = {
                "text": data[args.cls_data_text_name],
                "label": data[args.cls_data_label_name]
            }
            new_data_list.append(new_data)
            labels.append(new_data['label'])
        
        labels = list(set(labels))
        for i, label in enumerate(labels):
            args.id2label[i] = label
            args.label2id[label] = i

        if args.cls_data_label_nums is None:
            args.cls_data_label_nums = len(labels)
        data_list = new_data_list
    
    return Dataset.from_list(data_list)


def loadTokenizerAndModel(args: CustomArguments):
    if args.task_type == 'reward':
        if args.model_type == 'bert':
            config = BertConfig.from_pretrained(args.model_name_or_path)
            config.num_labels = 1
            tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, truncation_side=args.truncation_side)
            tokenizer.model_max_length = args.model_max_length
            model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
        elif args.model_type == 'llama':
            tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, truncation_side=args.truncation_side, padding_side=args.padding_side)
            tokenizer.model_max_length = args.model_max_length
            model = LlamaRewardModel.from_pretrained(args.model_name_or_path)
            set_llama_special_token(tokenizer, model)
        else:
            raise ValueError(f"Training reward model do not support the model type {args.model_type}.")
    elif args.task_type in ['sft', 'offline_rejection_sampling', "offline_RRHF", 'contrastive_learning']:
        if args.model_type == 'llama':
            tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, truncation_side=args.truncation_side, padding_side=args.padding_side)
            tokenizer.model_max_length = args.model_max_length
            model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
            set_llama_special_token(tokenizer, model)
    elif args.task_type == 'classification':
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        config.num_labels = args.cls_data_label_nums
        config.id2label = args.id2label
        config.label2id = args.label2id
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, truncation_side=args.truncation_side, padding_side=args.padding_side, trust_remote_code=True)
        tokenizer.model_max_length = args.model_max_length
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
        if args.model_type == 'llama':
            set_llama_special_token(tokenizer, model)
    
    return tokenizer, model


def getTestDataset(args):
    if args.data_path is not None:
        data_paths = [args.data_path]
    else:
        data_paths = [os.path.join(args.data_dir, path) for path in os.listdir(args.data_dir)]        
    data_list = load_data_from_paths(data_paths)
    if args.task_type == 'ppl':
        if args.data_prompt_name != 'prompt' or args.data_answer_name != 'answer':
            new_data_list = []
            for data in data_list:
                new_data = {
                    "prompt": data[args.sft_data_prompt_name],
                    "answer": data[args.sft_data_answer_name]
                }
                new_data_list.append(new_data)
            data_list = new_data_list

    if args.debug_mode:
        data_list = data_list[:100]
    return data_list


def loadTestTokenizerAndModel(args):
    if args.model_type == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, truncation_side='left', padding_side='right')
        tokenizer.model_max_length = args.model_max_length
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
        set_llama_special_token(tokenizer, model)
    return tokenizer, model