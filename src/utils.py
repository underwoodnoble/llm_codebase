import torch
import json
from tqdm import tqdm
from transformers import (LlamaTokenizer, LlamaPreTrainedModel, BertForSequenceClassification, BertConfig, 
BertTokenizer, AutoConfig, LlamaForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel)
from model.RewardModel import LlamaRewardModel
from arguments import CustomArguments
from datasets import Dataset
import os
from typing import List, Dict, Any, Optional, Tuple


def print_rank_0(message) -> None:
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


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

    
def load_data_from_paths(data_paths: List[str]) -> List[Dict[str, Any]]:
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

    
def set_llama_special_token(tokenizer: LlamaTokenizer, model: LlamaPreTrainedModel) -> None:
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
        input_embeddings: torch.Tensor = model.get_input_embeddings().weight.data
        output_embeddings: torch.Tensor = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def dpo_transform(data_list: List[Dict[str, List]], args: CustomArguments) -> List[Dict[str, Any]]:
    new_data_list = []
    for data in data_list:
        if args.construct_method == 'best_over_rest':
            best_id = torch.tensor(data['scores']).argmax().item()
            best_text = data['texts'][best_id]
            prompt, chosen = best_text.split(args.sep_token)
            new_data_list.extend(
                [
                    {"prompt": prompt, "chosen": chosen, 'rejected': rejected} 
                    for rejected in [text for i, text in enumerate(data['texts']) if i != best_id]
                ]
            )
        elif args.construct_method == 'one_over_rest':
            for i in range(len(data['texts']) - 1):
                for j in range(i, len(data['texts'])):
                    prompt, ans1 = data['texts'][i].split(args.sep_token)
                    _, ans2 = data['texts'][j].split(args.sep_token)
                    if data['scores'][i] > data['scores'][j]:
                        new_data_list.append(
                            {"prompt": prompt, "chosen": ans1, "rejected": ans2}
                        )
                    else:
                        new_data_list.append(
                            {"prompt": prompt, "chosen": ans2, "rejected": ans1}
                        )
        elif args.construct_method == 'best_over_worst':
            best_id = torch.tensor(data['scores']).argmax().item()
            worst_id = torch.tensor(data['scores']).argmin().item()
            best_text: str = data['texts'][best_id]
            worst_text: str = data['texts'][worst_id]
            prompt, chosen = best_text.split(args.sep_token)
            _, rejected = worst_text.split(args.sep_token)
            new_data_list.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        else:
            raise ValueError(f"Do not support construct method {args.construct_method}")

    return new_data_list

def getDataset(args: CustomArguments, type='train') -> Dataset:
    if type == 'train':
        if args.data_paths is None and args.data_dir is None:
            return None
        if args.data_paths is not None:
            data_paths = args.data_paths
        else:
            data_paths = [os.path.join(args.data_dir, path) for path in os.listdir(args.data_dir)]
    else:
        if args.eval_data_paths is None and args.eval_data_dir is None:
            return None
        if args.eval_data_paths is not None:
            data_paths = args.eval_data_paths
        else:
            data_paths = [os.path.join(args.eval_data_dir, path) for path in os.listdir(args.eval_data_dir)]       
    data_list = load_data_from_paths(data_paths)

    if args.task_type in ['reward', "offline_rejection_sampling", "offline_RRHF", "DPO"]:
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
        if args.task_type == 'DPO':
            data_list = dpo_transform(data_list, args)

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
    elif args.task_type == 'weighted_learning':
        if args.weighted_data_prompt_name != 'prompt' or args.weighted_data_answer_name != 'answer' or args.weighted_data_score_name != 'score':
            new_data_list = []
            for data in data_list:
                new_data = {
                    "prompt": data[args.weighted_data_prompt_name],
                    "answer": data[args.weighted_data_answer_name],
                    "score": data[args.weighted_data_score_name]
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
    
    if args.debug_mode:
        data_list = data_list[:1000]
    return Dataset.from_list(data_list)


def loadTokenizerAndModel(args: CustomArguments) -> Tuple[PreTrainedTokenizer, PreTrainedModel, Optional[PreTrainedModel]]:
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
    elif args.task_type in ['sft', 'offline_rejection_sampling', "offline_RRHF", 'weighted_learning']:
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
    elif args.task_type == 'DPO':
        if args.model_type == 'llama':
            tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, truncation_side=args.truncation_side, padding_side=args.padding_side)
            tokenizer.model_max_length = args.model_max_length
            model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
            ref_model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
            set_llama_special_token(tokenizer, model)       
        return tokenizer, model, ref_model
    
    return tokenizer, model, None


def getTestDataset(args) -> List[Dict[str, Any]]:
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


def loadTestTokenizerAndModel(args) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    if args.model_type == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, truncation_side='left', padding_side='right')
        tokenizer.model_max_length = args.model_max_length
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
        set_llama_special_token(tokenizer, model)
    return tokenizer, model