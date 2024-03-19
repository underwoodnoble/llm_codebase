import torch
import json
from tqdm import tqdm
import transformers
from transformers import (LlamaTokenizer, LlamaPreTrainedModel, BertForSequenceClassification, BertConfig, 
BertTokenizer, AutoConfig, LlamaForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer,
PreTrainedModel, AutoModelForCausalLM)
from .models.RewardModel import LlamaRewardModel
from .arguments import TrainingArguments
import os
from typing import List, Dict, Any, Optional, Tuple, Union
from datasets import Dataset


def is_main_process():
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True


def print_rank_0(message, end='\n') -> None:
    if is_main_process():
        print(message, flush=True, end=end)


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
        for data in tqdm(data_list, disable=not is_main_process()):
            data['id'] = i
            i += 1
            total_data_list.append(data)
    print_rank_0(f">>> total load {len(total_data_list)} data.")
    return total_data_list

    
def set_special_tokens(tokenizer: LlamaTokenizer, model: LlamaPreTrainedModel) -> None:
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
    # If not set add_eos_token to True, Llama tokenizer do not add eos token in encoding automatically.
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings: torch.Tensor = model.get_input_embeddings().weight.data
        output_embeddings: torch.Tensor = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def dpo_transform(data_list: List[Dict[str, List]], args: TrainingArguments) -> List[Dict[str, Any]]:
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

def data_transform(data_list: List[Dict[str, List]], args: TrainingArguments) -> List[Dict[str, Any]]:
    new_data_list = []
    if args.task_type in ['reward', "offline_rejection_sampling", "offline_RRHF", "DPO"]:
        # transform to the format: {"texts": ["text1", "text2"], "scores": [s1, s2]}
        if args.preference_data_text_name != 'texts' or args.preference_data_score_name != 'scores':
            for data in data_list:
                new_data = {
                    "texts": data[args.preference_data_text_name],
                    "scores": data[args.preference_data_score_name]
                }
                new_data_list.append(new_data)
        else:
            new_data_list = data_list
        if args.task_type == 'DPO':
            new_data_list = dpo_transform(new_data_list, args)

    elif args.task_type == 'SFT':
        if args.sft_data_prompt_name != 'prompt' or args.sft_data_answer_name != 'answer':
            for data in data_list:
                new_data = {
                    "prompt": data[args.sft_data_prompt_name],
                    "answer": data[args.sft_data_answer_name]
                }
                new_data_list.append(new_data)
        else:
            new_data_list = data_list
    elif args.task_type == 'weighted_learning':
        if args.weighted_data_prompt_name != 'prompt' or args.weighted_data_answer_name != 'answer' or args.weighted_data_score_name != 'score':
            for data in data_list:
                new_data = {
                    "prompt": data[args.weighted_data_prompt_name],
                    "answer": data[args.weighted_data_answer_name],
                    "score": data[args.weighted_data_score_name]
                }
                new_data_list.append(new_data)
        else:
            new_data_list = data_list

    elif args.task_type == 'classification':
        labels = []
        args.id2label = {}
        args.label2id = {}
        if args.cls_data_text_name != 'text' or args.cls_data_label_name != 'label':
            for data in data_list:
                new_data = {
                    "text": data[args.cls_data_text_name],
                    "label": data[args.cls_data_label_name]
                }
                new_data_list.append(new_data)
                labels.append(new_data['label'])
        else:
            for data in data_list:
                labels.append(data['label'])
            new_data_list = data_list

        labels = list(set(labels))
        for i, label in enumerate(labels):
            args.id2label[i] = label
            args.label2id[label] = i

        if args.cls_data_label_nums is None:
            args.cls_data_label_nums = len(labels)
    
    elif args.task_type == 'KTO':
        if args.kto_pair_prompt_name != 'prompt' or args.kto_pair_answer_name != 'completion' or args.kto_pair_label_name != 'label':
            for data in data_list:
                new_data = {
                    "prompt": data[args.kto_pair_prompt_name],
                    "completion": data[args.kto_pair_answer_name],
                    "score": data[args.kto_pair_label_name]
                }
                new_data_list.append(new_data)
        else:
            new_data_list = data_list

    if args.debug_mode:
        new_data_list = new_data_list[:100]

    return new_data_list

def getDataset(args: TrainingArguments, type='train') -> Union[Dataset, Dict[str, Dataset]]:
    if type == 'train':
        if args.data_paths is None and args.data_dir is None:
            return None
        if args.data_paths is not None:
            data_paths = args.data_paths
        else:
            data_paths = [os.path.join(args.data_dir, path) for path in os.listdir(args.data_dir)]
        print_rank_0(data_paths)
        data_list = data_transform(load_data_from_paths(data_paths), args)
        return Dataset.from_list(data_list)
    else:
        eval_dataset = {}
        if args.eval_data_paths is None and args.eval_data_dir is None:
            return None
        if args.eval_data_paths is not None:
            data_paths = args.eval_data_paths
        else:
            data_paths = [os.path.join(args.eval_data_dir, path) for path in os.listdir(args.eval_data_dir)]       
        if args.eval_dataset_merge_mode in ['separate', 'both']:
            if args.eval_dataset_merge_mode == 'both':
                eval_dataset['all'] = []
            for path in data_paths:
                sub_data_list = data_transform(load_data_from_paths([path]), args)
                if args.eval_dataset_merge_mode == 'both':
                    eval_dataset['all'].extend(sub_data_list)
                _, name = os.path.split(path)
                eval_dataset[name] = Dataset.from_list(sub_data_list)
            if args.eval_dataset_merge_mode == 'both':
                eval_dataset['all'] = Dataset.from_list(eval_dataset['all'])

        elif args.eval_dataset_merge_mode == 'merge':
            eval_dataset = data_transform(load_data_from_paths(data_paths), args)
            eval_dataset = Dataset.from_list(eval_dataset)
        return eval_dataset


def loadTokenizerAndModel(args: TrainingArguments) -> Tuple[PreTrainedTokenizer, PreTrainedModel, Optional[PreTrainedModel]]:
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
            if args.set_special_tokens:
                set_special_tokens(tokenizer, model)
            else:
                if tokenizer.pad_token is None:
                    tokenizer.pad_token_id = 0
                    model.config.pad_token_id = 0
                    
        else:
            raise ValueError(f"Training reward model do not support the model type {args.model_type}.")
    elif args.task_type in ['SFT', 'offline_rejection_sampling', "offline_RRHF", 'weighted_learning']:
        if args.model_type == 'llama':
            tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, truncation_side=args.truncation_side, padding_side=args.padding_side)
            tokenizer.model_max_length = args.model_max_length
            model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
            if args.set_special_tokens:
                set_special_tokens(tokenizer, model)
            else:
                if tokenizer.pad_token is None:
                    tokenizer.pad_token_id = 0
                    model.config.pad_token_id = 0

    elif args.task_type == 'classification':
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        config.num_labels = args.cls_data_label_nums
        config.id2label = args.id2label
        config.label2id = args.label2id
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, truncation_side=args.truncation_side, padding_side=args.padding_side, trust_remote_code=True)
        tokenizer.model_max_length = args.model_max_length
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
        if args.model_type == 'llama':
            if args.set_special_tokens:
                set_special_tokens(tokenizer, model)
            else:
                tokenizer.pad_token_id = 0
                model.config.pad_token_id = 0

    elif args.task_type in ['DPO', 'KTO']:
        if args.model_type == 'llama':
            tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, truncation_side=args.truncation_side, padding_side=args.padding_side)
            tokenizer.model_max_length = args.model_max_length
            model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
            ref_tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, truncation_side=args.truncation_side, padding_side=args.padding_side)
            ref_model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
        elif args.model_type == 'other':
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, truncation_side=args.truncation_side, padding_side=args.padding_side, trust_remote_code=True)
            tokenizer.model_max_length = args.model_max_length
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
            ref_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, truncation_side=args.truncation_side, padding_side=args.padding_side, trust_remote_code=True)
            ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)

        if args.set_special_tokens:
            set_special_tokens(tokenizer, model)       
            set_special_tokens(ref_tokenizer, ref_model)
        else:
            tokenizer.pad_token_id = 0
            model.config.pad_token_id = 0
            ref_model.config.pad_token_id = 0

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
    elif args.task_type == 'win_rate':
        if args.pair_data_prompt_name != 'prompt' or args.pair_data_answers_name != 'answers':
            new_data_list = []
            for data in data_list:
                new_data = {
                    "prompt": data[args.pair_data_prompt_name],
                    "answers": data[args.pair_data_answers_name]
                }
                new_data_list.append(new_data)
            data_list = new_data_list
    elif args.task_type == 'ece':
        if args.preference_data_text_name != 'texts' or args.preference_data_score_name != 'scores':
            new_data_list = []
            for data in data_list:
                new_data = {
                    "texts": data[args.preference_data_text_name],
                    "scores": data[args.preference_data_score_name]
                }
                new_data_list.append(new_data)
            data_list = new_data_list
            
    if args.num_of_examples is not None:
        data_list = data_list[:args.num_of_examples]
    if args.debug_mode:
        data_list = data_list[:100]
    return data_list


def loadTestTokenizerAndModel(args) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    if args.task_type == 'ppl':
        if args.model_type == 'llama':
            tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, truncation_side='left', padding_side='right')
            tokenizer.model_max_length = args.model_max_length
            model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
    if args.task_type == 'ece':
        if args.model_type == 'llama':
            tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, truncation_side='left', padding_side='right')
            tokenizer.model_max_length = args.model_max_length
            model = LlamaRewardModel.from_pretrained(args.model_name_or_path)

    return tokenizer, model