import torch
from typing import List, Dict, Any, Callable
from copy import deepcopy

from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers.trainer_pt_utils import LabelSmoother

from .arguments.training_arguments import BaseTrainingArguments, RMTrainingArguments, SFTTrainingArguments, ALOLTrainingArguments


IGNORE_INDEX = LabelSmoother.ignore_index


def _llm_tokenize(prompts: List[str], texts: List[str], tokenizer: PreTrainedTokenizer, args: BaseTrainingArguments) -> Dict[str, torch.Tensor]:
    input_ids = []
    labels = []
    for prompt, text in zip(prompts, texts):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        response_start_idx = len(prompt_ids)
        if prompt_ids != text_ids[:response_start_idx]:
            response_start_idx -= 1
        if args.add_special_tokens:
            response_start_idx += 1
            text_ids = [tokenizer.bos_token_id] + text_ids + [tokenizer.eos_token_id]
        label = deepcopy(text_ids)
        if args.only_predict_answer:
            label[:response_start_idx] = [IGNORE_INDEX] * response_start_idx
        if len(text_ids) > args.model_max_length:
            text_ids = text_ids[-tokenizer.model_max_length:]
            label = label[-tokenizer.model_max_length:]

        input_ids.append(torch.tensor(text_ids))
        labels.append(torch.tensor(label))
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    if args.pad_labels_with_ignore:
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)

    if args.debug_mode: # If debug_model is True, then pad the sequence into the max token length.
        input_ids = F.pad(input_ids, (0, tokenizer.model_max_length - input_ids.shape[0]), mode='constant', value=tokenizer.pad_token_id)
        labels = F.pad(labels, (0, tokenizer.model_max_length - input_ids.shape[0]), mode='constant', value=IGNORE_INDEX
                    if args.pad_labels_with_ignore else tokenizer.pad_token_id)

    attention_mask = torch.ne(input_ids, tokenizer.pad_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    

def classfication_data_collator(tokenizer: PreTrainedTokenizer, args) -> Callable[[Dict[str, any]], Dict[str, torch.Tensor]]:
    def collator(examples) -> Dict[str, torch.Tensor]:
        texts = []
        labels = []
        for example in examples:
            texts.append(example['text'])
            labels.append(args.label2id[example['label']])
                
        encodings = tokenizer(texts, padding=True, truncation=True, add_special_tokens=True)

        return {
            "input_ids": torch.tensor(encodings['input_ids']),
            "attention_mask": torch.tensor(encodings['attention_mask']),
            "labels": torch.tensor(labels)
        }
    
    return collator
        

def multi_object_classification_data_collator(tokenizer: PreTrainedTokenizer, args) -> Callable[[Dict[str, any]], Dict[str, torch.Tensor]]:
    def collator(examples) -> Dict[str, torch.Tensor]:
        pass

 
def reward_data_collator(tokenizer: PreTrainedTokenizer, args: RMTrainingArguments) -> Callable[[Dict[str, any]], Dict[str, torch.Tensor]]:
    def collator(examples: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        batch_size = len(examples)
        num_sample = max([len(example['texts']) for example in examples])
        all_texts = []
        all_scores = []
        for example in examples:
            if len(example['texts']) < num_sample:
                example['texts'].extend([' ']*(num_sample - len(example['texts'])))
                min_score = torch.min(examples['scores'])
                example['scores'].extend([min_score-1]*(num_sample - len(example['scores'])))
            all_texts.extend(example['texts'])
            all_scores.extend(example['scores'])

        encodings = tokenizer(all_texts, padding=True, truncation=True, add_special_tokens=True)

        return {
            "input_ids": torch.tensor(encodings['input_ids']).reshape(batch_size, num_sample, -1),
            "attention_mask": torch.tensor(encodings['attention_mask']).reshape(batch_size, num_sample, -1),
            "scores": torch.tensor(all_scores).reshape(batch_size, -1)
        }

    return collator

    
def sft_data_collator(tokenizer: PreTrainedTokenizer, args: SFTTrainingArguments) -> Callable[[Dict[str, any]], Dict[str, torch.Tensor]]:
    def collator(examples) -> Dict[str, torch.Tensor]:
        texts = []
        prompts = []
        weights = []
        for example in examples:
            text = example['prompt'] + example['answer']
            texts.append(text)
            prompts.append(example['prompt'])
            weights.append(example['weight'])

        ret = _llm_tokenize(prompts, texts, tokenizer, args)
        ret['weights'] = torch.tensor(weights)
        return ret
    
    return collator


def rjs_data_collator(tokenizer: PreTrainedTokenizer, args) -> Callable[[Dict[str, any]], Dict[str, torch.Tensor]]:
    def collator(examples) -> Dict[str, torch.Tensor]:
        best_texts: List[str] = []
        for example in examples:
            texts = example['texts']
            scores = example['scores']
            best_text = texts[torch.argmax(torch.tensor(scores))]
            best_texts.append(best_text)
        
        if not args.only_predict_answer:
            prompts = None
            texts = best_texts
        else:
            prompts = []
            texts = []
            for text in best_texts:
                prompt, answer = text.split(args.sep_token)
                prompts.append(prompt)
                texts.append(prompt + answer)

        return _llm_tokenize(prompts, texts, tokenizer, args)

    return collator
        

def rrhf_data_collator(tokenizer: PreTrainedTokenizer, args) -> Callable[[Dict[str, any]], Dict[str, torch.Tensor]]:
    def collator(examples: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        all_texts: List[str] = []
        all_scores: List[float] = []
        batch_size = len(examples)
        num_sample = max([len(example['texts']) for example in examples])
        for example in examples:
            min_reward = min(example['scores'])
            if len(example['texts']) < num_sample:
                example['texts'].extend(['']*(num_sample - len(example['texts'])))
                example['scores'].extend([min_reward - 1]*(num_sample - len(example['texts'])))
            all_texts.extend(example['texts'])
            all_scores.extend(example['scores'])
        if not args.only_predict_answer:
            prompts = None
            texts = all_texts
        else:
            prompts = []
            texts = []
            for text in all_texts:
                prompt, answer = text.split(args.sep_token)
                prompts.append(prompt)
                texts.append(prompt + answer)
            
        ret = _llm_tokenize(prompts, texts, tokenizer, args)
        ret['scores'] = torch.tensor(all_scores)
        return {
            "input_ids": ret['input_ids'].view(batch_size, num_sample, -1),
            "labels": ret['labels'].view(batch_size, num_sample, -1),
            "attention_mask": ret['attention_mask'].view(batch_size, num_sample, -1),
            "scores": ret['scores'].view(batch_size, num_sample)
        }
    
    return collator
    
    
def alol_data_collator(tokenizer: PreTrainedTokenizer, args: ALOLTrainingArguments) -> Callable[[Dict[str, any]], Dict[str, torch.Tensor]]:
    def collator(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        prompts = []
        weights = []
        advantages = []
        for example in examples:
            text = example['prompt'] + example['answer']
            texts.append(text)
            prompts.append(example['prompt'])
            weights.append(example['weight'])
            advantages.append(example['advantage'])

        ret = _llm_tokenize(prompts, texts, tokenizer, args)
        ret['weight'] = torch.tensor(weights)
        ret['advantage'] = torch.tensor(advantages)
        return ret

    
    return collator
