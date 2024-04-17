import torch
from .arguments import TrainingArguments
from transformers import PreTrainedTokenizer
from typing import List, Dict
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers.trainer_pt_utils import LabelSmoother
from datasets import Dataset
from random import sample


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def _llm_tokenize(prompts: List[str], texts: List[str], tokenizer: PreTrainedTokenizer, args: TrainingArguments) -> Dict[str, torch.Tensor]:
    input_ids = []
    labels = []
    for prompt, text in zip(prompts, texts):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        response_start_idx = len(prompt_ids)
        if prompt_ids != text_ids[:response_start_idx]:
            response_start_idx -= 1
            prompt_ids = text_ids[:response_start_idx]
        if args.add_special_tokens:
            text_ids = [tokenizer.bos_token_id] + text_ids + [tokenizer.eos_token_id]
        if len(text_ids) > args.model_max_length:
            text_ids = text_ids[-args.model_max_length:]
        label = deepcopy(text_ids)
        if args.only_predict_answer:
            label[:len(prompt_ids) + 1] = [IGNORE_TOKEN_ID] * (len(prompt_ids) + 1)

        input_ids.append(torch.tensor(text_ids))
        labels.append(torch.tensor(label))
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    if args.pad_labels_with_ignore:
        labels = pad_sequence(labels, batch_first=True, padding_value=-IGNORE_TOKEN_ID)
    else:
        labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)

    if args.debug_mode: # If debug_model is True, then pad the sequence into the max token length.
        input_ids = F.pad(input_ids, (0, tokenizer.model_max_length - input_ids.shape[0]), mode='constant', value=tokenizer.pad_token_id)
        labels = F.pad(labels, (0, tokenizer.model_max_length - input_ids.shape[0]), mode='constant', value=args.ignore_token_id 
                    if args.pad_labels_with_ignore else tokenizer.pad_token_id)

    attention_mask = torch.ne(input_ids, tokenizer.pad_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    

def classfication_data_collactor(tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    def collator(examples):
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
        

def multi_object_classification_data_collactor(tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    def collator(examples):
        pass


def reward_data_collactor(tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    def collator(examples: List[Dict[str, List]]):
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

    
def sft_data_collactor(tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    def collator(examples):
        texts = []
        prompts = []
        for example in examples:
            text = example['prompt'] + example['answer']
            texts.append(text)
            prompts.append(example['prompt'])

        return _llm_tokenize(prompts, texts, tokenizer, args)
    
    return collator


def rjs_data_collactor(tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    def collator(examples):
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
        

def rrhf_data_collactor(tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    def collator(examples: List[Dict[str, List]]):
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

    
def weighted_data_collactor(tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    def collator(examples):
        texts = []
        prompts = []
        scores = []
        for example in examples:
            text = example['prompt'] + example['answer']
            texts.append(text)
            prompts.append(example['prompt'])
            scores.append(example['score'])

        ret = _llm_tokenize(prompts, texts, tokenizer, args)
        ret["scores"] = torch.tensor(scores)
        return ret
    
    return collator


def kto_data_collactor(tokenizer: PreTrainedTokenizer, args: TrainingArguments, dataset: Dataset):
    def collator(examples):
        texts = []
        kl_texts = []
        kl_prompts = []
        prompts = []
        scores = []
        for example in examples:
            text = example['prompt'] + example['completion']
            texts.append(text)
            prompts.append(example['prompt'])
            scores.append(example['score'])
            kl_completions = sample(dataset['completion'], args.kto_num_of_kl_completions_per_prompt)
            for completion in kl_completions:
                kl_texts.append(example['prompt'] + completion)
            kl_prompts.extend([example['prompt']] * args.kto_num_of_kl_completions_per_prompt)
        
        target = _llm_tokenize(prompts, texts, tokenizer, args)
        kl = _llm_tokenize(kl_prompts, kl_completions, tokenizer, args)
        target['scores'] = torch.tensor(scores)
        return {
            "target": target,
            "kl": kl
        }
    
    return collator