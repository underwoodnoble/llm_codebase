import torch
from arguments import CustomArguments
from transformers import PreTrainedTokenizer
from typing import List, Dict
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
from utils import print_rank_0


def _llm_tokenize(prompts: List[str], texts: List[str], tokenizer: PreTrainedTokenizer, args: CustomArguments) -> Dict[str, torch.Tensor]:
    input_ids = []
    labels = []
    for prompt, text in zip(prompts, texts):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        text_ids = tokenizer.encode(text, add_special_tokens=True)
        label = deepcopy(text_ids)
        if not args.only_predict_answer:
            label[:len(prompt_ids)+1] = [args.ignore_token_id] * (len(prompt_ids) + 1)
        input_ids.append(torch.tensor(text_ids[-args.model_max_length:]))
        labels.append(torch.tensor(label[-args.model_max_length:]))
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    if args.pad_labels_with_ignore:
        labels = pad_sequence(input_ids, batch_first=True, padding_value=args.ignore_token_id)
    else:
        labels = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.ne(input_ids, tokenizer.pad_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    

def classfication_data_collator(tokenizer: PreTrainedTokenizer, args: CustomArguments):
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
        

def multi_object_classification_data_collator(tokenizer: PreTrainedTokenizer, args: CustomArguments):
    def collator(examples):
        pass


def reward_data_collator(tokenizer: PreTrainedTokenizer):
    def collator(examples: List[Dict[str, List]]):
        batch_size = len(examples)
        num_sample = max([len(example['texts']) for example in examples])
        all_texts = []
        all_scores = []
        for example in examples:
            min_reward = min(example['scores'])
            if len(example['texts']) < num_sample:
                example['texts'].extend(['']*(num_sample - len(example['texts'])))
                example['scores'].extend([min_reward - 1]*(num_sample - len(example['scores'])))
            all_texts.extend(example['texts'])
            all_scores.extend(example['scores'])

        for i in range(len(all_texts)):
            all_texts[i] += tokenizer.eos_token
        encodings = tokenizer(all_texts, padding=True, truncation=True, add_special_tokens=True)

        return {
            "input_ids": torch.tensor(encodings['input_ids']).reshape(batch_size, num_sample, -1),
            "attention_mask": torch.tensor(encodings['attention_mask']).reshape(batch_size, num_sample, -1),
            "scores": torch.tensor(all_scores).reshape(batch_size, -1)
        }

    return collator

    
def sft_data_collator(tokenizer: PreTrainedTokenizer, args: CustomArguments):
    def collator(examples):
        texts = []
        prompts = []
        for example in examples:
            text = example['prompt'] + example['answer']
            texts.append(text)
            prompts.append(example['prompt'])

        return _llm_tokenize(prompts, texts, tokenizer, args)
    
    return collator


def rjs_data_collator(tokenizer: PreTrainedTokenizer, args: CustomArguments):
    def collator(examples):
        best_texts: List[str] = []
        for example in examples:
            texts = example['texts']
            scores = example['scores']
            best_text = texts[torch.argmax(torch.tensor(scores))]
            best_texts.append(best_text + tokenizer.eos_token)
        
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
        

def rrhf_data_collator(tokenizer: PreTrainedTokenizer, args: CustomArguments):
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

    
def weighted_data_collator(tokenizer: PreTrainedTokenizer, args: CustomArguments):
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

