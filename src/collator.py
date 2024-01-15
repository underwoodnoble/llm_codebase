import torch
from arguments import CustomArguments
from transformers import PreTrainedTokenizer
from typing import List
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
from utils import print_rank_0


def reward_data_collator(tokenizer: PreTrainedTokenizer):
    def collator(examples):
        batch_size = len(examples)
        num_sample = max([len(example['texts']) for example in examples])
        all_texts = []
        all_scores = []
        for example in examples:
            if len(example['texts']) < num_sample:
                example['texts'].extend(['']*(num_sample - len(example['texts'])))
                example['socores'].extend([-1.]*(num_sample - len(example['texts'])))
            all_texts.extend(example['texts'])
            all_scores.extend(example['scores'])

        for i in range(len(all_texts)):
            all_texts[i] += tokenizer.eos_token
        encodings = tokenizer(all_texts, padding=True, truncation=True)

        return {
            "input_ids": torch.tensor(encodings['input_ids']).reshape(batch_size, num_sample, -1),
            "attention_mask": torch.tensor(encodings['attention_mask']).reshape(batch_size, num_sample, -1),
            "scores": torch.tensor(all_scores).reshape(batch_size, -1),
        }

    return collator

    
def sft_data_collator(tokenizer: PreTrainedTokenizer, args: CustomArguments):
    def collator(examples):
        texts = []
        prompts = []
        for example in examples:
            text = example['prompt'] + example['answer']
            texts.append(text + tokenizer.eos_token)
            prompts.append(example['prompt'])

        if not args.only_predict_answer:
            encoding = tokenizer(texts, padding=True, truncation=True)
            return {
                'input_ids': torch.tensor(encoding['input_ids']),
                'attention_mask': torch.tensor(encoding['attention_mask']),
                'labels': torch.tensor(encoding['input_ids'])
            }
        else:
            input_ids = []
            labels = []
            for prompt, text in zip(prompts, texts):
                prompt_ids = tokenizer.encode(prompt)
                text_ids = tokenizer.encode(text)
                label = deepcopy(text_ids)
                label[:len(prompt_ids)] = [args.ignore_token_id] * len(prompt_ids)
                input_ids.append(torch.tensor(text_ids[-args.model_max_length:]))
                labels.append(torch.tensor(label[-args.model_max_length:]))

            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            labels = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_mask = torch.ne(input_ids, tokenizer.pad_token_id)

            print_rank_0(input_ids)
            print_rank_0(labels)
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask
            } 
    
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
            encoding = tokenizer(best_texts, padding=True, truncation=True)
            return {
                'input_ids': torch.tensor(encoding['input_ids']),
                'attention_mask': torch.tensor(encoding['attention_mask']),
                'labels': torch.tensor(encoding['input_ids'])
            }
        else:
            input_ids = []
            labels = []
            for text in best_texts:
                query, _ = text.split(args.sep_token)
                query_ids = tokenizer.encode(query)
                text_ids = tokenizer.encode(text)
                label = deepcopy(text_ids)
                label[:len(query_ids)] = [args.ignore_token_id] * len(query_ids)
                input_ids.append(torch.tensor(text_ids[-args.model_max_length:]))
                labels.append(torch.tensor(label[-args.model_max_length:]))

            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            labels = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_mask = torch.ne(input_ids, tokenizer.pad_token_id)


            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask
            }

    return collator
        
def rrhf_data_collator(tokenizer: PreTrainedTokenizer, args: CustomArguments):
    def collator(examples):
        print(examples)
    
    return collator