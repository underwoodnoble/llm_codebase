import torch


def reward_data_collator(tokenizer):
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

        encodings = tokenizer(all_texts, padding=True, truncation=True)

        return {
            "input_ids": torch.tensor(encodings['input_ids']).reshape(batch_size, num_sample, -1),
            "attention_mask": torch.tensor(encodings['attention_mask']).reshape(batch_size, num_sample, -1),
            "scores": torch.tensor(all_scores).reshape(batch_size, -1),
        }

    return collator