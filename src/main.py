from arguments import CustomArguments
from transformers import HfArgumentParser, Trainer
from trainer import RewardModelTrainer, ContrastiveTrainer, RRHFTrainer
from utils import print_rank_0, getDataset, loadTokenizerAndModel
from collator import (reward_data_collator, sft_data_collator, rjs_data_collator, 
                    rrhf_data_collator, contrastive_data_collator, classfication_data_collator)
from metrics import compute_reward_metrics, compute_classification_metrics
from trl import DPOTrainer


def main():
    parser = HfArgumentParser((CustomArguments,))
    args = parser.parse_args_into_dataclasses()[0]
    print_rank_0(args)

    if args.do_train:
        train_dataset = getDataset(args, type='train')
    eval_dataset = getDataset(args, type='eval')
    print_rank_0(">"*10 + "training set")
    print_rank_0(train_dataset)
    print_rank_0(">"*10 + "evaluation set")
    print_rank_0(eval_dataset)

    tokenizer, model, ref_model = loadTokenizerAndModel(args)
    print_rank_0(">"*10 + "tokenizer")
    print_rank_0(tokenizer)
    print_rank_0(">"*10 + "model")
    print_rank_0(model)

    if args.task_type == 'reward':
        trainer = RewardModelTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=reward_data_collator(tokenizer),
            compute_metrics=compute_reward_metrics
            )
    elif args.task_type in ['sft', 'offline_rejection_sampling']:
        if args.task_type == 'sft':
            print_rank_0("Using sft data collator")
            data_collator = sft_data_collator(tokenizer, args)
        elif args.task_type == 'offline_rejection_sampling':
            print_rank_0("Using rejection sampling data collator")
            data_collator = rjs_data_collator(tokenizer, args)

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
    elif args.task_type == 'contrastive_learning':
        print_rank_0("Using contrastive learning data collator")
        data_collator = contrastive_data_collator(tokenizer, args)

        trainer = ContrastiveTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
    elif args.task_type == 'classification':
        print_rank_0("Using classification data collator")
        data_collator = classfication_data_collator(tokenizer, args)
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_classification_metrics
        )
    elif args.task_type == 'offline_RRHF':
        print_rank_0("Using offline RRHF data collator")
        data_collator = rrhf_data_collator(tokenizer, args)
        trainer = RRHFTrainer(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer
        )
    elif args.task_type == 'DPO':
        print_rank_0("Using DPO data collator")
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=args,
            beta=args.dpo_beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            label_pad_token_id=args.ignore_token_id,
            padding_value=tokenizer.pad_token_id,
            max_length=tokenizer.model_max_length,
            max_prompt_length=args.max_prompt_length
        )

    trainer.train()

if __name__ == '__main__':
    main()