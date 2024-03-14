from src.arguments import TrainingArguments
from transformers import HfArgumentParser
from src.utils import print_rank_0, getDataset, loadTokenizerAndModel
from typing import Dict
import os
import json


def main():
    parser = HfArgumentParser((TrainingArguments,))
    args: TrainingArguments = parser.parse_args_into_dataclasses()[0]
    print_rank_0(args)

    print_rank_0("Loading data>>>>>>>>>>>>>>>>>>>>>>>>>>>")
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

    if args.training_type in ['lora', 'p-tuning']:
        from peft import get_peft_model
        from src.arguments.utils import load_peft_config_from_json
        peft_config = load_peft_config_from_json(args)
        model = get_peft_model(model, peft_config)
        print_rank_0(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        model.print_trainable_parameters()
        print_rank_0(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    if args.task_type == 'reward':
        from src.trainers import RewardModelTrainer
        from src.collator import reward_data_collator
        from src.metrics import compute_reward_metrics
        trainer = RewardModelTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=reward_data_collator(tokenizer, args),
            compute_metrics=lambda x: compute_reward_metrics(args, x)
            )
    elif args.task_type in ['sft', 'offline_rejection_sampling', 'classification']:
        from transformers import Trainer
        if args.task_type == 'sft':
            from src.collator import sft_data_collator

            print_rank_0("Using sft data collator")
            data_collator = sft_data_collator(tokenizer, args)
            compute_metrics = None

        elif args.task_type == 'offline_rejection_sampling':
            from src.collator import rjs_data_collator

            print_rank_0("Using rejection sampling data collator")
            data_collator = rjs_data_collator(tokenizer, args)
            compute_metrics = None
        
        elif args.task_type == 'classification':
            from src.collator import classfication_data_collator
            from src.metrics import compute_classification_metrics

            print_rank_0("Using classification data collator")
            data_collator = classfication_data_collator(tokenizer, args)
            compute_metrics = compute_classification_metrics

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

    elif args.task_type == 'weighted_learning':
        from src.trainers import WeightedTrainer
        from src.collator import weighted_data_collator

        print_rank_0("Using weighted learning data collator")
        data_collator = weighted_data_collator(tokenizer, args)

        trainer = WeightedTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

    elif args.task_type == 'offline_RRHF':
        from src.collator import rrhf_data_collator
        from src.trainers import RRHFTrainer

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
        from trl import DPOTrainer
        
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
            max_prompt_length=args.max_prompt_length,
        )
    
    elif args.task_type == 'KTO':
        from src.collator import kto_data_collator
        from src.trainers import KTOTrainer
        print_rank_0("Using KTO data collator")
        data_collator = kto_data_collator(tokenizer, args, train_dataset)
        
        model.ref_model = ref_model
        for param in model.ref_model.parameters():
            param.requires_grad = False

        trainer = KTOTrainer(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer
        )
        

    if args.evaluate_at_beginning and eval_dataset is not None:
        from src.callbacks import EvaluateFirstStepCallback
        trainer.add_callback(EvaluateFirstStepCallback())

    trainer.train()
    if args.save_training_states:
        trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)

    print_rank_0(">>>>>> Final evaluation:")
    if eval_dataset is not None:
        if isinstance(eval_dataset, Dict):
            eval_result = {}
            for key, dataset in eval_dataset.items():
                result = trainer.evaluate(dataset, metric_key_prefix="eval_"+key)
                eval_result.update(result)
        else:
            eval_result = trainer.evaluate()
        print_rank_0(eval_result)
        trainer.log_metrics('eval', eval_result)
        with open(os.path.join(args.output_dir, 'final_results.json'), 'w') as f:
            json.dump(eval_result, f)

if __name__ == '__main__':
    main()