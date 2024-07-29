import json
import os
from copy import deepcopy

from src.utils.args_utils import get_args
from src.arguments import GenericDataArguments, GenericTrainingArguments
from src.utils.data_utils import load_dataset
from src.utils.model_utils import load_tokenizer_and_model, get_collator_and_trainer
from src.utils.general_utils import print_object_on_main_process, print_rank_0, set_seed


def main(
    algorithm: str,
    training_args: GenericTrainingArguments,
    data_args: GenericDataArguments
    ):

    set_seed(training_args.seed)
    # Loading Dataset
    train_dataset, eval_dataset = load_dataset(data_args, algorithm)
    print_object_on_main_process("train_dataset", train_dataset)
    print_object_on_main_process("eval_dataset", eval_dataset)

    # Loading Model
    tokenizer, model, ref_model, reward_model, value_model = load_tokenizer_and_model(training_args, algorithm)
    print_object_on_main_process("tokenizer", tokenizer)
    print_object_on_main_process("model", model)

    # Create PEFT model.
    if training_args.peft_config_path is not None:
        from peft import get_peft_model
        from src.arguments.utils import load_peft_config_from_json

        peft_config = load_peft_config_from_json(training_args.peft_config_path)
        model = get_peft_model(model, peft_config=peft_config)
        print_rank_0('>' * 100)
        model.print_trainable_parameters()
        print_rank_0('>' * 100)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    compute_metrics = None
    if algorithm == 'rm':
        from src.metrics import compute_reward_metrics
        compute_metrics = lambda x: compute_reward_metrics(training_args, x)

    if algorithm != 'ppo_v2':
        COLLATOR, TRAINER = get_collator_and_trainer(algorithm)
        trainer = TRAINER(
            model=model,
            ref_model=ref_model,
            args=training_args,
            data_collator=COLLATOR(tokenizer, training_args),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
    else:
        from trl.trainer.ppov2_trainer import PPOv2Trainer
        # todo: there may have some tokenize problem
        train_dataset = train_dataset.map(
            lambda x: {"input_ids": tokenizer(x['prompt'], padding=False)['input_ids']},
            remove_columns=train_dataset.column_names,
            batched=True,
            num_proc=4,
            load_from_cache_file=False
        )
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                lambda x: {"input_ids": tokenizer(x['prompt'], padding=False)['input_ids']},
                remove_columns=eval_dataset.column_names,
                batched=True,
                num_proc=4,
                load_from_cache_file=False
            )
        else:
            eval_dataset = train_dataset

        trainer = PPOv2Trainer(
            config=training_args,
            policy=model,
            ref_policy=ref_model,
            reward_model=reward_model,
            value_model=value_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    # Operation before training
    
    # Train
    if training_args.do_train:
        trainer.train()
    
        # Operation after training
        ## save model
        trainer.save_model(output_dir=training_args.output_dir)

    ## final evaluation
    if eval_dataset is not None:
        eval_result = trainer.evaluate(eval_dataset)
        print_object_on_main_process('eval_result', eval_result)

    ## save log history
    with open(os.path.join(training_args.output_dir, 'log_history.txt'), 'w') as f:
        json.dump(trainer.state.log_history, f)

if __name__ == '__main__':
    algorithm, (training_args, data_args) = get_args()
    main(algorithm, training_args, data_args)
