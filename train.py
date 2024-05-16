import json
import os

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
    tokenizer, model, ref_model = load_tokenizer_and_model(training_args, algorithm)
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


    COLLATOR, TRAINER = get_collator_and_trainer(algorithm)
    trainer = TRAINER(
        model=model,
        ref_model=ref_model,
        args=training_args,
        data_collator=COLLATOR(tokenizer, training_args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    # Operation before training
    
    # Train
    trainer.train()
    
    # Operation after training
    ## save model
    trainer.save_model(output_dir=training_args.output_dir)
    ## save log history
    with open(os.path.join(training_args.output_dir, 'log_history.txt'), 'w') as f:
        json.dump(trainer.state.log_history, f)


if __name__ == '__main__':
    algorithm, (training_args, data_args) = get_args()
    main(algorithm, training_args, data_args)
