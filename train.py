from src.utils.args_utils import get_args
from src.arguments import GenericDataArguments, GenericTrainingArguments
from src.utils.data_utils import load_dataset
from src.utils.model_utils import load_tokenizer_and_model
from src.utils.general_utils import print_object_on_main_process


def main(
    algorithm: str,
    training_args: GenericTrainingArguments,
    data_args: GenericDataArguments
    ):

    # Loading Dataset
    train_dataset, eval_dataset = load_dataset(data_args, algorithm)
    print_object_on_main_process("train_dataset", train_dataset)
    print_object_on_main_process("eval_dataset", eval_dataset)
    

    # Loading Model
    tokenizer, model, ref_model = load_tokenizer_and_model(training_args, algorithm)
    print_object_on_main_process("tokenizer", tokenizer)
    print_object_on_main_process("model", model)


    # Initialize Trainer
    if algorithm == 'sft':
        from src.algorithms.sft import SFTTrainer
        from src.collator import sft_data_collator
        trainer = SFTTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            data_collator=sft_data_collator(tokenizer, training_args),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer
        )

    # Operation before training
    

    # Train
    trainer.train()
    
    # Operation after training
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == '__main__':
    algorithm, (training_args, data_args) = get_args()
    main(algorithm, training_args, data_args)