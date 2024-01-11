from arguments import CustomArguments
from transformers import HfArgumentParser, BertForSequenceClassification, BertTokenizer, BertConfig
from trainer import RewardModelTrainer
from utils import print_rank_0, load_data_from_paths
from datasets import Dataset
from collator import reward_data_collator
from metrics import compute_reward_metrics2
import os


def getDataset(args: CustomArguments, type='train'):
    if type == 'train':
        if args.data_path is not None:
            data_paths = [args.data_path]
        else:
            data_paths = [os.path.join(args.data_dir, path) for path in os.listdir(args.data_dir)]
    else:
        if args.eval_data_path is not None:
            data_paths = [args.eval_data_path]
        else:
            data_paths = [os.path.join(args.eval_data_dir, path) for path in os.listdir(args.eval_data_dir)]       
    data_list = load_data_from_paths(data_paths)

    if args.task_type == 'reward':
        # transform to data: {"texts": ["text1", "text2"], "scores": [s1, s2]}
        if args.preference_data_texts_name != 'texts' or args.preference_data_scores_name != 'scores':
            new_data_list = []
            for data in data_list:
                new_data = {
                    "texts": data[args.preference_data_texts_name],
                    "scores": data[args.preference_data_scores_name]
                }
                new_data_list.append(new_data)
        data_list = new_data_list
    
    return Dataset.from_list(data_list)
    

def loadTokenizerAndModel(args: CustomArguments):
    if args.task_type == 'reward':
        if args.model_type == 'bert':
            config = BertConfig.from_pretrained(args.model_name_or_path)
            config.num_labels = 1
            tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, truncation_side=args.truncation_side)
            tokenizer.model_max_length = args.model_max_length
            model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    
    return tokenizer, model


def main():
    parser = HfArgumentParser((CustomArguments,))
    args = parser.parse_args_into_dataclasses()[0]
    print_rank_0(args)

    if args.do_train:
        train_dataset = getDataset(args, type='train')
    eval_dataset = getDataset(args, type='eval')
    print_rank_0(train_dataset)
    print_rank_0(eval_dataset)

    tokenizer, model = loadTokenizerAndModel(args)
    print_rank_0(model)

    if args.task_type == 'reward':
        trainer = RewardModelTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=reward_data_collator(tokenizer),
            compute_metrics=compute_reward_metrics2
            )
    trainer.train()

if __name__ == '__main__':
    main()