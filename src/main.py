from arguments import CustomArguments
from transformers import HfArgumentParser, BertForSequenceClassification, \
    BertTokenizer, BertConfig, LlamaForSequenceClassification, LlamaConfig, LlamaTokenizer, LlamaForCausalLM, Trainer
from model.reward_model import LlamaRewardModel
from trainer import RewardModelTrainer, ContrastiveTrainer
from utils import print_rank_0, load_data_from_paths, set_llama_special_token
from datasets import Dataset
from collator import reward_data_collator, sft_data_collator, rjs_data_collator, rrhf_data_collator, contrastive_data_collator
from metrics import compute_reward_metrics
import os


def getDataset(args: CustomArguments, type='train'):
    if type == 'train':
        if args.data_paths is not None:
            data_paths = args.data_paths
        else:
            data_paths = [os.path.join(args.data_dir, path) for path in os.listdir(args.data_dir)]
    else:
        if args.eval_data_paths is not None:
            data_paths = args.eval_data_paths
        else:
            data_paths = [os.path.join(args.eval_data_dir, path) for path in os.listdir(args.eval_data_dir)]       
    data_list = load_data_from_paths(data_paths)

    if args.task_type in ['reward', "offline_rejection_sampling", "offline_RRHF"]:
        # transform to the format: {"texts": ["text1", "text2"], "scores": [s1, s2]}
        if args.preference_data_texts_name != 'texts' or args.preference_data_scores_name != 'scores':
            new_data_list = []
            for data in data_list:
                new_data = {
                    "texts": data[args.preference_data_texts_name],
                    "scores": data[args.preference_data_scores_name]
                }
                new_data_list.append(new_data)
            data_list = new_data_list
    elif args.task_type == 'sft':
        if args.sft_data_prompt_name != 'prompt' or args.sft_data_answer_name != 'answer':
            new_data_list = []
            for data in data_list:
                new_data = {
                    "prompt": data[args.sft_data_prompt_name],
                    "answer": data[args.sft_data_answer_name]
                }
                new_data_list.append(new_data)
            data_list = new_data_list
    elif args.task_type == 'contrastive_learning':
        if args.contrastive_data_prompt_name != 'prompt' or args.contrastive_data_answer_name != 'answer' or args.contrastive_data_score_name != 'score':
            new_data_list = []
            for data in data_list:
                new_data = {
                    "prompt": data[args.contrastive_data_prompt_name],
                    "answer": data[args.contrastive_data_answer_name],
                    "score": data[args.contrastive_data_score_name]
                }
    
    return Dataset.from_list(data_list)
    

def loadTokenizerAndModel(args: CustomArguments):
    if args.task_type == 'reward':
        if args.model_type == 'bert':
            config = BertConfig.from_pretrained(args.model_name_or_path)
            config.num_labels = 1
            tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, truncation_side=args.truncation_side)
            tokenizer.model_max_length = args.model_max_length
            model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
        elif args.model_type == 'llama':
            tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, truncation_side=args.truncation_side, padding_side=args.padding_side)
            tokenizer.model_max_length = args.model_max_length
            model = LlamaRewardModel.from_pretrained(args.model_name_or_path)
            set_llama_special_token(tokenizer, model)
        else:
            raise ValueError(f"Training reward model do not support the model type {args.model_type}.")
    elif args.task_type in ['sft', 'offline_rejection_sampling', "offline_RRHF", 'contrastive_learning']:
        if args.model_type == 'llama':
            tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, truncation_side=args.truncation_side, padding_side=args.padding_side)
            tokenizer.model_max_length = args.model_max_length
            model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
            set_llama_special_token(tokenizer, model)
    
    return tokenizer, model


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

    tokenizer, model = loadTokenizerAndModel(args)
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
    elif args.task_type in ['sft', 'offline_rejection_sampling', 'offline_RRHF']:
        if args.task_type == 'sft':
            print_rank_0("Using sft data collator")
            data_collator = sft_data_collator(tokenizer, args)
        elif args.task_type == 'offline_rejection_sampling':
            print_rank_0("Using rejection sampling data collator")
            data_collator = rjs_data_collator(tokenizer, args)
        elif args.task_type == 'offline_RRHF':
            print_rank_0("Using RRHF data collator")
            data_collator = rrhf_data_collator(tokenizer, args)

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

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == '__main__':
    main()