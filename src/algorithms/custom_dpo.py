from ..arguments import DPODataArguments

def dpo_transform(data_args: DPODataArguments):
    def transform(example):
        return {
            "prompt": example[data_args.prompt_name],
            "chosen": example[data_args.chosen_name],
            "rejected": example[data_args.rejected_name]
        }

    return transform