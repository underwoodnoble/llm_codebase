from ..arguments import PPOv2DataArguments

def ppo_transform(data_args: PPOv2DataArguments):
    def transform(example):
        return {
            "prompt": example[data_args.prompt_name]
        }

    return transform