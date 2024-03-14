import json
from .training_arguments import TrainingArguments

def load_peft_config_from_json(args: TrainingArguments):
    from peft import LoraConfig, PromptEncoderConfig

    if args.peft_config_path is None:
        raise ValueError("You mush set lora_config_path in TrainingArguments if your training_type is lora.")
    with open(args.peft_config_path, 'r') as f:
        config = json.load(f)
    if args.training_type == 'lora':
        return LoraConfig(**config)
    else:
        return PromptEncoderConfig(**config)