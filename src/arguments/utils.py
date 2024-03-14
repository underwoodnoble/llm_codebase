import json
from .training_arguments import TrainingArguments

def load_lora_config_for_json(args: TrainingArguments):
    from peft import LoraConfig, TaskType

    if args.lora_config_path is None:
        raise ValueError("You mush set lora_config_path in TrainingArguments if your training_type is lora.")
    with open(args.lora_config_path, 'r') as f:
        config = json.load(f)
    return LoraConfig(**config)