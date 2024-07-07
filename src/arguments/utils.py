import json


def load_peft_config_from_json(peft_config_path):
    from peft import LoraConfig, PromptEncoderConfig

    with open(peft_config_path, 'r') as f:
        json_file = json.load(f)
        peft_type = json_file['type']
        peft_config = json_file['config']
    if peft_type == 'lora':
        return LoraConfig(**peft_config)
    elif peft_type == 'p-tuning':
        return PromptEncoderConfig(**peft_config)
    else:
        raise ValueError(f"Do not support peft type {peft_type}")

        
