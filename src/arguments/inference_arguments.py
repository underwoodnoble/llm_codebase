from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class InferenceArguments(TrainingArguments):
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Model name or path."})
    data_paths: Optional[List[str]] = field(default=None, metadata={"help": "paths of dataset"})
    save_names: Optional[List[str]] = field(default=None, metadata={"help": "name of file to save"})
    save_steps: Optional[int] = field(default=50, metadata={"help": "save steps"})
    inference_type: Optional[str] = field(default=None, metadata={"help": "name of inference type"})
    model_type: Optional[str] = field(default=None, metadata={"help": "model type"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "batch size"})
    task_type: Optional[str] = field(default='llm_inference')


    # rm inference arguments
    data_texts_name: Optional[str] = field(default='texts')

    # llm inference arguments
    data_prompt_name: Optional[str] = field(default='prompt')
    num_return_sequences: Optional[int] = field(default=1)

    # zero inference arguments

    # deepspeed inference argumetns
    checkpoint_type: Optional[str] = field(default=None)
    tp_size: Optional[int] = field(default=1)


