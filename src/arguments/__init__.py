from typing import TypeVar
from .training_arguments import SFTTrainingArguments, RMTrainingArguments, ALOLTrainingArguments
from .data_arguments import SFTDataArguments, RMDataArguments, ALOLDataArguments
from .inference_arguments import InferenceArguments


GenericTrainingArguments = TypeVar(
    "GenericTrainingArguments",
    SFTTrainingArguments,
    RMTrainingArguments,
    ALOLTrainingArguments
)


GenericDataArguments = TypeVar(
    "GenericDataArguments",
    SFTDataArguments,
    RMDataArguments,
    ALOLDataArguments
)