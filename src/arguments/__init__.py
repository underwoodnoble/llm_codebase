from typing import TypeVar
from .training_arguments import SFTTrainingArguments, RMTrainingArguments, OfflinePPOTrainingArguments, DPOTrainingArguments, PPOv2TrainingArguments
from .data_arguments import SFTDataArguments, RMDataArguments, OfflinePPODataArguments, DPODataArguments, PPOv2DataArguments
from .inference_arguments import InferenceArguments


GenericTrainingArguments = TypeVar(
    "GenericTrainingArguments",
    SFTTrainingArguments,
    RMTrainingArguments,
    OfflinePPOTrainingArguments,
    DPOTrainingArguments,
    PPOv2TrainingArguments
)


GenericDataArguments = TypeVar(
    "GenericDataArguments",
    SFTDataArguments,
    RMDataArguments,
    OfflinePPODataArguments,
    DPODataArguments,
    PPOv2DataArguments
)