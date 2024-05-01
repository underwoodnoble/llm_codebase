from .base import BaseTrainer
from ..arguments import OfflinePPOTrainingArguments



class OfflinePPOTrainer(BaseTrainer):
    args: OfflinePPOTrainingArguments