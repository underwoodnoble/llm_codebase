from transformers import LlamaConfig
from typing import Optional


class LlamaClassiferConfig(LlamaConfig):
    def __init__(self, cls_info: Optional[dict]=None, **kwargs):
        super().__init__(**kwargs)
        self.cls_info: Optional[dict] = cls_info
        self.cls_label_nums: Optional[dict] = {key: len(item['labels']) for key, item in cls_info.items()} if cls_info is not None else None
        self.cls_labels: Optional[dict] = {key: item['labels'] for key, item in cls_info.items()} if cls_info is not None else None
        self.cls_coeffs: Optional[dict] = {key: item['coeff'] for key, item in cls_info.items()} if cls_info is not None else None