from random import sample
import json


name = "harmless"
dataset_type = 'test'
ratio = 0.2
with open(f"/apdcephfs_cq10/share_1567347/share_info/nobelhu/code/llm_codebase/data/preference_data/{name}/{name}.{dataset_type}.json", 'r') as f:
    dataset = json.load(f)

new_dataset = dataset[:int(ratio*len(dataset))]

with open(f"/apdcephfs_cq10/share_1567347/share_info/nobelhu/code/llm_codebase/data/preference_data/{name}/mini_{name}.{dataset_type}.json", 'w') as f:
    json.dump(new_dataset, f)