from transformers import Qwen2Tokenizer
from src.utils import read_json_or_jsonl_data


def main():
    file_path = "/apdcephfs_cq10/share_1567347/share_info/nobelhu/code/llm-self-play/data/neo/result_conversations.json"
    model_name_or_path = "/apdcephfs_cq10/share_1567347/share_info/llm_models/Qwen1.5-7B"
    tokenizer = Qwen2Tokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    dataset = read_json_or_jsonl_data(file_path)
    for data in dataset:
        prompt = tokenizer.apply_chat_template(data[:-1], add_generation_prompt=True, tokenize=False)
        total = tokenizer.apply_chat_template(data, add_generation_prompt=False, tokenize=False)
        answer = total[len(prompt):]
        print(prompt)
        print('='*100)
        print(answer)
        break

if __name__ == '__main__':
    main()