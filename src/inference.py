from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, PreTrainedTokenizer, PreTrainedModel, AutoTokenizer
from model.RewardModel import PythiaRewardModel, LlamaRewardModel
import json
from typing import List, Dict, Tuple
import torch
from argparse import ArgumentParser
from utils import read_json_or_jsonl_data
from accelerate import PartialState


def load_dataset(args):
    data_list = read_json_or_jsonl_data(args.data_path)
    if args.debug_mode:
        data_list = data_list[:100]
    if args.add_data_id:
        for i, data in enumerate(data_list):
            data['id'] = i
    return data_list

def hh_to_dialog(text: str):
    temps = [t.split('\n\nAssistant: ') for t in text.split('\n\nHuman: ') if len(t) > 0]
    ret = []
    for temp in temps:
        ret.extend(temp)
    dialog = []
    role = 'user'
    for text in ret[:-1]:
        dialog.append(
            {"role": role, 'content': text}
        )
        if role == 'user':
            role = 'assistant'
        else:
            role = 'user'
    return dialog

def dialog_to_llama_tokens(tokenizer: LlamaTokenizer, dialog: List[Dict[str, str]]):
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    B_INST, E_INST = "[INST]", "[/INST]"
    if dialog[0]['role'] == 'system':
        dialog = [
            {
                "role": dialog[1]['role'],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"]
            }
        ] + dialog[2:]
    assert all([msg['role'] == 'user' for msg in dialog[::2]]) and all(
        [msg['role'] == 'assistant' for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', the 'user' and alternation (u/a/u/a/u...)"
    )
    dialog_tokens: List[int] = sum(
        [
            tokenizer.encode(
                f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ", add_special_tokens=True
            )
            for prompt, answer in zip(
                dialog[::2],
                dialog[1::2]
            )
        ],
        [],
    )
    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"
    dialog_tokens += tokenizer.encode(
        f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        add_special_tokens=True
    )
    return dialog_tokens


def load_tokenizer_and_model(args) -> Tuple[PreTrainedModel, PreTrainedModel]:
    if args.task_type == 'llm_inference':
        if args.model_type == 'llama':
            tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
            model = LlamaForCausalLM.from_pretrained(args.model_path)
    elif args.task_type == 'reward_model_inference':
        if args.model_type == 'pythia':
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trucation_side='left', trust_remote_code=True)
            model = PythiaRewardModel.from_pretrained(args.model_name_or_path)
        elif args.model_type == 'llama':
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trucation_side='left')
            model = LlamaRewardModel.from_pretrained(args.model_name_or_path)

    return tokenizer, model


def llm_inference(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, 
                  generation_config: GenerationConfig, dataset: List[Dict[str, str]], args):
    new_dataset = []
    for data in dataset:
        answers = []
        for _ in range(args.sample_num):
            prompt = data['prompt']
            print("prompt>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print(prompt)
            input_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)
            output = model.generate(torch.tensor(input_ids).unsqueeze(0).to(model.device),
                           generation_config=generation_config)
            response = tokenizer.batch_decode(output, skip_special_tokens=True)
            answer = response[0][len(prompt):]
            print("answer>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print(answer)
            answers.append(answer)
        new_data = {"id": data['id'], "prompt": prompt, "answer": answers}
        new_dataset.append(new_data)
    with open(args.save_path, 'a+') as f:
        for data in new_dataset:
            f.write(json.dumps(data) + '\n')
            
def reward_model_inference(tokenzier: PreTrainedTokenizer, model: PreTrainedModel, 
                           dataset: List[Dict[str, str]], args):
    new_dataset = []
    for data in dataset:
        prompt = data['prompt']
        answers = data['answers']
        texts = [prompt + answer for answer in answers]
        input_ids = tokenzier(texts, padding=True, truncation=True)['input_ids']
        with torch.no_grad():
            rewards: torch.Tensor = model(input_ids=torch.tensor(input_ids).to(model.device))['rm_logits']
        new_data = {
            "texts": [prompt+'<sep>'+answer for answer in answers],
            "scores": rewards.tolist()
        }
        new_dataset.append(new_data)
    with open(args.save_path, 'a+') as f:
        for new_data in new_dataset:
            s = json.dumps(new_data)
            f.write(s+'\n')

def main(args):
    distributed_state = PartialState()

    if distributed_state.is_main_process:
        print("Loading Dataset")
    dataset = load_dataset(args)
    if distributed_state.is_main_process:
        print("Loading Model")
    tokenizer, model = load_tokenizer_and_model(args)
    model.to(distributed_state.device)
    for i in range(0, len(dataset), args.chunk_size):
        with distributed_state.split_between_processes(dataset[i:i+args.chunk_size]) as sub_dataset:
            if args.task_type == 'llm_inference':
                generation_config = GenerationConfig.from_pretrained(args.model_path, do_sample=True, max_new_tokens=args.max_new_tokens)
                llm_inference(tokenizer, model, generation_config, sub_dataset, args)
            elif args.task_tye == 'reward_model_inference':
                reward_model_inference(tokenizer, model, dataset)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--model_type', type=str, choices=['llama', 'pythia']
    )
    parser.add_argument(
        '--task_type', type=str, choices=['llm_inference', 'reward_model_inference']
    )
    parser.add_argument(
        '--model_path', type=str
    )
    parser.add_argument(
        '--data_path', type=str
    )
    parser.add_argument(
        '--add_data_id', type=bool, default=True
    )
    parser.add_argument(
        '--save_path', type=str
    )
    parser.add_argument(
        '--data_type', type=str, choices=['helpful_and_harmless']
    )
    parser.add_argument(
        '--max_new_tokens', type=int, default=256
    )
    parser.add_argument(
        '--chunk_size', type=int, default=128
    )
    parser.add_argument(
        '--sample_num', type=int, default=4
    )
    
    parser.add_argument(
        '--debug_mode', type=bool, default=False
    )
    args = parser.parse_args()
    main(args)