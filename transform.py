"""
The function of this script is to convert the conversation data into the specified format for subsequent training
"""
from argparse import ArgumentParser
from utils import read_json_or_jsonl_data
from typing import List, Dict
import json


def data_to_dialog(dataset, args):
    new_dataset = []
    for data in dataset:
        if args.input_format == 'hh':
            text = data['text'][0]
            temps = [t.split('\n\nAssistant: ') for t in text.split('\n\nHuman: ') if len(t) > 0]
            ret = []
            for temp in temps:
                ret.extend(temp)
            print("ret>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.")
            print(ret)
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
        elif args.input_format == 'alpaca':
            system_message = data[args.system_name]
            user_message = data[args.input_name]
            assistant_message = data[args.output_name]
            new_dataset.append([
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_message
                },
                {
                    "role": "assistant",
                    "content": assistant_message
                }
            ])
        elif args.input_format == 'dialog':
            new_data = []
            if data[0]['role'] == args.system_name:
                new_data.append({
                    "role": "system",
                    "content": data[0]['content']
                })
            for turn in data[1:]:
                pass

    
    return new_dataset


def format_llama(dialog: List[Dict[str, str]], bos_token: str, eos_token: str, args):
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    B_INST, E_INST = "[INST]", "[/INST]"
    if dialog[0]['role'] == args.system_name:
        dialog = [
            {
                "role": dialog[1]['role'],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"]
            }
        ] + dialog[2:]
    assert all([msg['role'] == args.input_name for msg in dialog[::2]]) and all(
        [msg['role'] == args.output_name for msg in dialog[1::2]]
    ), (
        f"model only supports {args.system_name}, {args.input_name} and {args.output_name} roles, "
        f"starting with {args.system_name}, the {args.input_name} and alternation ({args.input_name}/{args.output_name})"
    )
    turns: List[str] =[
        bos_token + f"{B_INST} {(p['content']).strip()} {E_INST} {(a['content']).strip()} " + eos_token
        for p, a in zip(dialog[::2],dialog[1::2])]

    assert (
        dialog[-1]["role"] == args.input_name
    ), f"Last message must be from {args.input_name}, got {dialog[-1]['role']}"
    turns.append(bos_token + f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}")
    
    prompt = ''
    for turn in turns:
        prompt += turn
    return prompt


def dialog_to_format(dialogs, args):
    new_dataset = []
    for dialog in dialogs:
        if args.output_format == 'alpaca':
            PROMPT_DICT = {
                "prompt_input": (
                    "Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
                ),
                "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"), 
            }
            if args.input_format == 'alpaca':
                if len(dialog[1]['content']) > 0:
                    prompt = PROMPT_DICT['prompt_input'].format(instruction=dialog[0]['content'], input=dialog[1]['content'])
                else:
                    prompt = PROMPT_DICT['prompt_no_input'].format(instruction=dialog[0]['content'])
                answer = dialog[2]['content']
        
            new_dataset.append({
                "prompt": prompt,
                "answer": answer
            })

    return new_dataset


def main(args):
    data_list = read_json_or_jsonl_data(args.data_path)
    dialogs = data_to_dialog(data_list, args)
    new_dataset = dialog_to_format(dialogs, args)

    with open(args.save_path, 'w') as f:
        json.dump(new_dataset, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--input_format', type=str, choices=['hh', 'dialog', 'alpaca']
    )
    parser.add_argument(
        '--output_format', type=str, choices=['alpaca', 'llama', 'hh']
    )
    parser.add_argument(
        '--system_name', type=str, default='system'
    )
    parser.add_argument(
        '--input_name', type=str, default='user'
    )
    parser.add_argument(
        '--output_name', type=str, default='assistant'
    )
    parser.add_argument(
        '--data_path', type=str
    )
    parser.add_argument(
        '--save_path', type=str
    )
    parser.add_argument(
        '--only_predict_last', type=bool, default=False
    )
    args = parser.parse_args()
    main(args)