from transformers import LlamaTokenizer, LlamaForCausalLM
import os
import json
from typing import List, Dict


repo_dir = "/apdcephfs_cq10/share_1567347/nobelhu/code/llm_codebase"

def text_to_dialog(text: str):
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

def dilog_to_tokens(tokenizer: LlamaTokenizer, dialog: List[Dict[str, str]]):
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

with open(os.path.join(repo_dir, "data/preference_data/helpful/helpful.test.json"), 'r') as f:
    dataset = json.load(f)
    tokenizer = LlamaTokenizer.from_pretrained('/apdcephfs_cq10/share_1567347/share_info/llm_models/Llama-2-7b-chat-hf')
    for data in dataset[:10]:
        dialog = text_to_dialog(data['text'][0])
        print(dilog_to_tokens(tokenizer, dialog))