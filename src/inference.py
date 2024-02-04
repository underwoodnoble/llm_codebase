from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import os
import json
from typing import List, Dict
import torch


repo_dir = "/apdcephfs_cq10/share_1567347/nobelhu/code/llm_codebase"

def text_to_dialog(text: str):
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

def dialog_to_tokens(tokenizer: LlamaTokenizer, dialog: List[Dict[str, str]]):
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
    generation_config = GenerationConfig.from_pretrained('/apdcephfs_cq10/share_1567347/share_info/llm_models/Llama-2-7b-chat-hf', do_sample=True, max_length=512)
    model = LlamaForCausalLM.from_pretrained('/apdcephfs_cq10/share_1567347/share_info/llm_models/Llama-2-7b-chat-hf')
    model.to('cuda:0')
    print(generation_config)
    results = []
    for data in dataset[:20]:
        dialog = text_to_dialog(data['text'][0])
        print("Dialog>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(dialog)
        outputs = model.generate(
            (torch.tensor(dialog_to_tokens(tokenizer, dialog))).unsqueeze(0).to(model.device),
            generation_config=generation_config
        )
        print("Reply>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(result)
        results.append(result)
    
    with open(os.path.join(repo_dir, 'inference_results.json'), 'w') as f:
        json.dump(results, f)
