from transformers import LlamaTokenizer, LlamaForCausalLM
import os
import json
from typing import List, Dict
import transformers
import argparse
import torch
import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, Sequence
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from datasets import load_dataset
import copy

IGNORE_INDEX = -100

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
    ), f"Last message must be from user, got {dialog[-1]['role']}, dialog is :{dialog}"
    dialog_tokens += tokenizer.encode(
        f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        add_special_tokens=True
    )
    return dialog_tokens

def padding(inputs, padding_token, cutoff = None):
    num_elems = len(inputs)
    if cutoff is None:
        cutoff = max([len(item) for item in inputs])
    tokens = torch.ones(num_elems, cutoff).long().to(inputs[0].device) * padding_token
    for i in range(num_elems):
        toks = inputs[i]
        length = min(cutoff, len(toks))
        tokens[i, -length:] = toks[-length:]
    return tokens

def sequence_gather(s, world_size, pad_tok_id):
    local_size = torch.tensor(s.size(), device=s.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_length = max(size[1] for size in all_sizes)
    length_diff = max_length.item() - local_size[1].item()
    if length_diff:
        pad_size = (*s.shape[:-1], length_diff)
        padding = torch.ones(pad_size, device=s.device, dtype=s.dtype) * pad_tok_id
        s = torch.concat((s, padding), dim = -1)
    gathered_s = [torch.ones_like(s)*pad_tok_id for _ in range(world_size)]
    dist.all_gather(gathered_s, s)

    return gathered_s
    

def read_json(data_path):
    all_test_data = []
    with open(data_path, 'r') as jfile:
        all_test_data = json.load(jfile)
        print(all_test_data[0])
        # for line in jfile.readlines():
        #     all_test_data.append(json.loads(line))

    return all_test_data

def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    new_strings = list(map(text_to_dialog, strings))

    tokenized_list = [
        torch.tensor(dilog_to_tokens(tokenizer, text))
        # tokenizer(
        #     text,
        #     return_tensors="pt",
        #     padding="longest",
        #     max_length=tokenizer.model_max_length,
        #     truncation=True,
        # )
        for text in new_strings
    ]
    input_ids = labels = tokenized_list
    input_ids_lens = labels_lens = [
        torch.tensor(tokenized).ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
):
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    input_ids = sources_tokenized["input_ids"]
    return dict(input_ids=input_ids, labels=copy.deepcopy(input_ids))

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        all_data_dict = read_json(data_path)
        list_data_dict = all_data_dict[:]
        sources = []
        for example in list_data_dict:
            sources.append(example['text'][0]) # sources: ['user:..., assisten:..., ...', '...', ...]
        if int(os.environ["LOCAL_RANK"]) == 0:
            print("total number of sources in each rank:", len(sources))
            print(sources[0])
            print(sources[6])

        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.query_id = [[example['query_id']] for example in list_data_dict]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], query_id=self.query_id[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, query_id = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", 'query_id'))
        input_ids = padding(input_ids, self.tokenizer.pad_token_id, cutoff = 512)
        labels = padding(labels, IGNORE_INDEX, cutoff = 512)

        return dict(
            input_ids=input_ids,
            labels=labels,
            query_id=torch.tensor(query_id).to(input_ids.device),
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path):
    """Make dataset and collator for supervised fine-tuning."""
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return eval_dataset, data_collator

def main(rank, args):
    dist.init_process_group("nccl")
    world_size = torch.cuda.device_count()
    base_model = args.base_model
    data_path = args.data_path
    batch_size = args.batch_size

    if 'pytorch_model.bin' not in base_model:
        model = LlamaForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                # device_map=rank
            )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
        )
        ckpt_state = torch.load(base_model)
        ckpt_state = {k[11:]:v for k, v in ckpt_state.items() if k.startswith('base_model.')}
        model.load_state_dict(ckpt_state, strict=False)

    tokenizer = AutoTokenizer.from_pretrained(base_model,
                                              padding_side="left",  # for batch decode
                                              truncation_side='left')
                                            
    torch.cuda.set_device(rank)
    model.to(torch.cuda.current_device())
    model = DDP(model, device_ids=[torch.cuda.current_device()])
    model.eval()

    eval_dataset, data_collator = make_supervised_data_module(tokenizer, data_path)

    sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        eval_dataset, 
        shuffle=False, 
        collate_fn=data_collator, 
        batch_size=batch_size,
        sampler=sampler,
    )
    
    # generation_config = GenerationConfig(
    #     temperature=1.2,  # default=0.8
    #     do_sample=True,
    #     min_length=1,
    #     max_new_tokens=256,
    #     num_return_sequences=4,
    #
    # )
    generation_config = GenerationConfig(
        temperature=0.3,
        do_sample=True,
        max_new_tokens=2048,
        top_k=5,
        top_p=0.85,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        repetition_penalty=1.05,
    )

    all_outputs = []
    progress_bar = tqdm(range(len(dataloader)), disable=(rank!=0))
    for step, batch in enumerate(dataloader):
        progress_bar.update(1)
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        query_id = batch['query_id'].to(model.device)
        with torch.no_grad():
            generation_output = model.module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
            )
        s = generation_output.sequences
        this_batch_size = input_ids.size(0)

        gather_outputs = sequence_gather(s, world_size, tokenizer.pad_token_id)
        gathered_inputs = sequence_gather(input_ids, world_size, tokenizer.pad_token_id)
        gather_outputs = torch.stack(gather_outputs).reshape(world_size, this_batch_size,1,-1)
        gathered_inputs = torch.stack(gathered_inputs)
        gather_outputs = gather_outputs.transpose(0,1).reshape(this_batch_size*world_size*1, -1)
        gathered_inputs = gathered_inputs.transpose(0,1).reshape(this_batch_size*world_size,-1)
        outputs_string = tokenizer.batch_decode(gather_outputs, skip_special_tokens=True)
        inputs_string = tokenizer.batch_decode(gathered_inputs, skip_special_tokens=True)

        gathered_query_id = sequence_gather(query_id, world_size, tokenizer.pad_token_id)
        gathered_query_id = torch.stack(gathered_query_id).transpose(0, 1).reshape(this_batch_size * world_size)
        query_id_list = gathered_query_id.tolist()
        
        for idx in range(len(inputs_string)):
            temp = []
            # for i in range(4):
            temp.append([inputs_string[idx], outputs_string[idx].replace(inputs_string[idx], ''), query_id_list[idx]])
            all_outputs.append(temp)
        
        if rank == 0 and (step % 500 == 0):
            output_list = []
            for item in all_outputs:
                tmp = {}
                tmp["instruction"] = item[0][0]
                tmp["query_id"] = item[0][2]
                for j, ans in enumerate(item):
                    tmp[f"ans_{j}"] = ans[1]
                output_list.append(tmp)
            print(output_list[-1])
            with open(args.out_path + f'/raw_generation_step{step}.json', 'w', encoding='utf-8') as f:
                json.dump(output_list, f, ensure_ascii=False, indent=2)

    if rank == 0:
        output_list = []
        for item in all_outputs:
            tmp = {}
            tmp["instruction"] = item[0][0]
            tmp["query_id"] = item[0][2]
            for j, ans in enumerate(item):
                tmp[f"ans_{j}"] = ans[1]
            output_list.append(tmp)
        with open(args.out_path + f'/raw_generation.json', 'w', encoding='utf-8') as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--base_model", default="", type=str, help="model path")
    parser.add_argument("--data_path", default="", type=str, help="config path")
    parser.add_argument("--batch_size", type=int, default=0, help="batch size")
    parser.add_argument("--port", type=int, default=0, help="batch size")
    parser.add_argument("--diverse_beam", type=int, default=1, help="batch size")
    parser.add_argument("--out_path", default="", type=str, help="config path")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    main(local_rank, args)

    # with open("/apdcephfs_qy3/share_301372554/share_info/peixincao/data/test_dataset/hh_cleaned_origin.test.json", 'r') as f:
    #     dataset = json.load(f)
    #     tokenizer = LlamaTokenizer.from_pretrained('/apdcephfs_qy3/share_301372554/share_info/peixincao/llm_models/Llama-2-7b-chat-hf')
    #     results = []
    #     for data in dataset:
    #         dialog = text_to_dialog(data['text'][0])
    #         results.append(dilog_to_tokens(tokenizer, dialog))