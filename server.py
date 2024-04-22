from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.RewardModel import LlamaRewardModel, PythiaRewardModel
from typing import List
import torch
from fastapi import FastAPI, Body
from pydantic import BaseModel
import argparse
import uvicorn


app = FastAPI()

parser = argparse.ArgumentParser()
parser.add_argument("--host")
parser.add_argument("--port", type=int)
parser.add_argument("--model_path")
parser.add_argument("--task_type")
parser.add_argument("--model_type")
parser.add_argument("--device")
args = parser.parse_args()


class CasualLMModel:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, truncation_side=True, padding_side=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

class RefModel:
    def __init__(self, model_path):
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        if args.device != 'cpu':
            self.ref_model.to(f"cuda:{int(args.device)}")

    @torch.inference_mode()
    def predict(self, input_ids: List[List[int]]):
        input_ids = torch.tensor(input_ids).to(self.ref_model.device)
        outputs = self.ref_model(input_ids, output_hidden_states=True)
        return outputs.hidden_states[-1]
        

class RewardModel:
    def __init__(self, tokenizer, reward_model):
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = 512
        self.reward_model = reward_model
        if args.device != 'cpu':
            self.reward_model.to(f'cuda:{int(args.device)}')

    @torch.inference_mode()
    def predict(self, responses: List[str]):
        reward_inputs = torch.tensor(self.tokenizer(responses, truncation=True, padding=True)['input_ids'])
        rewards = [r.item() for r in self.reward_model(reward_inputs.to(self.reward_model.device))['rm_logits']]
        return rewards 


if args.task_type == 'reference':
    print("Loading reference model")
    ref_model = RefModel(model_path=args.model_path)
    print("Loading Successful!")

    class Inputs(BaseModel):
        input_ids: List[List[int]]

    @app.post('/')
    async def get_last_hidden_state(input_ids: Inputs = Body(default=...)):
        print(input_ids.input_ids)
        last_hidden_state: torch.Tensor = ref_model.predict(input_ids.input_ids)
        print(last_hidden_state.shape)
        return { 
            "last_hidden_state": last_hidden_state.tolist()
        }

elif args.task_type == "reward":
    print("Loading Reward Model")
    if args.model_type == 'llama':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, truncation_side='left', padding_side='right', trust_remote_code=True)
        reward_model = LlamaRewardModel.from_pretrained(args.model_path, trust_remote_code=True)
        reward_model = RewardModel(tokenizer, reward_model)
    elif args.model_type == 'pythia':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, truncation_side='left', padding_side='right', trust_remote_code=True)
        reward_model = PythiaRewardModel.from_pretrained(args.model_path, trust_remote_code=True)
        reward_model = RewardModel(tokenizer, reward_model)
    class Responses(BaseModel):
        response_list: List[str]
    @app.post('/')
    async def get_reward(responses: Responses = Body(default=...)):
        print(responses.response_list)
        rewards = reward_model.predict(responses.response_list)
        return {
            "rewards": rewards
        }


print("Loading Reward Model")
if args.model_type == 'llama':
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, truncation_side='left', padding_side='right', trust_remote_code=True)
    reward_model = LlamaRewardModel.from_pretrained(args.model_path, trust_remote_code=True)
    reward_model = RewardModel(tokenizer, reward_model)
elif args.model_type == 'pythia':
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, truncation_side='left', padding_side='right', trust_remote_code=True)
    reward_model = PythiaRewardModel.from_pretrained(args.model_path, trust_remote_code=True)
    reward_model = RewardModel(tokenizer, reward_model)



if __name__ == '__main__':
    uvicorn.run("server:app", host=args.host, port=args.port)