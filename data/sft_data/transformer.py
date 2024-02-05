import json

new_dataset = []
with open("data/sft_data/alpaca_data.json", 'r') as f:
    
    dataset = json.load(f)
    for data in dataset:
        prompt = "Human: " + data['instruction'] + '\n'
        if len(data['input']) > 0:
            prompt += data['input'] + '\n'
        prompt += '\nAssistant: '
        answer = data['output']
        new_data = {
            "prompt": prompt,
            "answer": answer
        }
        new_dataset.append(new_data)


with open("data/sft_data/alpaca_train.json", 'w') as f:
    json.dump(new_dataset, f)