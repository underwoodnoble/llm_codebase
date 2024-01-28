import json
import numpy as np


new_dataset = []
prefix = 'Assistant: '
with open("./data/preference_data/helpful/helpful.test.json", 'r') as f:
    dataset = json.load(f)
    for data in dataset:
        idx = np.argmax(data['score'])
        text = data['text'][idx]
        dialog = text.split('\n\n')
        answer = dialog[-1][len(prefix):]
        prompt = '\n\n'.join(dialog[:-1])
        prompt = prompt + '\n\n' + prefix
        new_dataset.append(
            {
                "prompt": prompt,
                "answer": answer
            }
        )

with open("./data/preference_data/helpful/helpful.ppl.test.json", 'w') as f:
    json.dump(new_dataset, f)