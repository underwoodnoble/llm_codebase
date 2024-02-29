# Large Language Model Codebase
This repository is a framework for beginners to dive into llm training, inference and evaluation. The repository is based on huggingface transformers.

## Overview

### Training
* Supported task：Classification, Reward Model Training, Supervised Finetuning, Rejection Sampling, Weighted Learning, RRHF
* Supported model: bert, llama

| Model | Supported training method |
| --- | --- | 
| bert | Classification, Reward Model Training |
| llama | Reward Model Training, Supervised Finetuning, Rejection Sampling, RRHF, Weighted Learning|

### Inference
* Supported task: Reward Model Inference, LLM Inference.
* Supported model: llama

### Evaluation
* Supported task: PPL, GPT4 win rate
* Supported model:
    * PPL: llama

## Installation
For most of the tasks, the following packages are required.
```bash
git clone https://github.com/underwoodnoble/llm_codebase.git
cd llm_codebase
pip install -r requirements/normal.txt
```
For DPO, the following packages are required.
```bash
pip install -r requirements/dpo.txt
```
For llama1, the following packages are required.
```bash
pip install -r requirements/llama1.txt
```
## Classification

There are two ways to train a classification model in the repository. The first one is to train a model with a classification head. The second one is to train a model with a classification dict head. The classification dict head is a head that can be used to train a model with multiple classification tasks.

### Single objective classification

Supported Models: Bert, Llama

Under the hood, the classification head is a linear layer on top of the model output. This repository uses transformers AutoModelForSequenceClassification to train a classification model. The following code shows how to train a classification model with a classification head.

Specific Arguments
* cls_data_text_name: field name of the text in data.
* cls_data_label_name: field name of the label in data.
* cls_data_label_nums: number of classes.

#### Bert
```bash
bash scripts/bert_classification.sh
```

#### Llama
You can use accelerate or deepspeed to speed up the training process.
##### Accelerate
```bash
REPO_DIR=repo_dir
DATA_DIR=data_dir
```
##### DeepSpeed
```bash
```

### Multi-objective classification

## Reward Model Training
Supported Models: Bert, Llama
**Specific Arguments**
* preference_data_text_name: field name of the text in data.
* preference_data_score_name: field name of the score in data.
* add_lm_loss: whether to add language model loss to the reward model. (Do not support bert)
* lm_loss_coeff: the coefficient of the language model loss. (Do not support bert)
* pooling_type: the pooling type of the model output. The pooling type can be "last", "max", "eos", or "average". (Do not support bert)
* rm_calibration: whether to calibrate the reward model using ece.
* calibration_bins: ece bins

**Data Format**
```json
{
    "text": ["text1", "text2"],
    "score": ["score1", "score2"]
}
```
* You need to align parameter <font color='orange'><preference_data_text_name></font> with the field name of the text in data and <font color='orange'><preference_data_score_name></font> with the field name of the score in data.
* You can specify multiple datasets in <font color='orange'><data_paths></font> and <font color='orange'><eval_data_paths></font>. If different dataset have different number of texts(scores), the data will be padded to the maximum length of the texts(scores) in the dataset(padding text with " " and padding score with -100). 
### bert
```bash
bash scripts/bert_reward.sh
```
### Llama
For Llama1, we recommend to use the requirements/llama1.txt to install the required packages. Because we find that higher version of transformers may degrade the preformance of Llama1.

```bash
# Install the required packages
pip install -r requirements/llama1.txt
# Train the reward model
bash scripts/llama1_reward.sh
```