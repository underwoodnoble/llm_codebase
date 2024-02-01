# Large Language Model Codebase
This repository is a framework for beginners to dive in llm training. The repository is based on huggingface transformers.

## Overview
* Supported task：Classification, Reward Model Training, Supervised Finetuning, Rejection Sampling, Weighted Learning, RRHF

* Supported model: bert, llama

| Model | Supported training method | 
| --- | --- | 
| bert | Classification, Reward Model Training |
| llama | Reward Model Training, Supervised Finetuning, Rejection Sampling, RRHF, Weighted Learning|
| alpaca | Reward Model Training, Supervised Finetuning, Rejection Sampling, RRHF, Weighted Learning|

## Installation
```bash
git clone https://github.com/underwoodnoble/llm_codebase.git
cd llm_codebase
pip install -r requirements/normal.txt
```
## Classification

There are two ways to train a classification model in the repository. The first one is to train a model with a classification head. The second one is to train a model with a classification dict head. The classification dict head is a head that can be used to train a model with multiple classification tasks.

### One Classification Task

Supported Models: Bert, Llama, Alpaca

Under the hood, the classification head is a linear layer on top of the model output. This repository uses transformers AutoModelForSequenceClassification to train a classification model. The following code shows how to train a classification model with a classification head.

#### Bert
```bash
REPO_DIR=repo_dir
DATA_DIR=data_dir
python3 ${REPO_DIR}/src/main.py \
    --task_type classification \
    --model_type bert \
    --do_train True \
    --data_paths ${DATA_DIR}/train.json \
    --eval_data_paths ${DATA_DIR}/test.json \
    --cls_data_text_name text \
    --cls_data_label_name label \
    --cls_data_label_nums 3 \
    --num_train_epochs 10 \
    --model_name_or_path bert-base-uncased
    --output_dir output_dir \
    --remove_unused_columns False \
    --report_to none \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy no \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --eval_steps 10 \
    --evaluation_strategy steps \
    --logging_first_step True
```

### Multiple Classification Tasks

## Reward Model Training

## Supervised Finetuning

## Rejection Sampling

