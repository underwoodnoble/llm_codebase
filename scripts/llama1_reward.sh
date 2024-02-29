REPO_DIR=repo_dir
DATA_DIR=${REPO_DIR}/data/preference_data
TRAIN_DATA_LIST="${DATA_DIR}/helpful/helpful.train.json ${DATA_DIR}/harmless/harmless.train.json ${DATA_DIR}/webgpt/webgpt.train.json ${DATA_DIR}/gpt4llm/gpt4llm.train.json"
EVAL_DATA_LIST="${DATA_DIR}/helpful/helpful.test.json ${DATA_DIR}/harmless/harmless.test.json ${DATA_DIR}/webgpt/webgpt.test.json ${DATA_DIR}/gpt4llm/gpt4llm.test.json"
OUTPUT_DIR=output_dir
MODEL_PATH=model_path

# wandb setting
export WANDB_MODE=offline
export WANDB_DIR=${OUTPUT_DIR}
export WANDB_JOB_NAME=llama_general_rm_all
export WANDB_NAME=${WANDB_JOB_NAME}

deepspeed --num_gpus 8 ${REPO_DIR}/main.py \
    --do_train True \
    --task_type reward \
    --data_paths ${TRAIN_DATA_LIST} \
    --eval_data_paths ${EVAL_DATA_LIST} \
    --preference_data_text_name text \
    --preference_data_score_name score \
    --model_type llama \
    --model_name_or_path ${MODEL_PATH} \
    --model_max_length 512 \
    --add_lm_loss True \
    --lm_loss_coeff 0.0 \
    --lm_score_thresh 0.85 \
    --calibration_bins 5 10 15 20 \
    --remove_unused_columns False \
    --report_to wandb \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation 1 \
    --num_train_epochs 1 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 300 \
    --learning_rate 1e-6 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --warmup_steps 100 \
    --logging_first_step True \
    --gradient_checkpointing True \
    --bf16 True \
    --deepspeed ${REPO_DIR}/configs/8A100/llama_reward_stage3.json