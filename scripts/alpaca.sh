REPO_DIR=repo_dir
DATA_DIR=${REPO_DIR}/data/sft_data
deepspeed --num_gpus 8 ${REPO_DIR}/src/main.py \
    --task_type sft \
    --model_type llama \
    --do_train True \
    --data_paths ${DATA_DIR}/alpaca_train.json \
    --sft_data_prompt_name prompt \
    --sft_data_answer_name answer \
    --model_name_or_path Llama2 \
    --output_dir output_dir \
    --remove_unused_columns False \
    --report_to none \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 1 \
    --num_train_epochs 3 \
    --model_max_length 512 \
    --learning_rate 2e-5 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --logging_first_step False \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --bf16 True \
    --deepspeed ${REPO_DIR}/configs/8A100/alpaca_stage3.json