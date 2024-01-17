REPO_DIR=repo_dir
DATA_DIR=${REPO_DIR}/data/preference_data/helpful
deepspeed --num_gpus 1 ${REPO_DIR}/src/main.py \
    --task_type offline_rejection_sampling \
    --model_type llama \
    --do_train True \
    --data_path ${DATA_DIR}/helpful.train.json \
    --eval_data_path ${DATA_DIR}/helpful.test.json \
    --preference_data_texts_name text \
    --preference_data_scores_name score \
    --model_name_or_path alpaca-7b-converted \
    --output_dir saved_model/alpaca_rs \
    --remove_unused_columns False \
    --report_to none \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy no \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --logging_first_step True \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --deepspeed ${REPO_DIR}/configs/alpaca_rjs/ds_stage3.json