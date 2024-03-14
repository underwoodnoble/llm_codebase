REPO_DIR=repo_dir
OUTPUT_DIR=output_dir
DATA_DIR=${REPO_DIR}/data/preference_data

export CUDA_VISIBLE_DEVICES=0
python3 ${REPO_DIR}/main.py \
    --task_type reward \
    --training_type lora \
    --lora_config_path ${REPO_DIR}/configs/lora_config.json \
    --model_type llama \
    --model_name_or_path model_name_or_path \
    --output_dir ${OUTPUT_DIR} \
    --do_train True \
    --data_paths ${DATA_DIR}/helpful/helpful.train.json ${DATA_DIR}/harmless/harmless.train.json \
    --eval_data_paths ${DATA_DIR}/helpful/helpful.test.json ${DATA_DIR}/harmless/harmless.test.json\
    --preference_data_text_name text \
    --preference_data_score_name score \
    --remove_unused_columns False \
    --report_to none \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy no \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --eval_steps 500 \
    --evaluation_strategy steps \
    --logging_first_step True