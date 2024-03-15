REPO_DIR=repo_dir
DATA_DIR=data_dir
MODEL_PATH=model_path
OUTPUT_DIR=output_dir
python3 ${REPO_DIR}/main.py \
    --task_type classification \
    --model_type bert \
    --do_train True \
    --data_paths ${DATA_DIR}/train.json \
    --eval_data_paths ${DATA_DIR}/test.json \
    --cls_data_text_name text \
    --cls_data_label_name label \
    --cls_data_label_nums 3 \
    --num_train_epochs 10 \
    --model_name_or_path ${MODEL_PATH} \
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