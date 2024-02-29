REPO_DIR=/apdcephfs_cq10/share_1567347/share_info/nobelhu/code/llm_codebase
DATA_DIR=${REPO_DIR}/mydata/cls_data/daiyong
MODEL_PATH=/apdcephfs_cq10/share_1567347/share_info/nobelhu/models/bert-base-uncased
OUTPUT_DIR=/apdcephfs_cq10/share_1567347/share_info/nobelhu/saved_model/test/bert_rewrad_test
python3 ${REPO_DIR}/main.py \
    --task_type classification \
    --model_type bert \
    --do_train True \
    --data_paths ${DATA_DIR}/es_favor_train_balanced.json \
    --eval_data_paths ${DATA_DIR}/es_favor_test.json \
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