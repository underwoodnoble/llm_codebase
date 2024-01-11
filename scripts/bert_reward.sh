REPO_DIR=REPO_DIR
DATA_DIR=${REPO_DIR}/data/preference_data/helpful
export CUDA_VISIBLE_DEVICES=0
python3 ${REPO_DIR}/src/main.py \
    --task_type reward \
    --do_train True \
    --data_path ${DATA_DIR}/helpful.train.json \
    --eval_data_path ${DATA_DIR}helpful.test.json \
    --preference_data_texts_name text \
    --preference_data_scores_name score \
    --model_name_or_path model_name_or_path \
    --output_dir output_dir \
    --remove_unused_columns False \
    --report_to none \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy no \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --eval_steps 500 \
    --evaluation_strategy steps \
    --logging_first_step True
