REPO_DIR=repo_dir
DATA_DIR=${REPO_DIR}/data/preference_data/helpful
MODEL_PATH=model_path
OUTPUT_DIR=output_dir
export CUDA_VISIBLE_DEVICES=0
python3 ${REPO_DIR}/main.py \
    --task_type reward \
    --do_train True \
    --data_paths ${DATA_DIR}/helpful.train.json \
    --eval_data_paths ${DATA_DIR}/helpful.test.json \
    --preference_data_text_name text \
    --preference_data_score_name score \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
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
