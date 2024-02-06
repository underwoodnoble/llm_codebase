REPO_DIR=repo_dir
python3 ${REPO_DIR}/src/transform.py \
    --input_format alpaca \
    --output_format alpaca \
    --system_name instruction \
    --input_name input \
    --output_name output \
    --data_path ${REPO_DIR}/data/sft_data/alpaca_data.json \
    --save_path ${REPO_DIR}/data/sft_data/alpaca_train.json