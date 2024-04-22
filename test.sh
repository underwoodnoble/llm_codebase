GPU_TYPE=V100
export NCCL_IB_TIMEOUT=24
if [ ${GPU_TYPE} == "V100" ]
then
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECKS_DISABLE=1
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
    export NCCL_SOCKET_IFNAME=eth1
elif [ ${GPU_TYPE} == "A100" ]
then
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=1 
fi

REPO_DIR=/apdcephfs_cq10/share_1567347/share_info/nobelhu/code/llm_codebase

MASTER_ADDR=$CHIEF_IP
MASTER_PORT=6000
NUM_GPUS=$NODE_NUM

OUTPUT_DIR=/apdcephfs_cq10/share_1567347/share_info/nobelhu/code/llm_codebase/
mkdir -p $OUTPUT_DIR
TMP_DIR=${OUTPUT_DIR}/tmp
mkdir -p $TMP_DIR

echo $NODE_IP_LIST > ${TMP_DIR}/env.txt 
# generate hostfile and pssh.hosts
sed "s/:/ slots=/g" ${TMP_DIR}/env.txt | sed "s/,/\n/g" >  ${TMP_DIR}/hostfile
sed "s/:.//g" ${TMP_DIR}/env.txt | sed "s/,/\n/g" >  ${TMP_DIR}/pssh.hosts


bash /apdcephfs_cq10/share_1567347/share_info/nobelhu/code/make_container/unoccupy_plus.sh

deepspeed --hostfile ${TMP_DIR}/hostfile --master_addr ${MASTER_ADDR} --master_port=${MASTER_PORT} \
    ${REPO_DIR}/deepspeed_zero_inference.py \
    --model_name_or_path /apdcephfs_cq10/share_1567347/share_info/llm_models/internlm2-chat-20b \
    --data_path /apdcephfs_cq10/share_1567347/share_info/nobelhu/code/llm_codebase/mydata/T/response_training_data/t_response_training_data_v0.13_step2_after_preprocess.json \
    --output_dir ${OUTPUT_DIR} \
    --save_name t_response_training_data_v0.13_step2_inference \
    --deepspeed /apdcephfs_cq10/share_1567347/share_info/nobelhu/code/llm_codebase/configs/8V100/zero_inference.json  \
    
bash /apdcephfs_cq10/share_1567347/share_info/nobelhu/code/make_container/occupy_plus.sh