#!/bin/bash

EXP_NAME="dinov2_vits14_96px_official"

SAVE_DIR="/root/autodl-tmp/checkpoints"

CHECKPOINT_DIR="${SAVE_DIR}/${EXP_NAME}"

LOG_DIR="./log"

OUTPUT_FILE="${LOG_DIR}/${EXP_NAME}.out"

# 评估相关配置
EVAL_CUB_DATA_DIR="/root/dl/eval_data/data"  # CUB 数据目录
EVAL_FREQ=5                              # 每5个epoch评估一次
EVAL_METHOD="linear_probe"              # 评估方法: knn 或 linear_probe
EVAL_KNN_K=20                            # k-NN 的 k 值
EVAL_BATCH_SIZE=256                      # 评估时的批次大小
EVAL_NUM_WORKERS=4                       # 评估时的数据加载线程数

# 其他配置
DISABLE_TQDM="true"                      # 是否禁用 tqdm 进度条（推荐在后台运行时设为 true）

# 创建目录（如果不存在）
mkdir -p ${CHECKPOINT_DIR}  # checkpoint 目录（在 /root/autodl-tmp/checkpoints/）
mkdir -p ${LOG_DIR}         # 日志目录（在项目目录下）

# 构建训练命令
# 注意：这是针对 train from scratch 的优化配置
CMD="python train.py \
    --method dinov2 \
    --backbone_type vit_s_14 \
    --img_size 112 \
    --batch_size 256 \
    --epochs 100 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --optimizer_type adamw \
    --scheduler_type cosine \
    --warmup_epochs 10 \
    --temperature 0.1 \
    --proj_hidden_dim 2048 \
    --proj_output_dim 256 \
    --aug_strength strong \
    --use_amp \
    --log_freq 100 \
    --num_workers 8 \
    --dataset_type local \
    --dataset_root ../data \
    --save_dir ${SAVE_DIR} \
    --exp_name ${EXP_NAME} \
    --eval_enabled \
    --eval_cub_data_dir ${EVAL_CUB_DATA_DIR} \
    --eval_freq ${EVAL_FREQ} \
    --eval_method ${EVAL_METHOD} \
    --eval_knn_k ${EVAL_KNN_K} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --eval_num_workers ${EVAL_NUM_WORKERS} \
    --eval_use_cls_token"

# 如果启用禁用 tqdm，添加参数
if [ "${DISABLE_TQDM}" = "true" ]; then
    CMD="${CMD} --disable_tqdm"
fi

# 输出命令到日志文件
echo "==========================================" >> ${OUTPUT_FILE}
echo "实验名称: ${EXP_NAME}" >> ${OUTPUT_FILE}
echo "开始时间: $(date)" >> ${OUTPUT_FILE}
echo "==========================================" >> ${OUTPUT_FILE}
echo "" >> ${OUTPUT_FILE}
echo "训练命令:" >> ${OUTPUT_FILE}
echo "${CMD}" >> ${OUTPUT_FILE}
echo "" >> ${OUTPUT_FILE}
echo "评估配置:" >> ${OUTPUT_FILE}
echo "  CUB 数据目录: ${EVAL_CUB_DATA_DIR}" >> ${OUTPUT_FILE}
echo "  评估频率: 每 ${EVAL_FREQ} 个 epoch" >> ${OUTPUT_FILE}
echo "  评估方法: ${EVAL_METHOD}" >> ${OUTPUT_FILE}
echo "  禁用 tqdm: ${DISABLE_TQDM}" >> ${OUTPUT_FILE}
echo "" >> ${OUTPUT_FILE}
echo "==========================================" >> ${OUTPUT_FILE}
echo "" >> ${OUTPUT_FILE}

# 执行训练（后台运行）
nohup ${CMD} >> ${OUTPUT_FILE} 2>&1 &

# 获取进程ID
PID=$!

echo "🚀 训练已启动！"
echo "   实验名称: ${EXP_NAME}"
echo "   进程ID: ${PID}"
echo "   Checkpoint 目录: ${CHECKPOINT_DIR}"
echo "   日志文件: ${OUTPUT_FILE}"
echo "   查看日志: tail -f ${OUTPUT_FILE}"
echo ""
echo "📊 评估配置:"
echo "   CUB 数据目录: ${EVAL_CUB_DATA_DIR}"
echo "   评估频率: 每 ${EVAL_FREQ} 个 epoch"
echo "   评估方法: ${EVAL_METHOD}"
echo ""
echo "⚙️  其他配置:"
echo "   禁用 tqdm: ${DISABLE_TQDM}"