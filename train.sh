#!/bin/bash

# DINOv2 ViT-B/16 训练脚本
# 实验名称：dinov2_vitb16_96px

EXP_NAME="dinov2_vits16_96px"
CHECKPOINT_DIR="./checkpoints/${EXP_NAME}"
OUTPUT_FILE="${CHECKPOINT_DIR}/${EXP_NAME}.out"

# 创建 checkpoint 目录（如果不存在）
mkdir -p ${CHECKPOINT_DIR}

# 构建训练命令
CMD="python train.py \
    --method dinov2 \
    --backbone_type vit_s_16 \
    --img_size 96 \
    --batch_size 512 \
    --epochs 50 \
    --lr 5e-4 \
    --weight_decay 1e-4 \
    --optimizer_type adamw \
    --scheduler_type cosine \
    --warmup_epochs 10 \
    --temperature 0.1 \
    --proj_hidden_dim 2048 \
    --proj_output_dim 65536 \
    --aug_strength strong \
    --use_amp \
    --save_freq 10 \
    --log_freq 100 \
    --num_workers 8 \
    --dataset_type local \
    --dataset_root ../data \
    --exp_name ${EXP_NAME}"

# 输出命令到日志文件
echo "==========================================" >> ${OUTPUT_FILE}
echo "实验名称: ${EXP_NAME}" >> ${OUTPUT_FILE}
echo "开始时间: $(date)" >> ${OUTPUT_FILE}
echo "==========================================" >> ${OUTPUT_FILE}
echo "" >> ${OUTPUT_FILE}
echo "训练命令:" >> ${OUTPUT_FILE}
echo "${CMD}" >> ${OUTPUT_FILE}
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
echo "   日志文件: ${OUTPUT_FILE}"
echo "   查看日志: tail -f ${OUTPUT_FILE}"

