#!/bin/bash
# SVG Logo 训练环境一键搭建
# 用法: bash setup.sh

set -e

echo "===== 1. 安装依赖 ====="
pip install ms-swift==4.4.1 "transformers>=5.0" peft==0.19.1 \
  datasets==3.1.0 accelerate "torchao>=0.16.0"

echo "===== 2. 下载模型 (ModelScope, 国内更快) ====="
pip install modelscope -q
modelscope download --model LLM-Research/gemma-3-270m-it --local_dir ./gemma3-270m

echo "===== 3. 下载训练数据 ====="
git clone https://github.com/roboticcam/logo-detailed-prompt

echo "===== 4. 转换数据格式 ====="
python convert_data.py

echo "===== Done! 运行 python run_train.py 开始训练 ====="
