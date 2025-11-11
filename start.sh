#!/bin/bash
# GPU显存计算器启动脚本

echo "==================================="
echo " GPU显存计算器 - 启动中..."
echo "==================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误：未找到Python3，请先安装Python 3.8+"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import gradio, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装依赖..."
    pip install -r requirements.txt
fi

# 启动服务
echo "启动GPU显存计算器..."
echo "浏览器将自动打开：http://localhost:7860"
echo "按 Ctrl+C 停止服务"
echo ""

python3 gpu_memory_calculator.py
