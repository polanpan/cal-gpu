@echo off
REM GPU显存计算器启动脚本 (Windows)

echo ===================================
echo  GPU显存计算器 - 启动中...
echo ===================================

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查依赖
echo 检查依赖...
python -c "import gradio, pandas" >nul 2>&1
if errorlevel 1 (
    echo 安装依赖...
    pip install -r requirements.txt
)

REM 启动服务
echo 启动GPU显存计算器...
echo 浏览器将自动打开：http://localhost:7860
echo 按 Ctrl+C 停止服务
echo.

python gpu_memory_calculator.py
pause
