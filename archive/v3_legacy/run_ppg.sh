#!/bin/bash
# PPG生成器快速启动脚本

echo "====================================="
echo "PPG Signal Generator"
echo "====================================="
echo ""

# 检查虚拟环境是否存在
if [ ! -d ".venv" ]; then
    echo "❌ 虚拟环境不存在"
    echo "请先创建虚拟环境: python3 -m venv .venv"
    exit 1
fi

# 检查依赖是否安装
if ! .venv/bin/python -c "import matplotlib" 2>/dev/null; then
    echo "⚠️  依赖未安装，正在安装..."
    .venv/bin/pip install -r requirements.txt
    echo ""
fi

echo "✓ 使用虚拟环境运行 PPG 生成器"
echo ""

# 运行主程序
.venv/bin/python main_ppg.py

echo ""
echo "====================================="
echo "✓ 完成！查看 output/ 目录获取结果"
echo "====================================="
