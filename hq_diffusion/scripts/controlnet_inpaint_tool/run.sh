#!/bin/bash

# SD3 Inpainting 工具启动脚本

cd "$(dirname "$0")"

echo "==================================="
echo "SD3 Inpainting 工具"
echo "==================================="
echo ""
echo "正在启动 Flask 应用..."
echo "访问地址: http://0.0.0.0:5000"
echo ""

python app.py

