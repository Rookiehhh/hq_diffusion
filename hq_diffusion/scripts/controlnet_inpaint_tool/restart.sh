#!/bin/bash

# 重启Flask应用脚本

cd "$(dirname "$0")"

echo "==================================="
echo "正在停止旧的Flask应用..."
echo "==================================="

# 查找并停止运行中的app.py进程
pkill -f "python.*app.py" 2>/dev/null
sleep 2

# 再次确认
ps aux | grep -E "python.*app.py" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null

echo "旧进程已停止"
echo ""
echo "==================================="
echo "正在启动新的Flask应用..."
echo "==================================="
echo "访问地址: http://0.0.0.0:5000"
echo "测试页面: http://0.0.0.0:5000/test"
echo ""

python app.py

