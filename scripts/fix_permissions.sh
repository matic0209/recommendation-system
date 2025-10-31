#!/bin/bash
# 修复数据目录权限，确保容器内的 appuser 可以写入

set -e

echo "正在修复数据目录权限..."

# 修复 evaluation 目录权限
if [ -d "data/evaluation" ]; then
    chmod 777 data/evaluation
    echo "✓ evaluation 目录权限已修复"
fi

# 修复 exposure_log.jsonl 文件权限（如果存在）
if [ -f "data/evaluation/exposure_log.jsonl" ]; then
    chmod 666 data/evaluation/exposure_log.jsonl
    echo "✓ exposure_log.jsonl 文件权限已修复"
fi

# 修复 logs 目录权限
if [ -d "logs" ]; then
    chmod 777 logs
    echo "✓ logs 目录权限已修复"
fi

echo "权限修复完成！"
