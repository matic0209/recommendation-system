#!/bin/bash
# 修复 Docker swap limit 警告
# 警告: 需要重启系统才能生效

set -e

echo "=============================================="
echo "修复 Docker Swap Limit 警告"
echo "=============================================="
echo ""

# 检查是否已启用
if grep -q "cgroup_enable=memory swapaccount=1" /etc/default/grub 2>/dev/null; then
    echo "✓ Swap limit 已经启用"
    echo ""
    echo "如果仍然看到警告，可能需要重启系统。"
    exit 0
fi

echo "当前状态: Swap limit 未启用"
echo ""

# 检查 GRUB 配置
if [ ! -f /etc/default/grub ]; then
    echo "错误: 未找到 /etc/default/grub"
    echo "此脚本仅支持使用 GRUB 引导的系统"
    exit 1
fi

echo "即将修改 GRUB 配置..."
echo ""
echo "修改内容:"
echo "  在 GRUB_CMDLINE_LINUX 添加: cgroup_enable=memory swapaccount=1"
echo ""

read -p "是否继续? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 备份原始配置
sudo cp /etc/default/grub /etc/default/grub.backup.$(date +%Y%m%d_%H%M%S)
echo "✓ 已备份 GRUB 配置"

# 修改 GRUB 配置
if grep -q "^GRUB_CMDLINE_LINUX=" /etc/default/grub; then
    # 已存在 GRUB_CMDLINE_LINUX，追加参数
    sudo sed -i 's/GRUB_CMDLINE_LINUX="\(.*\)"/GRUB_CMDLINE_LINUX="\1 cgroup_enable=memory swapaccount=1"/' /etc/default/grub
else
    # 不存在，添加新行
    echo 'GRUB_CMDLINE_LINUX="cgroup_enable=memory swapaccount=1"' | sudo tee -a /etc/default/grub
fi

echo "✓ 已修改 GRUB 配置"

# 更新 GRUB
sudo update-grub
echo "✓ 已更新 GRUB"

echo ""
echo "=============================================="
echo "配置完成！"
echo "=============================================="
echo ""
echo "重要: 需要重启系统才能生效"
echo ""
echo "重启前请确保:"
echo "  1. 保存所有工作"
echo "  2. 停止关键服务"
echo "  3. 通知相关人员"
echo ""
echo "重启命令: sudo reboot"
echo ""
echo "=============================================="
