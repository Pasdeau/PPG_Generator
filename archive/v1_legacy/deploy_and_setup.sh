#!/bin/bash
# PPG远程部署和训练 - 完整流程
# 使用SLURM作业调度系统

echo "=========================================="
echo "PPG远程部署和训练 - 完整流程"
echo "=========================================="

# 1. 清理旧文件
echo -e "\n[1/6] 清理旧文件..."
rm -rf ppg_deployment.tar.gz ppg_project
echo "✓ 完成"

# 2. 解压包
echo -e "\n[2/6] 解压PPG_Python_v1.2.tar.gz..."
if [ ! -f PPG_Python_v1.2.tar.gz ]; then
    echo "错误: PPG_Python_v1.2.tar.gz 不存在"
    echo "请先上传: scp PPG_Python_v1.2.tar.gz front.convergence.lip6.fr:~/"
    exit 1
fi

tar -xzf PPG_Python_v1.2.tar.gz
cd PPG_generation || exit 1
mv ../PPG_generation ~/ppg_project 2>/dev/null || true
cd ~/ppg_project || exit 1
echo "✓ 完成"

# 3. 检查环境
echo -e "\n[3/6] 检查环境..."
python3 --version
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "⚠️  前端节点无GPU（正常）"
echo "✓ 完成"

# 4. 安装依赖
echo -e "\n[4/6] 安装Python依赖..."
pip3 install --user -q -r requirements.txt
pip3 install --user -q -r ml_training/requirements.txt
echo "✓ 完成"

# 5. 创建日志目录
echo -e "\n[5/6] 创建必要目录..."
mkdir -p logs checkpoints_resnet_waveform
echo "✓ 完成"

# 6. 上传SLURM脚本
echo -e "\n[6/6] 准备SLURM作业脚本..."
if [ ! -f slurm_datagen_v2.sh ] || [ ! -f slurm_train_v2.sh ]; then
    echo "⚠️  SLURM V2脚本未找到 (请确保已上传)"
else
    chmod +x slurm_*.sh
    echo "✓ 完成"
fi

echo -e "\n=========================================="
echo "V2.0 部署完成！(UNet Segmentation)"
echo "=========================================="

echo -e "\n下一步操作:"
echo ""
echo "1. 提交V2数据生成作业 (生成Masks和Robust Signals):"
echo "   cd ~/ppg_project"
echo "   sbatch slurm_datagen_v2.sh"
echo ""
echo "2. 数据生成完成后，提交V2训练作业:"
echo "   sbatch slurm_train_v2.sh"
echo ""
echo "3. 查看日志:"
echo "   tail -f logs/datagen_v2_*.out"
echo "   tail -f logs/train_v2_*.out"
echo "=========================================="
