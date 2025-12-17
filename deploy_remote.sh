#!/bin/bash
# 远程GPU服务器部署脚本

echo "=========================================="
echo "PPG ML训练 - 远程部署"
echo "=========================================="

# 1. 清理旧文件
echo -e "\n[1/5] 清理旧文件..."
rm -rf ppg_deployment.tar.gz ppg_project
echo "✓ 旧文件已删除"

# 2. 解压新包
echo -e "\n[2/5] 解压PPG_Python_v1.2.tar.gz..."
tar -xzf PPG_Python_v1.2.tar.gz
mv PPG_generation ppg_project
cd ppg_project
echo "✓ 包已解压到 ppg_project/"

# 3. 检查Python和GPU
echo -e "\n[3/5] 检查环境..."
python3 --version
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "⚠️  未检测到GPU"

# 4. 安装依赖
echo -e "\n[4/5] 安装Python依赖..."
pip3 install --user -r requirements.txt
pip3 install --user -r ml_training/requirements.txt
echo "✓ 依赖已安装"

# 5. 生成数据集
echo -e "\n[5/5] 生成训练数据集 (20,000样本)..."
echo "这将需要约30-40分钟..."
python3 batch_generate.py --num_samples 20000 --output_dir ml_dataset

echo -e "\n=========================================="
echo "部署完成！"
echo "=========================================="
echo -e "\n数据集位置: ppg_project/ml_dataset/"
echo "样本数量: 20,000"
echo -e "\n下一步: 开始训练"
echo "  cd ppg_project"
echo "  python3 ml_training/train.py --data_dir ml_dataset --model resnet --epochs 150"
