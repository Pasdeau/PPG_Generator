#!/bin/bash
# 安装所有必需的Python依赖

echo "安装PyTorch和ML训练依赖..."

# 安装PyTorch (CPU版本，GPU版本会在训练时自动使用CUDA)
pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip3 install --user tensorboard scikit-learn seaborn tqdm

echo "✓ 依赖安装完成"

# 验证安装
python3 -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
