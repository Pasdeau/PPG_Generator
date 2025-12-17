# ML Training Module

PPG信号分类的机器学习训练模块

## 文件说明

- `models.py` - 模型定义（CNN, LSTM, CNN+LSTM, ResNet）
- `dataset.py` - 数据加载器和数据增强
- `train.py` - 训练脚本
- `evaluate.py` - 评估脚本
- `requirements.txt` - Python依赖

## 快速开始

### 1. 安装依赖

```bash
pip install -r ml_training/requirements.txt
```

### 2. 生成训练数据

```bash
python batch_generate.py --num_samples 5000 --output_dir training_data
```

### 3. 训练模型

```bash
python ml_training/train.py \
    --data_dir training_data \
    --task waveform \
    --model cnn \
    --epochs 50 \
    --batch_size 32
```

### 4. 评估模型

```bash
python ml_training/evaluate.py \
    --model_path checkpoints/best_model.pth \
    --data_dir training_data \
    --task waveform
```

## 支持的任务

- `waveform` - 波形分类 (5类)
- `artifact` - 伪影分类 (5类)
- `rhythm` - 心律分类 (2类)

## 支持的模型

- `cnn` - 1D CNN（推荐新手）
- `lstm` - LSTM
- `cnn_lstm` - CNN+LSTM混合
- `resnet` - ResNet1D

## 查看训练日志

```bash
tensorboard --logdir=runs
```
