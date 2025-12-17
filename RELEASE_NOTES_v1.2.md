# PPG Python v1.2 Release Notes

## 新增功能

### 🤖 机器学习训练模块 (ml_training/)

完整的深度学习训练系统，用于PPG信号分类：

#### 模型架构
- **CNN1D** - 1D卷积神经网络（推荐新手）
- **LSTM** - 长短期记忆网络
- **CNN+LSTM** - 混合模型
- **ResNet1D** - 残差网络

#### 训练工具
- **train.py** - 完整训练脚本（早停、TensorBoard）
- **evaluate.py** - 评估工具（混淆矩阵、ROC曲线）
- **dataset.py** - 数据加载器和数据增强

#### 支持任务
- 波形分类 (5类)
- 伪影分类 (5类)
- 心律分类 (2类)

### 📊 批量数据生成

- **batch_generate.py** - 简化的批量生成脚本
- 支持自定义样本数量
- 随机组合波形、心律、伪影类型

---

## 版本对比

| 功能 | v1.0 | v1.1 | v1.2 |
|------|------|------|------|
| PPG生成 | ✅ | ✅ | ✅ |
| FFT峰值标注 | ❌ | ✅ | ✅ |
| 干净波形 | ❌ | ✅ | ✅ |
| 批量生成 | ❌ | ❌ | ✅ |
| ML训练 | ❌ | ❌ | ✅ |
| 模型代码 | ❌ | ❌ | ✅ |

---

## 快速开始

### 数据生成
```bash
python batch_generate.py --num_samples 5000 --output_dir training_data
```

### 模型训练
```bash
python ml_training/train.py \
    --data_dir training_data \
    --task waveform \
    --model cnn \
    --epochs 50
```

### 模型评估
```bash
python ml_training/evaluate.py \
    --model_path checkpoints/best_model.pth \
    --data_dir training_data
```

---

## 文件结构

```
PPG_Python_v1.2/
├── README.md
├── LICENSE
├── requirements.txt
│
├── 核心模块
│   ├── ppg_pulse.py
│   ├── ppg_generator.py
│   ├── ppg_artifacts.py
│   └── data_loader.py
│
├── 主脚本
│   ├── main_ppg.py
│   ├── batch_generate.py          # 新增
│   └── generate_training_data.py
│
├── ML训练模块 (新增)
│   ├── ml_training/
│   │   ├── models.py
│   │   ├── dataset.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── requirements.txt
│   │   └── README.md
│
├── 示例
│   ├── examples/
│   │   ├── demo_hr_resp_coupling.py
│   │   ├── validate_ppg.py
│   │   └── validate_artifact_isolation.py
│
└── 数据
    └── data/
        ├── artifact_param.mat
        └── pulse_templates.mat
```

---

## 升级指南

从v1.1升级到v1.2：

1. 安装ML依赖：
   ```bash
   pip install -r ml_training/requirements.txt
   ```

2. 生成训练数据：
   ```bash
   python batch_generate.py --num_samples 10000 --output_dir ml_data
   ```

3. 开始训练：
   ```bash
   python ml_training/train.py --data_dir ml_data
   ```

---

## 推荐工作流程

### 研究人员
1. 使用`batch_generate.py`生成大规模数据集
2. 使用`ml_training/train.py`训练分类器
3. 使用`ml_training/evaluate.py`评估性能

### 开发者
1. 使用`main_ppg.py`生成单个PPG信号
2. 使用`examples/`中的脚本验证功能
3. 集成到自己的应用中

---

## 性能预期

| 模型 | 数据量 | 训练时间(GPU) | 准确率 |
|------|--------|---------------|--------|
| CNN1D | 5,000 | 15-20分钟 | 85-90% |
| CNN1D | 10,000 | 30-40分钟 | 88-95% |
| CNN+LSTM | 10,000 | 45-60分钟 | 90-96% |

---

## 已知问题

无

---

## 下一步计划

- [ ] 添加预训练模型
- [ ] 支持多GPU训练
- [ ] 添加更多数据增强方法
- [ ] Web界面演示

---

**发布日期**: 2024-12-17
**版本**: v1.2
**许可证**: GNU GPL v3
