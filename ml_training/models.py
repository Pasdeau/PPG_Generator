#!/usr/bin/env python3
"""
PPG信号分类 - 模型定义
支持多种深度学习模型架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D_Classifier(nn.Module):
    """
    1D CNN分类器 - 推荐新手使用
    
    适用于:
    - 波形分类 (5类)
    - 伪影分类 (5类)
    - 心律分类 (2类)
    """
    
    def __init__(self, input_length=8000, num_classes=5):
        """
        Parameters:
        -----------
        input_length : int
            输入信号长度 (采样点数)
        num_classes : int
            分类类别数
        """
        super(CNN1D_Classifier, self).__init__()
        
        # 卷积层1: 提取低级特征
        self.conv1 = nn.Conv1d(
            in_channels=1,      # 单通道PPG信号
            out_channels=32,    # 32个特征图
            kernel_size=50,     # 50ms窗口 (假设1000Hz采样)
            stride=2,
            padding=25
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # 卷积层2: 提取中级特征
        self.conv2 = nn.Conv1d(32, 64, kernel_size=25, stride=2, padding=12)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # 卷积层3: 提取高级特征
        self.conv3 = nn.Conv1d(64, 128, kernel_size=10, stride=1, padding=5)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # 计算全连接层输入维度
        # input_length -> conv1 -> pool1 -> conv2 -> pool2 -> conv3 -> pool3
        fc_input_dim = self._get_fc_input_dim(input_length)
        
        # 全连接层
        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_classes)
    
    def _get_fc_input_dim(self, input_length):
        """计算全连接层输入维度"""
        # 模拟前向传播计算维度
        x = torch.zeros(1, 1, input_length)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        """
        前向传播
        
        Parameters:
        -----------
        x : torch.Tensor
            输入信号 [batch_size, 1, signal_length]
        
        Returns:
        --------
        torch.Tensor
            分类logits [batch_size, num_classes]
        """
        # 卷积块1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 卷积块2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 卷积块3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


class LSTM_Classifier(nn.Module):
    """
    LSTM分类器 - 适合时序建模
    
    适用于:
    - 心律分类 (长期模式)
    - 复杂时序特征
    """
    
    def __init__(self, input_length=8000, num_classes=5, hidden_size=128, num_layers=2):
        super(LSTM_Classifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=1,           # 每个时间步的特征维度
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True      # 双向LSTM
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size * 2, 128)  # *2因为双向
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """
        Parameters:
        -----------
        x : torch.Tensor
            [batch_size, 1, signal_length]
        """
        # 转换为LSTM输入格式 [batch, seq_len, features]
        x = x.transpose(1, 2)  # [batch, signal_length, 1]
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        # h_n: [num_layers*2, batch, hidden_size]
        # 取最后一层的前向和后向隐藏状态
        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # 全连接层
        x = F.relu(self.fc1(hidden))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class CNN_LSTM_Classifier(nn.Module):
    """
    CNN + LSTM 混合模型 - 推荐进阶使用
    
    CNN提取局部特征，LSTM建模时序依赖
    """
    
    def __init__(self, input_length=8000, num_classes=5):
        super(CNN_LSTM_Classifier, self).__init__()
        
        # CNN特征提取
        self.conv1 = nn.Conv1d(1, 64, kernel_size=50, stride=2, padding=25)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=25, stride=2, padding=12)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        
        # LSTM时序建模
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # 分类头
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # CNN特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # 转换为LSTM输入 [batch, seq_len, features]
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后的隐藏状态
        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # 分类
        x = F.relu(self.fc1(hidden))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ResNet1D_Block(nn.Module):
    """1D ResNet残差块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super(ResNet1D_Block, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


class ResNet1D_Classifier(nn.Module):
    """
    ResNet1D分类器 - 深层网络
    
    适用于大规模数据集
    """
    
    def __init__(self, input_length=8000, num_classes=5, dropout=0.5):
        super(ResNet1D_Classifier, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(3, stride=2, padding=1)
        
        # 残差块
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)  # 添加dropout防止过拟合
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResNet1D_Block(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResNet1D_Block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)  # 应用dropout
        x = self.fc(x)
        
        return x


def create_model(model_type='cnn', input_length=8000, num_classes=5, **kwargs):
    """
    模型工厂函数
    
    Parameters:
    -----------
    model_type : str
        'cnn', 'lstm', 'cnn_lstm', 'resnet'
    input_length : int
        输入信号长度
    num_classes : int
        分类类别数
    
    Returns:
    --------
    nn.Module
        PyTorch模型
    """
    models = {
        'cnn': CNN1D_Classifier,
        'lstm': LSTM_Classifier,
        'cnn_lstm': CNN_LSTM_Classifier,
        'resnet': ResNet1D_Classifier
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    model = models[model_type](input_length=input_length, num_classes=num_classes, **kwargs)
    return model


if __name__ == '__main__':
    # 测试模型
    print("=" * 70)
    print("PPG分类模型测试")
    print("=" * 70)
    
    batch_size = 4
    signal_length = 8000
    num_classes = 5
    
    # 创建测试输入
    x = torch.randn(batch_size, 1, signal_length)
    
    models_to_test = ['cnn', 'lstm', 'cnn_lstm', 'resnet']
    
    for model_type in models_to_test:
        print(f"\n测试 {model_type.upper()} 模型:")
        model = create_model(model_type, signal_length, num_classes)
        
        # 前向传播
        output = model(x)
        
        # 统计参数量
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  参数量: {num_params:,}")
        print(f"  ✓ 测试通过")
    
    print("\n" + "=" * 70)
    print("所有模型测试完成！")
    print("=" * 70)
