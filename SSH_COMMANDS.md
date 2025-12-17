# SSH终端执行命令 - 快速参考

## 立即执行（复制粘贴到SSH终端）

```bash
# 1. 运行部署脚本
cd ~
chmod +x deploy_and_setup.sh
./deploy_and_setup.sh

# 2. 进入项目目录
cd ~/ppg_project

# 3. 提交数据生成作业
sbatch slurm_datagen.sh

# 4. 查看作业状态
squeue -u $USER

# 5. 查看数据生成日志（实时）
tail -f logs/datagen_*.out
# 按 Ctrl+C 退出

# 6. 等待数据生成完成后，提交训练作业
sbatch slurm_train.sh

# 7. 查看训练日志（实时）
tail -f logs/train_*.out
# 按 Ctrl+C 退出
```

## 常用命令

```bash
# 查看作业状态
squeue -u $USER

# 查看所有日志文件
ls -lh logs/

# 取消作业
scancel <job_id>

# 检查数据集
ls -lh ~/ppg_training_data/ | head -20

# 检查模型
ls -lh checkpoints_resnet_waveform/
```

## 完成后

邮件通知会发送到: **wenzheng.wang@lip6.fr**

您可以：
- ✅ 关闭本地Mac
- ✅ 断开网络
- ✅ 等待邮件通知

预计总时间: **3-4小时**
