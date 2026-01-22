# 基于Pytorch的手写数字识别项目

这是一个使用CNN网络识别MINST手写数字集的深度学习项目，该项目实现了从模型定义，训练到验证的完整过程，并使用Tensorboard进行数据可视化。

## 项目特点：
- **高准确率：** 模型在MINST数据集上的准确率达到99.2%
- **可视化：** 集成Tensorboard以监测训练损失和测试准确率
- **结构清晰：** 模型定义，训练，验证逻辑清晰，易于阅读与二次开发

## 环境配置:
- Python 3.11.14
- Pytorch: 2.9.1 + CUDA 12.8
- Torchvision
- Tensorboard

## 文件结构说明：
- `Num_Model.py`: 神经网络模型架构
- `Num_Recognition.py`: 模型训练脚本
- `Num_Validation.py`: 测试与验证脚本
- `NumRecog_method1.pth`: 训练好的模型权重文件‘
- `NumRec_log/`: 训练过程的Tensorboard日志文件
- `MINST/`： 本地存储的MINST数据集

## 使用指南:
1. 训练模型: `python Num_Recognition.py`
2. 验证模型: `python Num_Validation.py`
3. 查看训练曲线: `tensorboard --logdir="NUmRec_log"`

## 实验结果:
| 指标 | 数值 |
| 准确率 | 99.2% |
| 平均损失 | 0.00043 |


