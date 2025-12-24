# 多GPU训练指南

## 概述

当前代码**已支持多GPU自动适配**！如果你有多张GPU，代码会自动检测并使用所有可用的GPU进行训练。

## 自动检测机制

### 当前实现

代码会自动：
1. ✅ **检测可用GPU数量**
2. ✅ **自动使用DataParallel进行多GPU训练**
3. ✅ **处理模型保存/加载的兼容性**
4. ✅ **优化评估过程**

### 工作原理

```python
# 在 Trainer 初始化时自动检测
trainer = Trainer(
    model=model,
    train_data=train_loader,
    eval_data=valid_loader,
    device='cuda',  # 只需指定 'cuda'，会自动检测多GPU
    use_multi_gpu=True  # 默认启用（可选）
)
```

**自动行为**：
- 如果检测到 **1张GPU**：使用单GPU训练
- 如果检测到 **多张GPU**：自动使用 `DataParallel` 进行多GPU训练
- 如果 **没有GPU**：自动回退到CPU（会显示警告）

## 使用示例

### 基本用法（自动检测）

```python
from dataset import ML1MDataset
from dataloader import TrainDataLoader, EvalDataLoader
from gpt2_recommender_enhanced import GPT2RecommenderEnhanced
from trainer import Trainer

# 加载数据
dataset = ML1MDataset('/path/to/ml-1m', split_ratio=[0.8, 0.1, 0.1])
train_data = dataset.get_split_data('train')
valid_data = dataset.get_split_data('validation')
test_data = dataset.get_split_data('test')

# 创建模型
model = GPT2RecommenderEnhanced(
    n_users=dataset.get_user_num(),
    n_items=dataset.get_item_num(),
    embed_dim=64,
    dataset=dataset
)

# 创建数据加载器
train_loader = TrainDataLoader(train_data, batch_size=2048, shuffle=True, device='cuda')
valid_loader = EvalDataLoader(valid_data, train_data, batch_size=2048, device='cuda')
test_loader = EvalDataLoader(test_data, train_data, batch_size=2048, device='cuda')

# 创建训练器（自动检测多GPU）
trainer = Trainer(
    model=model,
    train_data=train_loader,
    eval_data=valid_loader,
    test_data=test_loader,
    device='cuda',  # 只需指定 'cuda'，会自动使用所有可用GPU
    epochs=50,
    lr=1e-3
)

# 开始训练
trainer.fit(save_model=True, model_path='model.pth')
```

### 禁用多GPU（强制单GPU）

如果你只想使用单GPU，可以设置：

```python
trainer = Trainer(
    model=model,
    train_data=train_loader,
    eval_data=valid_loader,
    device='cuda',
    use_multi_gpu=False  # 禁用多GPU，强制使用单GPU
)
```

### 指定特定GPU

如果你想使用特定的GPU，可以在运行前设置：

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 只使用GPU 0和1

# 然后正常创建训练器
trainer = Trainer(..., device='cuda')
```

## 性能优化建议

### 1. 批次大小调整

使用多GPU时，**有效批次大小 = batch_size × GPU数量**

例如：
- 单GPU：`batch_size=2048`
- 4 GPU：`batch_size=512`（每个GPU 512，总共2048）

**建议**：
```python
# 根据GPU数量调整批次大小
num_gpus = torch.cuda.device_count()
batch_size = 2048 // num_gpus if num_gpus > 1 else 2048

train_loader = TrainDataLoader(train_data, batch_size=batch_size, ...)
```

### 2. 学习率调整

多GPU训练时，由于有效批次大小增加，可能需要调整学习率：

```python
# 线性缩放学习率
num_gpus = torch.cuda.device_count()
base_lr = 1e-3
lr = base_lr * num_gpus  # 线性缩放

trainer = Trainer(..., lr=lr)
```

### 3. 内存优化

多GPU训练时，每个GPU都会加载模型副本，注意：
- **缓存模式**（`use_cache=True`）：每个GPU都会缓存embeddings，内存占用较大
- **动态模式**（`use_cache=False`）：内存占用较小，但训练较慢

**建议**：
- 阶段1（快速训练）：使用缓存模式 + 多GPU
- 阶段2（精细调优）：使用动态模式 + 多GPU（如果内存足够）

## 技术细节

### DataParallel vs DistributedDataParallel

当前实现使用 **DataParallel (DP)**，适合：
- ✅ 单机多GPU
- ✅ 模型不是特别大
- ✅ 简单易用

如果需要更高性能，可以考虑使用 **DistributedDataParallel (DDP)**：
- 更高效的通信
- 更好的扩展性
- 但需要更复杂的设置

### 模型保存/加载兼容性

代码已自动处理：
- ✅ 多GPU保存的模型可以单GPU加载
- ✅ 单GPU保存的模型可以多GPU加载
- ✅ 自动处理 `module.` 前缀问题

### GPT-2 Encoder 处理

GPT-2 encoder 作为模型的一部分，会被 DataParallel 自动处理，无需特殊配置。

## 常见问题

### Q1: 如何查看是否使用了多GPU？

训练开始时会显示：
```
检测到 4 张GPU，将使用 DataParallel 进行多GPU训练
```

### Q2: 多GPU训练速度没有提升？

可能原因：
1. **批次大小太小**：每个GPU的批次太小，GPU利用率低
2. **数据传输瓶颈**：DataParallel需要将数据复制到各GPU
3. **模型太小**：小模型可能无法充分利用多GPU

**解决方案**：
- 增加批次大小
- 考虑使用 DistributedDataParallel
- 检查GPU利用率：`nvidia-smi`

### Q3: 内存不足怎么办？

**选项1**：减少批次大小
```python
batch_size = 1024  # 从2048减少到1024
```

**选项2**：使用动态模式（不缓存embeddings）
```python
model = GPT2RecommenderEnhanced(..., use_cache=False)
```

**选项3**：使用更少的GPU
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 只使用2张GPU
```

### Q4: 评估时是否使用多GPU？

评估时，代码会自动处理：
- 使用 `model.module` 进行单样本推荐（避免DataParallel的开销）
- 批量评估仍然可以利用多GPU（通过forward方法）

## 性能对比

### 预期加速比

| GPU数量 | 理论加速比 | 实际加速比（近似） |
|---------|-----------|------------------|
| 1       | 1.0x      | 1.0x            |
| 2       | 2.0x      | 1.8x - 1.9x     |
| 4       | 4.0x      | 3.2x - 3.6x     |
| 8       | 8.0x      | 5.5x - 6.5x     |

*注：实际加速比受数据传输、同步开销等因素影响*

## 总结

✅ **当前代码已支持多GPU自动适配**
✅ **无需修改代码，自动检测和使用**
✅ **兼容单GPU和多GPU场景**
✅ **自动处理模型保存/加载兼容性**

只需设置 `device='cuda'`，代码会自动使用所有可用的GPU！

