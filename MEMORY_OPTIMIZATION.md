# 多GPU显存优化指南

## 当前多GPU配置的显存分析

### DataParallel 的显存使用

使用 DataParallel 时，每个GPU的显存占用包括：

1. **模型参数**（每个GPU都有一份完整副本）：
   - GPT-2模型：~500MB（如果使用PEFT，会减少）
   - Embedding层：~10-50MB（取决于用户/物品数量）
   - MLP层：~5-20MB
   - **总计：~500-600MB/GPU**

2. **缓存模式下的Metadata Embeddings**（每个GPU都有一份）：
   - User embeddings: `n_users × embed_dim × 4 bytes`
   - Item embeddings: `n_items × embed_dim × 4 bytes`
   - ML-1M数据集：~6000 users × 64 × 4 = ~1.5MB
   - ML-1M数据集：~4000 items × 64 × 4 = ~1MB
   - **总计：~2-3MB/GPU**

3. **训练时的激活值（Activations）**：
   - Batch数据：`batch_size × 3 × 4 bytes`（users, pos_items, neg_items）
   - Forward activations：取决于batch size和模型大小
   - **每个GPU处理：`batch_size / num_gpus` 的数据**

### 显存占用估算

假设使用8张GPU，batch_size=2048：

- **模型参数**：~600MB/GPU
- **缓存embeddings**：~3MB/GPU
- **每个GPU的batch**：2048 / 8 = 256 samples
- **激活值**：~100-200MB/GPU（取决于模型复杂度）
- **梯度**：~600MB/GPU（与参数相同）
- **优化器状态**：~1200MB/GPU（Adam需要2倍参数空间）

**总显存需求**：~2.5-3GB/GPU（缓存模式）

## 潜在问题

### ❌ 当前配置的问题

1. **没有自动调整batch size**：
   - 如果batch_size太大，每个GPU可能显存不足
   - 需要手动根据GPU数量调整

2. **缓存模式显存占用大**：
   - 每个GPU都会缓存metadata embeddings
   - 如果显存有限，可能导致OOM

3. **没有显存监控**：
   - 无法提前发现显存问题
   - 错误发生时已经太晚

4. **没有错误恢复机制**：
   - OOM错误会导致训练中断
   - 没有自动降级方案

## ✅ 优化方案

### 方案1：自动调整Batch Size（推荐）

```python
import torch

def get_optimal_batch_size(num_gpus, base_batch_size=2048):
    """
    根据GPU数量自动调整batch size
    
    Args:
        num_gpus: GPU数量
        base_batch_size: 单GPU的基础batch size
    
    Returns:
        调整后的batch size（每个GPU）
    """
    if num_gpus > 1:
        # 多GPU时，每个GPU的batch size = 总batch size / GPU数量
        per_gpu_batch = base_batch_size // num_gpus
        # 确保至少为1
        per_gpu_batch = max(1, per_gpu_batch)
        return per_gpu_batch
    return base_batch_size

# 使用示例
num_gpus = torch.cuda.device_count()
batch_size = get_optimal_batch_size(num_gpus, base_batch_size=2048)
print(f"使用 {num_gpus} 张GPU，每个GPU的batch size: {batch_size}")
```

### 方案2：使用动态模式（减少显存）

```python
# 阶段1：使用缓存模式（快速但显存占用大）
model = GPT2RecommenderEnhanced(
    ...,
    use_cache=True,  # 缓存模式
    freeze_gpt2=True
)

# 阶段2：切换到动态模式（慢但显存占用小）
model.use_cache = False  # 动态模式，不缓存embeddings
```

### 方案3：梯度累积（处理大batch）

如果显存不足，可以使用梯度累积来模拟更大的batch size：

```python
# 在trainer中添加gradient_accumulation_steps参数
trainer = Trainer(
    ...,
    gradient_accumulation_steps=4  # 累积4个batch的梯度
)
```

### 方案4：使用更少的GPU

如果显存仍然不足，可以限制使用的GPU数量：

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # 只使用4张GPU
```

## 推荐的显存优化配置

### 配置1：显存充足（>8GB/GPU）

```python
batch_size = 2048  # 总batch size
use_cache = True   # 使用缓存模式
num_gpus = 8       # 使用所有GPU
```

### 配置2：显存中等（4-8GB/GPU）

```python
batch_size = 1024  # 减少batch size
use_cache = True   # 仍可使用缓存
num_gpus = 4       # 使用部分GPU
```

### 配置3：显存有限（<4GB/GPU）

```python
batch_size = 512   # 小batch size
use_cache = False  # 使用动态模式
num_gpus = 2       # 使用少量GPU
```

## 显存监控和错误处理

建议添加显存监控代码：

```python
import torch

def check_gpu_memory():
    """检查GPU显存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            print(f"GPU {i}: {allocated:.2f}GB / {reserved:.2f}GB / {total:.2f}GB")

# 在训练前检查
check_gpu_memory()
```

## 总结

### ✅ DataParallel 的优势

1. **自动分割batch**：每个GPU处理 `batch_size / num_gpus` 的数据
2. **减少单GPU显存压力**：相比单GPU，每个GPU的显存需求减少
3. **简单易用**：无需修改代码逻辑

### ⚠️ 需要注意的问题

1. **模型复制**：每个GPU都有完整的模型副本
2. **缓存embeddings**：如果使用缓存模式，每个GPU都会缓存
3. **Batch size调整**：需要根据GPU数量手动调整

### 💡 最佳实践

1. **根据GPU显存调整batch size**
2. **显存不足时使用动态模式**
3. **监控显存使用情况**
4. **准备错误恢复机制**

