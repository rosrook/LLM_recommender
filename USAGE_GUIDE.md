# 项目使用指南 - 完整版

## 一、项目文件清单

### 核心代码文件（必需）

#### 1. 数据相关
- **`dataset.py`** - ML-1M数据集加载模块
  - `ML1MDataset` 类：加载和处理MovieLens-1M数据集
  - 功能：数据加载、分割、元数据访问

- **`dataloader.py`** - 数据加载器模块
  - `AbstractDataLoader` 抽象基类
  - `TrainDataLoader` 训练数据加载器（支持负采样）
  - `EvalDataLoader` 评估数据加载器

#### 2. 模型相关
- **`recommender.py`** - 推荐模型基类和基础实现
  - `AbstractRecommender` 抽象基类
  - `NCFRecommender` NCF基线模型
  - `GPT2Recommender` 基础GPT-2推荐模型（预计算版本）

- **`gpt2_encoder.py`** - GPT-2编码器模块 ⭐
  - `GPT2MetadataEncoder` GPT-2元数据编码器
  - `format_user_metadata()` 用户元数据格式化
  - `format_item_metadata()` 物品元数据格式化
  - 支持PEFT（LoRA）微调

- **`gpt2_recommender_enhanced.py`** - 增强版推荐模型 ⭐⭐⭐
  - `CrossAttentionFusion` Cross-Attention特征融合模块
  - `GPT2RecommenderEnhanced` 增强版推荐模型
  - 核心创新：动态metadata提取、Cross-Attention融合、对比学习

#### 3. 训练相关
- **`trainer.py`** - 模型训练器
  - `Trainer` 训练器类
  - 功能：训练、验证、测试、早停、模型保存

### 示例文件

- **`example_gpt2_recommender.py`** - 基础版使用示例
  - 演示如何使用 `GPT2Recommender`（预计算版本）

- **`example_enhanced.py`** - 增强版使用示例 ⭐
  - 演示三种训练策略：
    - 策略1：快速原型（缓存模式）
    - 策略2：端到端微调（推荐）
    - 策略3：混合模式

### 配置文件

- **`requirements.txt`** - Python依赖包列表
  - torch, transformers, peft, pandas, numpy, scipy, tqdm

### 文档文件

- **`PROJECT_SUMMARY.md`** - 项目总结文档（详细技术说明）
- **`IMPROVEMENTS.md`** - 改进说明文档
- **`README_ENHANCED.md`** - 增强版使用指南
- **`USAGE_GUIDE.md`** - 本文档（完整使用接口）

### 其他文件

- **`gpt2.py`** - GPT-2基础示例（参考文件，非必需）

---

## 二、快速开始

### 2.1 环境配置

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置HuggingFace镜像（可选，加速下载）
export HF_ENDPOINT=https://hf-mirror.com
```

### 2.2 数据准备

确保 `ml-1m` 数据集在正确路径：
```
ml-1m/
├── users.dat
├── movies.dat
└── ratings.dat
```

---

## 三、使用接口示例

### 3.1 基础版：GPT2Recommender（预计算模式）

```python
from dataset import ML1MDataset
from dataloader import TrainDataLoader, EvalDataLoader
from recommender import GPT2Recommender
from trainer import Trainer

# 1. 加载数据集
dataset = ML1MDataset('/path/to/ml-1m', split_ratio=[0.8, 0.1, 0.1])

# 2. 获取数据分割
train_data = dataset.get_split_data('train')
valid_data = dataset.get_split_data('validation')
test_data = dataset.get_split_data('test')

# 3. 创建模型（基础版）
model = GPT2Recommender(
    n_users=dataset.get_user_num(),
    n_items=dataset.get_item_num(),
    embed_dim=64,
    dataset=dataset,
    gpt2_model_name='gpt2',
    use_peft=True,  # 使用PEFT
    freeze_gpt2=True  # 冻结GPT-2（预计算模式）
)

# 4. 创建数据加载器
train_loader = TrainDataLoader(
    train_data, 
    batch_size=2048, 
    shuffle=True, 
    device='cuda'
)
valid_loader = EvalDataLoader(
    valid_data, 
    train_data, 
    batch_size=2048, 
    device='cuda'
)
test_loader = EvalDataLoader(
    test_data, 
    train_data, 
    batch_size=2048, 
    device='cuda'
)

# 5. 创建训练器并训练
trainer = Trainer(
    model=model,
    train_data=train_loader,
    eval_data=valid_loader,
    test_data=test_loader,
    device='cuda',
    epochs=50,
    lr=1e-3,
    weight_decay=1e-5,
    eval_step=1,
    early_stop_patience=10
)

# 6. 开始训练
valid_result, test_result = trainer.fit(
    save_model=True, 
    model_path='gpt2_recommender.pth'
)

print(f"Best Validation: {valid_result}")
print(f"Test Result: {test_result}")
```

---

### 3.2 增强版：GPT2RecommenderEnhanced（推荐使用）⭐⭐⭐

#### 策略1：快速原型（缓存模式）

```python
from gpt2_recommender_enhanced import GPT2RecommenderEnhanced

# 创建模型（快速模式）
model = GPT2RecommenderEnhanced(
    n_users=dataset.get_user_num(),
    n_items=dataset.get_item_num(),
    embed_dim=64,
    dataset=dataset,
    use_cache=True,        # 使用缓存（快速）
    use_attention=False,   # 简单拼接
    freeze_gpt2=True,      # 冻结GPT-2
    use_peft=False
)

# 训练（快速）
trainer = Trainer(
    model=model,
    train_data=train_loader,
    eval_data=valid_loader,
    test_data=test_loader,
    device='cuda',
    epochs=20,
    lr=1e-3,
    early_stop_patience=5
)
trainer.fit(save_model=True, model_path='enhanced_fast.pth')
```

#### 策略2：端到端微调（推荐）⭐⭐⭐

```python
# 创建模型（端到端微调模式）
model = GPT2RecommenderEnhanced(
    n_users=dataset.get_user_num(),
    n_items=dataset.get_item_num(),
    embed_dim=64,
    dataset=dataset,
    use_cache=False,       # 动态提取（端到端）
    use_attention=True,     # Cross-Attention融合
    freeze_gpt2=False,      # 微调GPT-2
    use_peft=True,         # 使用PEFT
    contrastive_weight=0.1  # 对比学习权重
)

# 训练（端到端微调）
trainer = Trainer(
    model=model,
    train_data=train_loader,
    eval_data=valid_loader,
    test_data=test_loader,
    device='cuda',
    epochs=50,
    lr=5e-4,              # 较低学习率
    weight_decay=1e-5,
    early_stop_patience=10
)
trainer.fit(save_model=True, model_path='enhanced_end2end.pth')
```

#### 策略3：混合模式（两阶段训练）

```python
# 阶段1：快速训练（缓存模式）
model = GPT2RecommenderEnhanced(
    n_users=dataset.get_user_num(),
    n_items=dataset.get_item_num(),
    embed_dim=64,
    dataset=dataset,
    use_cache=True,        # 缓存模式
    use_attention=True,
    freeze_gpt2=True
)

trainer = Trainer(model=model, ..., epochs=15)
trainer.fit(save_model=False)

# 阶段2：精细调优（动态模式）
model.use_cache = False    # 切换到动态模式
model.freeze_gpt2 = False  # 解冻GPT-2

# 重新启用梯度
for param in model.gpt2_encoder.parameters():
    param.requires_grad = True

trainer = Trainer(model=model, ..., epochs=30, lr=5e-4)
trainer.fit(save_model=True, model_path='enhanced_hybrid.pth')
```

---

### 3.3 完整示例：直接运行

#### 方式1：运行增强版示例（推荐）

```bash
# 策略1：快速原型
python example_enhanced.py 1

# 策略2：端到端微调（推荐）
python example_enhanced.py 2

# 策略3：混合模式
python example_enhanced.py 3
```

#### 方式2：运行基础版示例

```bash
python example_gpt2_recommender.py
```

---

## 四、核心接口说明

### 4.1 GPT2RecommenderEnhanced 参数说明

```python
GPT2RecommenderEnhanced(
    n_users: int,              # 用户数量
    n_items: int,              # 物品数量
    embed_dim: int,            # ID embedding维度（推荐64）
    dataset: ML1MDataset,       # 数据集对象
    gpt2_model_name: str = 'gpt2',  # GPT-2模型名称
    use_peft: bool = True,     # 是否使用PEFT（推荐True）
    metadata_dim: int = None,  # Metadata维度（默认768）
    freeze_gpt2: bool = False,  # 是否冻结GPT-2
    use_cache: bool = False,   # 是否使用缓存（False=动态提取）
    use_attention: bool = True,  # 是否使用Cross-Attention
    contrastive_weight: float = 0.1  # 对比学习权重
)
```

**关键参数选择**：

| 场景 | use_cache | use_attention | freeze_gpt2 | use_peft |
|------|-----------|--------------|-------------|----------|
| 快速原型 | True | False | True | False |
| 端到端微调 | False | True | False | True |
| 特征提取器 | True | True | True | False |

### 4.2 Trainer 参数说明

```python
Trainer(
    model: nn.Module,           # 推荐模型
    train_data: TrainDataLoader,  # 训练数据加载器
    eval_data: EvalDataLoader,    # 验证数据加载器
    test_data: EvalDataLoader = None,  # 测试数据加载器
    device: str = 'cuda',      # 设备
    epochs: int = 300,         # 训练轮数
    batch_size: int = 2048,    # 批次大小（已废弃，在DataLoader中设置）
    optimizer: str = 'adam',   # 优化器（'adam'或'sgd'）
    lr: float = 1e-3,          # 学习率
    weight_decay: float = 0,   # 权重衰减
    eval_step: int = 1,        # 每N个epoch评估一次
    early_stop_patience: int = 10  # 早停耐心值
)
```

### 4.3 模型方法接口

```python
# 1. 前向传播
pos_scores, neg_scores = model(batch_data)
# batch_data: torch.Tensor [users, pos_items, neg_items]

# 2. 计算损失
loss = model.calculate_loss(pos_scores, neg_scores)

# 3. 预测得分
scores = model.predict(user_ids, item_ids)
# user_ids: torch.LongTensor
# item_ids: torch.LongTensor
# 返回: torch.FloatTensor (预测得分)

# 4. 推荐Top-K
recommendations = model.recommend(user_id, k=10)
# user_id: int
# k: int (推荐数量)
# 返回: torch.LongTensor (推荐的物品ID)
```

---

## 五、完整使用流程示例

### 5.1 最小示例（端到端微调）

```python
#!/usr/bin/env python
# coding: utf-8

from dataset import ML1MDataset
from dataloader import TrainDataLoader, EvalDataLoader
from gpt2_recommender_enhanced import GPT2RecommenderEnhanced
from trainer import Trainer

# Step 1: 加载数据
print("Loading dataset...")
dataset = ML1MDataset('/Users/zhuxuzhou/Downloads/ml-1m', split_ratio=[0.8, 0.1, 0.1])
train_data = dataset.get_split_data('train')
valid_data = dataset.get_split_data('validation')
test_data = dataset.get_split_data('test')

# Step 2: 创建模型
print("Creating model...")
model = GPT2RecommenderEnhanced(
    n_users=dataset.get_user_num(),
    n_items=dataset.get_item_num(),
    embed_dim=64,
    dataset=dataset,
    use_cache=False,      # 动态提取
    use_attention=True,  # Cross-Attention
    freeze_gpt2=False,   # 微调GPT-2
    use_peft=True        # PEFT
)

# Step 3: 创建数据加载器
print("Creating data loaders...")
train_loader = TrainDataLoader(train_data, batch_size=1024, shuffle=True, device='cuda')
valid_loader = EvalDataLoader(valid_data, train_data, batch_size=1024, device='cuda')
test_loader = EvalDataLoader(test_data, train_data, batch_size=1024, device='cuda')

# Step 4: 创建训练器
print("Creating trainer...")
trainer = Trainer(
    model=model,
    train_data=train_loader,
    eval_data=valid_loader,
    test_data=test_loader,
    device='cuda',
    epochs=50,
    lr=5e-4,
    weight_decay=1e-5,
    early_stop_patience=10
)

# Step 5: 训练
print("Starting training...")
valid_result, test_result = trainer.fit(
    save_model=True,
    model_path='best_model.pth'
)

# Step 6: 输出结果
print("\n" + "="*50)
print("Training Completed!")
print("="*50)
print(f"\nBest Validation Results:")
for metric, value in valid_result.items():
    print(f"  {metric}: {value:.4f}")

print(f"\nTest Results:")
for metric, value in test_result.items():
    print(f"  {metric}: {value:.4f}")
```

### 5.2 推理示例

```python
import torch
from gpt2_recommender_enhanced import GPT2RecommenderEnhanced
from dataset import ML1MDataset

# 加载模型
dataset = ML1MDataset('/path/to/ml-1m')
model = GPT2RecommenderEnhanced(
    n_users=dataset.get_user_num(),
    n_items=dataset.get_item_num(),
    embed_dim=64,
    dataset=dataset
)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 为用户推荐Top-10
user_id = 1
recommendations = model.recommend(user_id, k=10)
print(f"Top-10 recommendations for user {user_id}: {recommendations}")

# 预测特定用户-物品对的得分
user_ids = torch.LongTensor([1, 2, 3])
item_ids = torch.LongTensor([10, 20, 30])
scores = model.predict(user_ids, item_ids)
print(f"Prediction scores: {scores}")
```

---

## 六、文件依赖关系

```
dataset.py (数据加载)
    ↓
dataloader.py (数据加载器)
    ↓
gpt2_encoder.py (GPT-2编码器)
    ↓
recommender.py (基础模型)
    ↓
gpt2_recommender_enhanced.py (增强模型) ⭐
    ↓
trainer.py (训练器)
    ↓
example_*.py (示例脚本)
```

---

## 七、常见问题

### Q1: 如何选择使用基础版还是增强版？
- **基础版** (`GPT2Recommender`): 快速验证，预计算模式
- **增强版** (`GPT2RecommenderEnhanced`): 追求性能，端到端学习 ⭐推荐

### Q2: 动态模式 vs 缓存模式？
- **缓存模式** (`use_cache=True`): 训练快，但GPT-2无法微调
- **动态模式** (`use_cache=False`): 训练慢，但支持端到端学习 ⭐推荐

### Q3: GPU内存不足怎么办？
- 减小 `batch_size`（如1024或512）
- 使用缓存模式 (`use_cache=True`)
- 冻结GPT-2 (`freeze_gpt2=True`)

### Q4: 如何加速训练？
- 使用缓存模式
- 增大 `batch_size`
- 使用PEFT而不是全量微调

---

## 八、项目文件总结

### 必需文件（核心功能）
1. `dataset.py` - 数据集加载
2. `dataloader.py` - 数据加载器
3. `gpt2_encoder.py` - GPT-2编码器 ⭐
4. `recommender.py` - 基础模型
5. `gpt2_recommender_enhanced.py` - 增强模型 ⭐⭐⭐
6. `trainer.py` - 训练器

### 推荐文件（示例和文档）
7. `example_enhanced.py` - 增强版示例 ⭐
8. `requirements.txt` - 依赖列表
9. `USAGE_GUIDE.md` - 本文档

### 可选文件
10. `example_gpt2_recommender.py` - 基础版示例
11. `PROJECT_SUMMARY.md` - 项目总结
12. `IMPROVEMENTS.md` - 改进说明
13. `README_ENHANCED.md` - 增强版指南

---

## 九、快速命令参考

```bash
# 安装依赖
pip install -r requirements.txt

# 运行增强版（端到端微调，推荐）
python example_enhanced.py 2

# 运行增强版（快速原型）
python example_enhanced.py 1

# 运行增强版（混合模式）
python example_enhanced.py 3

# 运行基础版
python example_gpt2_recommender.py
```

---

## 十、推荐配置

### 最佳性能配置（推荐）⭐⭐⭐
```python
model = GPT2RecommenderEnhanced(
    embed_dim=64,
    use_cache=False,      # 动态提取
    use_attention=True,   # Cross-Attention
    freeze_gpt2=False,   # 微调GPT-2
    use_peft=True,       # PEFT
    contrastive_weight=0.1
)
trainer = Trainer(epochs=50, lr=5e-4, batch_size=1024)
```

### 快速验证配置
```python
model = GPT2RecommenderEnhanced(
    embed_dim=64,
    use_cache=True,      # 缓存模式
    use_attention=False, # 简单拼接
    freeze_gpt2=True    # 冻结GPT-2
)
trainer = Trainer(epochs=20, lr=1e-3, batch_size=2048)
```

