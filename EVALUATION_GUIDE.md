# 标准评估流程使用指南

## 概述

本指南说明如何在训练结束后，按照标准评估流程在测试集上评估模型。

## 评估标准

按照您的要求，评估流程如下：

1. **对每个用户，找到未交互过的item**：排除训练集+验证集中已交互的item
2. **计算user/item预测分数**：使用模型的`predict`方法（包含完整的embedding融合和MLP预测）
3. **排序得到top K推荐**：根据分数从大到小排序
4. **计算评价指标**：Recall@K, Precision@K, NDCG@K

## 使用方法

### 方法1：在训练脚本中自动评估（推荐）

`run.py` 已经集成了标准评估流程，训练结束后会自动执行：

```bash
python run.py
```

训练完成后，会自动：
1. 加载最佳模型
2. 执行标准评估
3. 显示评估结果
4. 保存结果到 `test_results_standard.json`

### 方法2：单独运行评估

如果您已经训练好模型，可以单独运行评估：

```python
import torch
from dataset import ML1MDataset
from gpt2_recommender_enhanced import GPT2RecommenderEnhanced
from evaluator import evaluate_model_standard

# 1. 加载数据集
dataset = ML1MDataset('/path/to/ml-1m', split_ratio=[0.8, 0.1, 0.1])
train_data = dataset.get_split_data('train')
valid_data = dataset.get_split_data('validation')
test_data = dataset.get_split_data('test')

# 2. 创建模型（与训练时相同的配置）
model = GPT2RecommenderEnhanced(
    n_users=dataset.get_user_num(),
    n_items=dataset.get_item_num(),
    embed_dim=64,
    dataset=dataset,
    use_cache=True,
    use_attention=True,
    freeze_gpt2=False
)

# 3. 加载训练好的模型权重
model.load_state_dict(torch.load('gpt2_recommender_enhanced.pth'))
model.eval()

# 4. 执行标准评估
results = evaluate_model_standard(
    model=model,
    dataset=dataset,
    train_data=train_data,
    valid_data=valid_data,
    test_data=test_data,
    device='cuda',
    k_list=[10, 20, 50],  # 评估Top-10, Top-20, Top-50
    verbose=True
)

# 5. 查看结果
print(results)
```

### 方法3：使用评估器类（更灵活）

如果需要更细粒度的控制：

```python
from evaluator import StandardEvaluator

# 创建评估器
evaluator = StandardEvaluator(
    model=model,
    dataset=dataset,
    train_data=train_data,
    valid_data=valid_data,
    test_data=test_data,
    device='cuda',
    k_list=[10, 20, 50]
)

# 评估所有用户
results = evaluator.evaluate(verbose=True)

# 评估单个用户
user_metrics = evaluator.evaluate_user(user_id=0, k_list=[10, 20])

# 打印结果
evaluator.print_results(results)
```

## 评估流程详解

### 1. 构建用户交互历史

```python
# 合并训练集和验证集
train_valid_data = pd.concat([train_data, valid_data])

# 为每个用户记录交互过的item
user_interacted_items[user_id] = {item1, item2, ...}
```

### 2. 获取候选item

```python
# 对每个用户，找到未交互过的item
all_items = set(range(n_items))
candidate_items = all_items - user_interacted_items[user_id]
```

### 3. 计算预测分数

```python
# 使用模型的predict方法计算分数
scores = model.predict(user_ids, candidate_item_ids)
```

**注意**：虽然您提到"user/item embedding做内积"，但当前模型使用的是：
- ID embeddings + Metadata embeddings
- Cross-Attention融合（如果启用）
- MLP预测层

`predict`方法已经包含了完整的预测流程，比简单内积更准确。

### 4. 排序和Top-K推荐

```python
# 根据分数排序
_, top_indices = torch.topk(scores, k=K)
top_items = candidate_items[top_indices]
```

### 5. 计算指标

#### Recall@K
```
Recall@K = (推荐的K个item中正样本数量) / (测试集中正样本总数)
```

#### Precision@K
```
Precision@K = (推荐的K个item中正样本数量) / K
```

#### NDCG@K
```
NDCG@K = DCG@K / IDCG@K

其中：
- DCG@K = Σ(rel_i / log2(i+2))，i从1到K
- IDCG@K = 理想情况下的DCG（所有正样本都在前K位）
```

## 输出示例

```
============================================================
测试集评估结果（标准评估流程）
============================================================

Top-10 推荐结果:
  Recall@10:    0.1234
  Precision@10: 0.0567
  NDCG@10:      0.0890

Top-20 推荐结果:
  Recall@20:    0.2345
  Precision@20: 0.0456
  NDCG@20:      0.1234

Top-50 推荐结果:
  Recall@50:    0.3456
  Precision@50: 0.0345
  NDCG@50:      0.1567

============================================================
```

## 评估结果保存

评估结果会自动保存为JSON格式：

```json
{
  "Recall@10": 0.1234,
  "Precision@10": 0.0567,
  "NDCG@10": 0.0890,
  "Recall@20": 0.2345,
  "Precision@20": 0.0456,
  "NDCG@20": 0.1234,
  "Recall@50": 0.3456,
  "Precision@50": 0.0345,
  "NDCG@50": 0.1567
}
```

## 注意事项

### 1. 候选item的排除

- ✅ **排除**：训练集+验证集中交互过的item
- ✅ **包含**：测试集中交互过的item（作为正样本）
- ✅ **包含**：从未交互过的item（作为负样本）

### 2. 预测方法

虽然您提到"embedding内积"，但实际使用的是模型的`predict`方法，它包含：
- ID embeddings
- Metadata embeddings（GPT-2编码）
- Cross-Attention融合（如果启用）
- MLP预测层

这比简单内积更准确，因为：
- 利用了metadata信息
- 考虑了特征交互
- 经过端到端训练优化

### 3. 性能优化

评估过程会自动：
- 批量计算预测分数（避免显存不足）
- 使用GPU加速
- 显示进度条

### 4. 多GPU支持

如果模型使用DataParallel，评估器会自动处理：
- 访问`model.module`获取实际模型
- 在正确的设备上计算

## 常见问题

### Q1: 为什么使用predict而不是简单内积？

A: 当前模型架构包含：
- Metadata embeddings（GPT-2编码）
- Cross-Attention融合
- MLP预测层

使用`predict`方法可以充分利用这些特性，比简单内积更准确。

### Q2: 如何只使用ID embedding的内积？

A: 如果您确实需要简单内积，可以修改评估器：

```python
# 在evaluator.py中，修改predict_scores_for_user方法
def predict_scores_for_user(self, user_id, candidate_items, batch_size=1024):
    # 获取ID embeddings
    user_embed = self.model_for_eval.user_embedding(
        torch.tensor([user_id], device=self.device)
    )
    item_embeds = self.model_for_eval.item_embedding(candidate_items)
    
    # 简单内积
    scores = (user_embed * item_embeds).sum(dim=-1)
    return scores.squeeze(0)
```

### Q3: 评估需要多长时间？

A: 取决于：
- 用户数量（测试集中的用户数）
- Item数量（候选item数量）
- GPU性能

对于ML-1M数据集（~6000用户，~4000物品），通常需要几分钟。

### Q4: 如何评估特定用户？

A: 使用`evaluate_user`方法：

```python
evaluator = StandardEvaluator(...)
metrics = evaluator.evaluate_user(user_id=100, k_list=[10, 20])
print(metrics)
```

## 总结

标准评估流程已经集成到`run.py`中，训练完成后会自动执行。评估结果会：
- ✅ 显示在控制台
- ✅ 保存到`test_results_standard.json`
- ✅ 包含Recall, Precision, NDCG指标
- ✅ 支持多个K值（10, 20, 50）

只需运行`python run.py`，训练完成后即可看到标准评估结果！

