# GPT-2 增强推荐系统

## 项目概述

本项目实现了基于GPT-2大语言模型的推荐系统，提供了两个版本：

1. **GPT2Recommender**（基础版）：预计算metadata embeddings，简单特征拼接
2. **GPT2RecommenderEnhanced**（增强版）：动态metadata提取，Cross-Attention融合，端到端微调

## 核心创新点

### 1. 动态Metadata提取（端到端训练）
- **问题**：预计算方式限制了GPT-2的端到端学习能力
- **解决方案**：在训练过程中动态提取metadata embeddings
- **优势**：GPT-2可以根据推荐任务进行端到端微调，充分利用PEFT

### 2. Cross-Attention特征融合
- **问题**：简单拼接无法捕捉ID embedding和metadata embedding的交互关系
- **解决方案**：使用Cross-Attention机制自适应融合特征
- **优势**：自适应学习特征重要性，捕捉复杂交互模式

### 3. 对比学习增强
- **问题**：缺乏对语义相似性的显式建模
- **解决方案**：引入对比学习损失
- **优势**：利用LLM的语义理解能力，提升模型泛化能力

### 4. 灵活的缓存机制
- **优势**：支持缓存模式和动态模式切换
- **应用**：训练初期使用缓存快速迭代，后期切换到动态模式精细调优

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 基础版使用
```python
from recommender import GPT2Recommender
from dataset import ML1MDataset
from dataloader import TrainDataLoader, EvalDataLoader
from trainer import Trainer

dataset = ML1MDataset('ml-1m')
model = GPT2Recommender(
    n_users=dataset.get_user_num(),
    n_items=dataset.get_item_num(),
    embed_dim=64,
    dataset=dataset
)
# ... 训练代码
```

### 增强版使用（推荐）
```python
from gpt2_recommender_enhanced import GPT2RecommenderEnhanced

# 端到端微调模式（推荐）
model = GPT2RecommenderEnhanced(
    n_users=dataset.get_user_num(),
    n_items=dataset.get_item_num(),
    embed_dim=64,
    dataset=dataset,
    use_cache=False,  # 动态提取
    use_attention=True,  # Cross-Attention融合
    freeze_gpt2=False,  # 微调GPT-2
    use_peft=True  # 使用PEFT
)
```

### 运行示例
```bash
# 策略1：快速原型
python example_enhanced.py 1

# 策略2：端到端微调（推荐）
python example_enhanced.py 2

# 策略3：混合模式
python example_enhanced.py 3
```

## 项目结构

```
NFC/
├── dataset.py                    # 数据集加载
├── dataloader.py                 # 数据加载器
├── recommender.py                # 基础推荐模型（NCF + GPT2Recommender）
├── gpt2_encoder.py              # GPT-2编码器模块
├── gpt2_recommender_enhanced.py # 增强版推荐模型
├── trainer.py                    # 训练器
├── example_gpt2_recommender.py  # 基础版示例
├── example_enhanced.py          # 增强版示例
├── IMPROVEMENTS.md              # 改进说明文档
└── requirements.txt             # 依赖包
```

## 技术对比

| 特性 | 基础版 | 增强版 |
|------|--------|--------|
| Metadata提取 | 预计算 | 动态提取 |
| 特征融合 | 拼接 | Cross-Attention |
| LLM微调 | 有限 | 完整端到端 |
| 训练速度 | 快 | 可配置 |
| 模型性能 | 基础 | 提升 |

## 项目定位建议

### 题目1：大语言模型特征增强的推荐算法（不微调）
- 使用 `use_cache=True, freeze_gpt2=True`
- 重点：LLM作为强大的特征提取器
- 创新点：Cross-Attention特征融合机制

### 题目2：基于大语言模型微调的推荐算法（微调）
- 使用 `use_cache=False, freeze_gpt2=False, use_peft=True`
- 重点：端到端微调LLM适应推荐任务
- 创新点：动态metadata提取 + 端到端学习

### 题目3：融合大语言模型语义理解的推荐系统（推荐）
- 结合两种模式的优势
- 重点：LLM语义理解 + 创新特征融合 + 端到端学习
- 创新点：Cross-Attention + 动态提取 + 对比学习

## 预期效果

- **性能提升**：NDCG@10提升3-5%，HR@10提升2-4%
- **创新性**：真正实现LLM与推荐系统的深度融合
- **可解释性**：Attention权重提供特征重要性解释

## 注意事项

1. **GPU内存**：动态提取模式需要更多GPU内存，建议batch_size=1024
2. **训练时间**：端到端微调需要更长时间，建议使用PEFT加速
3. **数据路径**：确保ml-1m数据集路径正确

## 参考文献

- GPT-2: https://huggingface.co/docs/transformers/model_doc/gpt2
- PEFT: https://huggingface.co/docs/transformers/peft
- Neural Collaborative Filtering: He et al., WWW 2017

