#!/usr/bin/env python
# coding: utf-8
"""
最优代码框架的训练脚本
采用两阶段训练策略：
1. 阶段1：快速训练（缓存模式，冻结GPT-2）
2. 阶段2：精细调优（动态模式，端到端微调）
"""

from dataset import ML1MDataset
from dataloader import TrainDataLoader, EvalDataLoader
from gpt2_recommender_enhanced import GPT2RecommenderEnhanced
from trainer import Trainer

# ==================== 配置参数 ====================
DATA_PATH = '/Users/zhuxuzhou/Downloads/ml-1m'  # 修改为你的数据路径
DEVICE = 'cuda'  # 或 'cpu'
BATCH_SIZE = 2048
EMBED_DIM = 64

# ==================== 阶段1：快速训练 ====================
print("="*60)
print("阶段1：快速训练（缓存模式，冻结GPT-2）")
print("="*60)

# 1. 加载数据集
print("\n[1/5] 加载数据集...")
dataset = ML1MDataset(DATA_PATH, split_ratio=[0.8, 0.1, 0.1])

# 2. 获取数据分割
print("[2/5] 获取数据分割...")
train_data = dataset.get_split_data('train')
valid_data = dataset.get_split_data('validation')
test_data = dataset.get_split_data('test')

# 3. 创建模型（缓存模式，冻结GPT-2）
print("[3/5] 创建模型（缓存模式）...")
model = GPT2RecommenderEnhanced(
    n_users=dataset.get_user_num(),
    n_items=dataset.get_item_num(),
    embed_dim=EMBED_DIM,
    dataset=dataset,
    use_cache=True,        # 缓存模式
    use_attention=True,
    freeze_gpt2=True      # 冻结GPT-2
)

# 4. 创建数据加载器
print("[4/5] 创建数据加载器...")
train_loader = TrainDataLoader(
    train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    device=DEVICE
)
valid_loader = EvalDataLoader(
    valid_data, 
    train_data, 
    batch_size=BATCH_SIZE, 
    device=DEVICE
)
test_loader = EvalDataLoader(
    test_data, 
    train_data, 
    batch_size=BATCH_SIZE, 
    device=DEVICE
)

# 5. 创建训练器并训练
print("[5/5] 开始阶段1训练...")
trainer = Trainer(
    model=model,
    train_data=train_loader,
    eval_data=valid_loader,
    test_data=test_loader,
    device=DEVICE,
    epochs=15,
    lr=1e-3,
    weight_decay=1e-5,
    eval_step=1,
    early_stop_patience=10
)

# 6. 开始训练
result = trainer.fit(save_model=False)
if isinstance(result, tuple):
    valid_result, test_result = result
    print(f"\n阶段1完成 - Validation: {valid_result}")
    print(f"阶段1完成 - Test: {test_result}")
else:
    valid_result = result
    print(f"\n阶段1完成 - Validation: {valid_result}")

# ==================== 阶段2：精细调优 ====================
print("\n" + "="*60)
print("阶段2：精细调优（动态模式，端到端微调）")
print("="*60)

# 切换到动态模式
print("\n[1/5] 切换到动态模式...")
model.use_cache = False    # 切换到动态模式
model.freeze_gpt2 = False   # 解冻GPT-2

# 重新启用梯度
print("[2/5] 解冻GPT-2参数...")
for param in model.gpt2_encoder.parameters():
    param.requires_grad = True

# 重新创建训练器（使用更小的学习率）
print("[3/5] 创建新的训练器（精细调优）...")
trainer = Trainer(
    model=model,
    train_data=train_loader,
    eval_data=valid_loader,
    test_data=test_loader,
    device=DEVICE,
    epochs=30,
    lr=5e-4,  # 更小的学习率
    weight_decay=1e-5,
    eval_step=1,
    early_stop_patience=10
)

# 开始阶段2训练
print("[4/5] 开始阶段2训练...")
result = trainer.fit(
    save_model=True, 
    model_path='gpt2_recommender_enhanced.pth'
)

# 输出最终结果
print("\n" + "="*60)
print("训练完成！最终结果：")
print("="*60)
if isinstance(result, tuple):
    valid_result, test_result = result
    print(f"Best Validation: {valid_result}")
    print(f"Test Result: {test_result}")
else:
    valid_result = result
    print(f"Best Validation: {valid_result}")

