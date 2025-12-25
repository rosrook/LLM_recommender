#!/usr/bin/env python
# coding: utf-8
"""
最优代码框架的训练脚本
采用两阶段训练策略：
1. 阶段1：快速训练（缓存模式，冻结GPT-2）
2. 阶段2：精细调优（动态模式，端到端微调）
"""

import torch
import torch.optim as optim
from dataset import ML1MDataset
from dataloader import TrainDataLoader, EvalDataLoader
from gpt2_recommender_enhanced import GPT2RecommenderEnhanced
from trainer import Trainer
from evaluator import evaluate_model_standard

# ==================== 配置参数 ====================
DATA_PATH = '/home/zhuxuzhou/LLM_recommender/data/ml-1m'  # 修改为你的数据路径
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
    lr=1e-4,  # 降低学习率：从1e-3降到1e-4，避免训练不稳定
    weight_decay=1e-5,
    eval_step=1,
    early_stop_patience=10,
    max_grad_norm=1.0,      # 梯度裁剪阈值
    enable_grad_clip=True   # 启用梯度裁剪
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
print("阶段2：精细调优（缓存模式 + 解冻GPT-2，加速训练）")
print("="*60)
print("优化策略：")
print("  1. 保持缓存模式（use_cache=True）- 避免每个batch重新编码metadata")
print("  2. 解冻GPT-2（使用LoRA，只训练少量参数）")
print("  3. 分层学习率：GPT-2参数使用更小学习率（1e-5），其他参数5e-5")
print("  4. 更严格的梯度裁剪（max_grad_norm=0.5）避免梯度爆炸")
print("="*60)

# 保持缓存模式，但解冻GPT-2（使用LoRA时只有少量参数需要梯度）
print("\n[1/5] 保持缓存模式，解冻GPT-2参数...")
# 注意：保持 use_cache=True，这样metadata embeddings是预计算的
# 只需要计算GPT-2的梯度（如果使用LoRA，只有少量参数）
model.freeze_gpt2 = False   # 解冻GPT-2

# 重新启用梯度
print("[2/5] 解冻GPT-2参数（LoRA参数）...")
for param in model.gpt2_encoder.parameters():
    param.requires_grad = True

# 重新创建训练器（使用分层学习率和更严格的梯度裁剪）
print("[3/5] 创建新的训练器（精细调优 + 分层学习率）...")
# 使用更小的学习率和更严格的梯度裁剪，避免Loss为Inf
trainer = Trainer(
    model=model,
    train_data=train_loader,
    eval_data=valid_loader,
    test_data=test_loader,
    device=DEVICE,
    epochs=30,
    lr=5e-5,  # 大幅降低基础学习率：从1e-4降到5e-5，避免数值溢出
    weight_decay=1e-5,
    eval_step=1,
    early_stop_patience=10,
    max_grad_norm=0.5,      # 更严格的梯度裁剪阈值：从1.0降到0.5
    enable_grad_clip=True,   # 启用梯度裁剪
    use_amp=False           # 暂时禁用混合精度训练，避免数值不稳定
)

# 手动设置分层学习率：GPT-2参数使用更小的学习率
print("[4/5] 设置分层学习率（GPT-2参数使用更小学习率）...")
model_for_optimizer = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model

# 分离GPT-2参数和其他参数
gpt2_params = []
other_params = []
for name, param in model_for_optimizer.named_parameters():
    if param.requires_grad:
        if 'gpt2_encoder' in name:
            gpt2_params.append(param)
        else:
            other_params.append(param)

# 重新创建optimizer，GPT-2参数使用更小的学习率
trainer.optimizer = optim.Adam(
    [
        {'params': gpt2_params, 'lr': 1e-5, 'weight_decay': 1e-5},  # GPT-2参数：非常小的学习率
        {'params': other_params, 'lr': 5e-5, 'weight_decay': 1e-5}  # 其他参数：稍大的学习率
    ]
)
print(f"  ✓ 分层学习率设置完成:")
print(f"    GPT-2参数学习率: 1e-5")
print(f"    其他参数学习率: 5e-5")
print(f"    梯度裁剪阈值: 0.5")

# 开始阶段2训练
print("[5/5] 开始阶段2训练...")
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

# ==================== 标准评估流程 ====================
print("\n" + "="*60)
print("标准评估流程（测试集）")
print("="*60)
print("评估标准：")
print("  1. 对每个用户，只考虑未在训练集+验证集中交互过的item")
print("  2. 使用模型的predict方法计算user-item分数")
print("  3. 排序得到top K推荐")
print("  4. 计算Recall, Precision, NDCG等指标")
print("="*60)

# 加载最佳模型（如果保存了）
if hasattr(trainer, 'best_model_path') and trainer.best_model_path:
    print(f"\n加载最佳模型: {trainer.best_model_path}")
    model_to_eval = model.module if hasattr(model, 'module') else model
    state_dict = torch.load(trainer.best_model_path, map_location=DEVICE)
    
    # 处理可能的DataParallel保存格式
    # 无论当前模型是否使用DataParallel，model_to_eval都不应该有module.前缀
    # 所以如果state_dict有module.前缀，一律去掉
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k[7:] if k.startswith('module.') else k: v 
                     for k, v in state_dict.items()}
        print("  检测到保存的模型使用了DataParallel格式（带module.前缀），已自动去除")
    
    # 使用 strict=False 允许部分加载
    missing_keys, unexpected_keys = model_to_eval.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️  警告: 加载模型时缺失以下键（共{len(missing_keys)}个）: {missing_keys[:5]}")
        if len(missing_keys) > 5:
            print(f"    ... 还有 {len(missing_keys) - 5} 个缺失的键")
    
    if unexpected_keys:
        print(f"⚠️  警告: 加载模型时发现意外的键（共{len(unexpected_keys)}个）: {unexpected_keys[:5]}")
        if len(unexpected_keys) > 5:
            print(f"    ... 还有 {len(unexpected_keys) - 5} 个意外的键")
    
    if not missing_keys and not unexpected_keys:
        print("✓ 模型加载完成（完全匹配）")
    else:
        print(f"✓ 模型加载完成（部分匹配：缺失{len(missing_keys)}个，多余{len(unexpected_keys)}个）")

# 执行标准评估
standard_results = evaluate_model_standard(
    model=model,
    dataset=dataset,
    train_data=train_data,
    valid_data=valid_data,
    test_data=test_data,
    device=DEVICE,
    k_list=[10, 20, 50],
    verbose=True
)

# 保存评估结果
import json
with open('test_results_standard.json', 'w') as f:
    json.dump({k: float(v) for k, v in standard_results.items()}, f, indent=2)
print("\n评估结果已保存到: test_results_standard.json")

