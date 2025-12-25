# GPT-2 解冻后训练加速方案

## 问题描述
当GPT-2解冻后，训练变得极其缓慢，主要原因：
1. **动态模式（use_cache=False）**：每个batch都要实时编码metadata，非常耗时
2. **全量微调**：即使使用LoRA，解冻GPT-2后仍需要计算梯度
3. **没有使用混合精度训练**：FP32训练速度较慢

## 优化方案

### ✅ 方案1：保持缓存模式 + 混合精度训练（推荐）

**策略**：
- 保持 `use_cache=True`：metadata embeddings预计算，避免每个batch重新编码
- 解冻GPT-2（使用LoRA）：只训练少量参数（LoRA rank=8时约0.1%的参数）
- 启用混合精度训练（FP16）：加速2-3倍

**优点**：
- 训练速度快（接近阶段1的速度）
- 仍然可以微调GPT-2（通过LoRA）
- 显存占用更少（FP16）

**实现**：
```python
# 阶段2：保持缓存模式，但解冻GPT-2
model.use_cache = True   # 保持缓存模式
model.freeze_gpt2 = False  # 解冻GPT-2（LoRA参数）

# 启用混合精度训练
trainer = Trainer(
    ...
    use_amp=True  # 启用FP16
)
```

### ✅ 方案2：只微调非GPT-2部分（最快）

**策略**：
- 保持 `use_cache=True` 和 `freeze_gpt2=True`
- 只微调projection层、fusion层和MLP层
- 这些层已经可以学习如何更好地利用GPT-2的embeddings

**优点**：
- 训练速度最快（接近阶段1）
- 显存占用最少
- 仍然可以提升性能（通过优化特征融合）

**缺点**：
- 不能微调GPT-2本身

**实现**：
```python
# 阶段2：只微调非GPT-2部分
model.use_cache = True
model.freeze_gpt2 = True  # 保持冻结

# 只解冻projection和fusion层
for param in model.user_meta_proj.parameters():
    param.requires_grad = True
for param in model.item_meta_proj.parameters():
    param.requires_grad = True
for param in model.user_fusion.parameters():
    param.requires_grad = True
for param in model.item_fusion.parameters():
    param.requires_grad = True
for param in model.mlp_layers.parameters():
    param.requires_grad = True
```

### ⚠️ 方案3：动态模式 + 混合精度（较慢但最灵活）

**策略**：
- 使用 `use_cache=False`：每个batch实时编码metadata
- 解冻GPT-2：端到端微调
- 启用混合精度训练：部分加速

**优点**：
- 可以完全端到端微调GPT-2
- 每个batch的metadata embeddings都会更新

**缺点**：
- 训练速度仍然较慢（比方案1慢5-10倍）
- 显存占用较大

**适用场景**：
- 时间充足
- 需要完全端到端微调
- 数据量较小

## 性能对比

| 方案 | 训练速度 | 显存占用 | 可微调GPT-2 | 推荐度 |
|------|---------|---------|------------|--------|
| 方案1：缓存+FP16+LoRA | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ (LoRA) | ⭐⭐⭐⭐⭐ |
| 方案2：只微调非GPT-2 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | ⭐⭐⭐⭐ |
| 方案3：动态+FP16 | ⭐⭐ | ⭐⭐⭐ | ✅ (全量) | ⭐⭐⭐ |

## 当前实现

当前 `run.py` 已实现**方案1**：
- ✅ 保持缓存模式（`use_cache=True`）
- ✅ 解冻GPT-2（使用LoRA）
- ✅ 启用混合精度训练（`use_amp=True`）

## 进一步优化建议

如果仍然觉得慢，可以尝试：

1. **减小LoRA rank**：
   ```python
   # 在 gpt2_encoder.py 中修改
   peft_config = LoraConfig(
       r=4,  # 从8减小到4（减少可训练参数）
       lora_alpha=16,  # 相应调整
       ...
   )
   ```

2. **增加batch size**（如果显存允许）：
   ```python
   BATCH_SIZE = 4096  # 从2048增加到4096
   ```

3. **减少训练epochs**：
   ```python
   epochs=20  # 从30减少到20
   ```

4. **使用梯度累积**（如果显存不足）：
   - 需要修改trainer.py添加梯度累积功能

5. **使用更小的GPT-2模型**：
   ```python
   gpt2_model_name='gpt2'  # 已经是base版本，可以尝试distilgpt2
   ```

## 总结

**推荐使用方案1**（当前实现）：
- 速度快（接近阶段1）
- 仍然可以微调GPT-2（通过LoRA）
- 显存占用合理
- 性能提升明显

如果时间非常紧急，可以使用**方案2**（只微调非GPT-2部分），速度最快。

