# run.py 运行输出说明

## 原本的 run.py 运行后会得到什么？

### 阶段1：快速训练（缓存模式，冻结GPT-2）

**输出内容**：

1. **训练过程日志**（每个epoch）：
   ```
   Epoch   1/15 - train_loss: 0.6931 - valid_ndcg@10: 0.0234 - valid_hr@10: 0.1234 - train_time: 45.23s - valid_time: 12.34s
   Epoch   2/15 - train_loss: 0.6823 - valid_ndcg@10: 0.0345 - valid_hr@10: 0.1456 - train_time: 44.56s - valid_time: 11.89s
   ...
   ```

2. **阶段1完成结果**：
   ```
   阶段1完成 - Validation: {'NDCG@10': 0.1234, 'HR@10': 0.2345, 'NDCG@20': 0.2345, 'HR@20': 0.3456, ...}
   ```
   或（如果有test_loader）：
   ```
   阶段1完成 - Validation: {'NDCG@10': 0.1234, ...}
   阶段1完成 - Test: {'NDCG@10': 0.1123, ...}
   ```

### 阶段2：精细调优（动态模式，端到端微调）

**输出内容**：

1. **训练过程日志**（每个epoch）：
   ```
   Epoch   1/30 - train_loss: 0.6543 - valid_ndcg@10: 0.1456 - valid_hr@10: 0.2567 - train_time: 78.90s - valid_time: 15.67s
   ...
   ```

2. **模型保存信息**（如果验证集指标提升）：
   - 模型会自动保存到 `gpt2_recommender_enhanced.pth`

3. **训练完成结果**：
   ```
   ============================================================
   训练完成！最终结果：
   ============================================================
   Best Validation: {'NDCG@10': 0.2345, 'HR@10': 0.3456, 'NDCG@20': 0.3456, 'HR@20': 0.4567, ...}
   Test Result: {'NDCG@10': 0.2234, 'HR@10': 0.3345, ...}
   ```

### Trainer.fit() 返回值的说明

`trainer.fit()` 的返回值：

- **如果有 test_loader**：返回 `(best_valid_result, test_result)` 元组
  - `best_valid_result`: 验证集上的最佳结果（字典）
  - `test_result`: 测试集上的结果（字典）
  
- **如果没有 test_loader**：返回 `best_valid_result`（字典）

**结果字典包含的指标**：
```python
{
    'NDCG@10': 0.1234,   # 归一化折损累积增益@10
    'HR@10': 0.2345,    # 命中率@10 (Hit Rate)
    'NDCG@20': 0.2345,
    'HR@20': 0.3456,
    'NDCG@50': 0.3456,
    'HR@50': 0.4567
}
```

### 训练过程中的其他输出

1. **GPU信息**（如果使用多GPU）：
   ```
   ✓ 检测到 8 张GPU，将使用 DataParallel 进行多GPU训练
   
   GPU显存信息:
     GPU 0 (NVIDIA A100): 已分配 0.50GB / 已保留 0.60GB / 总计 40.00GB
     ...
   ```

2. **早停信息**（如果触发）：
   ```
   Early stopping triggered
   ```

3. **模型缓存信息**（阶段1）：
   ```
   Precomputing user metadata embeddings (cached mode)...
   Precomputing item metadata embeddings (cached mode)...
   ```

4. **动态模式信息**（阶段2）：
   ```
   Using dynamic metadata extraction (end-to-end training mode)
   ```

## 使用独立脚本评估模型

### ✅ 可以！我已经创建了独立的评估脚本

我已经创建了 `evaluate_model.py`，可以单独加载训练好的模型进行评估。

### 使用方法

#### 方法1：使用默认参数

```bash
python evaluate_model.py
```

默认会：
- 加载 `gpt2_recommender_enhanced.pth`
- 使用默认数据路径
- 评估 K=[10, 20, 50]
- 保存结果到 `test_results_standard.json`

#### 方法2：指定参数

```bash
python evaluate_model.py \
    --model_path gpt2_recommender_enhanced.pth \
    --data_path /path/to/ml-1m \
    --device cuda \
    --k_list 10 20 50 \
    --output my_results.json
```

#### 方法3：在Python代码中使用

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
    use_cache=True,      # 与训练时保持一致
    use_attention=True,   # 与训练时保持一致
    freeze_gpt2=False
)

# 3. 加载模型权重
model.load_state_dict(torch.load('gpt2_recommender_enhanced.pth'))
model.eval()
model = model.to('cuda')

# 4. 执行标准评估
results = evaluate_model_standard(
    model=model,
    dataset=dataset,
    train_data=train_data,
    valid_data=valid_data,
    test_data=test_data,
    device='cuda',
    k_list=[10, 20, 50],
    verbose=True
)

# 5. 查看结果
print(results)
```

### 评估脚本的功能

`evaluate_model.py` 包含：

1. ✅ **自动加载模型**：支持单GPU和多GPU保存的模型
2. ✅ **参数配置**：可通过命令行参数配置
3. ✅ **错误处理**：检查模型文件是否存在
4. ✅ **标准评估**：按照您要求的流程评估
5. ✅ **结果保存**：自动保存JSON格式结果

### 命令行参数说明

```bash
python evaluate_model.py --help
```

参数列表：
- `--model_path`: 模型权重文件路径（默认：`gpt2_recommender_enhanced.pth`）
- `--data_path`: 数据集路径（默认：`/Users/zhuxuzhou/Downloads/ml-1m`）
- `--device`: 设备（默认：`cuda`）
- `--embed_dim`: 嵌入维度（默认：`64`）
- `--use_cache`: 是否使用缓存模式（默认：True）
- `--use_attention`: 是否使用注意力机制（默认：True）
- `--k_list`: K值列表（默认：`10 20 50`）
- `--output`: 结果保存路径（默认：`test_results_standard.json`）

### 注意事项

1. **模型配置必须一致**：
   - `embed_dim` 必须与训练时相同
   - `use_cache` 和 `use_attention` 应该与训练时相同
   - 数据集路径和分割比例必须相同

2. **模型文件位置**：
   - 默认查找 `gpt2_recommender_enhanced.pth`
   - 如果保存在其他位置，使用 `--model_path` 指定

3. **设备选择**：
   - 如果只有CPU，使用 `--device cpu`
   - 多GPU环境下会自动处理

## 总结

### 原本 run.py 的输出

1. **训练过程日志**：每个epoch的训练和验证指标
2. **阶段1结果**：验证集（和测试集）的最佳指标
3. **阶段2结果**：最终的最佳验证集和测试集指标
4. **模型文件**：`gpt2_recommender_enhanced.pth`（如果save_model=True）

### 独立评估脚本

✅ **可以使用** `evaluate_model.py` 单独评估：
- 加载训练好的模型
- 按照标准流程评估
- 输出Recall, Precision, NDCG指标
- 保存结果到JSON文件

### 推荐工作流程

1. **训练模型**：
   ```bash
   python run.py
   ```
   得到：训练日志 + 模型文件 `gpt2_recommender_enhanced.pth`

2. **评估模型**（可选，如果run.py中已包含评估则跳过）：
   ```bash
   python evaluate_model.py
   ```
   得到：标准评估结果 + `test_results_standard.json`

3. **查看结果**：
   - 控制台输出：实时查看评估进度和结果
   - JSON文件：保存详细结果供后续分析

