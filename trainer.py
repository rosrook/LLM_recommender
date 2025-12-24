import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time
from logging import getLogger
from collections import defaultdict
from tqdm import tqdm


class Trainer:
    def __init__(
            self,
            model,
            train_data,
            eval_data,
            test_data=None,
            device='cuda',
            epochs=300,
            batch_size=2048,
            optimizer='adam',
            lr=1e-3,
            weight_decay=0,
            eval_step=1,
            early_stop_patience=10,
            use_multi_gpu=True,  # 是否自动使用多GPU
            max_grad_norm=1.0,   # 梯度裁剪阈值
            enable_grad_clip=True  # 是否启用梯度裁剪
    ):
        # 检测并设置设备
        self.device = device
        self.use_multi_gpu = use_multi_gpu and device == 'cuda'
        self.num_gpus = 0
        
        if self.use_multi_gpu and torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            if self.num_gpus > 1:
                print(f"✓ 检测到 {self.num_gpus} 张GPU，将使用 DataParallel 进行多GPU训练")
                # 显示GPU显存信息
                self._print_gpu_memory_info()
                # 将模型移到主GPU
                self.model = model.to(device)
                # 使用 DataParallel 包装模型
                self.model = nn.DataParallel(self.model)
                self.device = device  # 主设备
            else:
                print(f"✓ 检测到 1 张GPU，使用单GPU训练")
                self.model = model.to(device)
                self.use_multi_gpu = False
        else:
            if device == 'cuda' and not torch.cuda.is_available():
                print("警告: CUDA不可用，将使用CPU训练")
                self.device = 'cpu'
            self.model = model.to(self.device)
            self.use_multi_gpu = False
        
        self.epochs = epochs
        self.eval_step = eval_step
        self.early_stop_patience = early_stop_patience
        self.logger = getLogger()

        # Setup dataloaders
        self.train_loader = train_data
        self.eval_loader = eval_data
        self.test_loader = test_data

        # Setup optimizer (注意：如果使用DataParallel，需要访问module)
        model_for_optimizer = self.model.module if self.use_multi_gpu else self.model
        if optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(model_for_optimizer.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(model_for_optimizer.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported")

        # Training records
        self.best_valid_score = -np.inf
        self.best_valid_result = None
        self.train_loss_dict = defaultdict(list)
        self.eval_loss_dict = defaultdict(list)
        self.best_model_path = None
        self.wait_epochs = 0
        
        # 梯度裁剪配置
        self.max_grad_norm = max_grad_norm
        self.enable_grad_clip = enable_grad_clip
        
        # Loss监测配置
        self.loss_history = []  # 记录loss历史，用于检测异常
        self.loss_anomaly_threshold = {
            'max_loss': 10.0,      # 最大loss阈值
            'min_loss': 0.0,       # 最小loss阈值（BPR loss应该是正数）
            'nan_check': True,     # 检查NaN
            'inf_check': True,     # 检查Inf
            'no_decrease_epochs': 10,  # 连续多少epoch不下降才报警
            'rising_threshold': 0.001  # loss上升阈值（超过此值才认为是真正上升）
        }
        
        # 梯度监测配置
        self.grad_norm_history = []  # 记录梯度范数历史
        self.score_stats_history = []  # 记录模型输出统计信息
    
    def _print_gpu_memory_info(self):
        """打印GPU显存信息"""
        if torch.cuda.is_available():
            print("\nGPU显存信息:")
            for i in range(self.num_gpus):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = props.total_memory / 1024**3
                print(f"  GPU {i} ({props.name}): "
                      f"已分配 {allocated:.2f}GB / "
                      f"已保留 {reserved:.2f}GB / "
                      f"总计 {total:.2f}GB")
            print()
    
    def _diagnose_slow_training(self, batch_idx, users, elapsed_time):
        """诊断训练缓慢的原因"""
        print(f"     - Batch索引: {batch_idx}")
        print(f"     - Batch大小: {len(users)}")
        print(f"     - 已耗时: {elapsed_time:.1f}秒 ({elapsed_time/60:.1f}分钟)")
        
        # 检查模型配置
        if hasattr(self.model, 'module'):
            model_check = self.model.module
        else:
            model_check = self.model
        
        if hasattr(model_check, 'use_cache'):
            use_cache = model_check.use_cache
            print(f"     - 缓存模式: {'是' if use_cache else '否（动态模式）'}")
            if not use_cache:
                print(f"       ⚠️  动态模式需要实时编码metadata，这是正常的但会很慢")
        
        if hasattr(model_check, 'freeze_gpt2'):
            freeze_gpt2 = model_check.freeze_gpt2
            print(f"     - GPT-2冻结: {'是' if freeze_gpt2 else '否（正在微调）'}")
            if not freeze_gpt2:
                print(f"       ⚠️  GPT-2未冻结，需要计算梯度，会增加计算时间")
        
        # 检查GPU使用情况
        if torch.cuda.is_available():
            print(f"     - GPU使用情况:")
            for i in range(self.num_gpus):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                props = torch.cuda.get_device_properties(i)
                memory_total = props.total_memory / 1024**3
                print(f"       GPU {i}: 显存={memory_allocated:.2f}GB/{memory_total:.2f}GB "
                      f"(已保留={memory_reserved:.2f}GB)")
        
        print(f"\n     建议：")
        print(f"       1. 如果这是第一个batch，请继续等待（动态模式需要2-5分钟）")
        print(f"       2. 如果持续很慢，考虑：")
        print(f"          - 使用缓存模式（use_cache=True）")
        print(f"          - 减少batch_size")
        print(f"          - 冻结GPT-2（freeze_gpt2=True）")
        print(f"       3. 如果超过10分钟仍无进展，可能是真的卡住了，请检查：")
        print(f"          - GPU是否正常工作（nvidia-smi）")
        print(f"          - 是否有其他进程占用GPU")
        print(f"          - 显存是否充足")
    
    def _check_loss_anomaly(self, loss_value, batch_idx, batch_size):
        """
        检查loss是否异常
        
        Args:
            loss_value: loss值
            batch_idx: batch索引
            batch_size: batch大小
        """
        threshold = self.loss_anomaly_threshold
        error_messages = []
        
        # 检查NaN
        if threshold['nan_check'] and (torch.isnan(torch.tensor(loss_value)) or np.isnan(loss_value)):
            error_messages.append(f"❌ 严重错误：Loss为NaN！")
            error_messages.append(f"   Batch索引: {batch_idx}, Batch大小: {batch_size}")
            error_messages.append(f"   可能原因：")
            error_messages.append(f"     1. 梯度爆炸（学习率太大）")
            error_messages.append(f"     2. 数值不稳定（模型输出异常）")
            error_messages.append(f"     3. 数据问题（包含NaN值）")
            error_messages.append(f"   建议：")
            error_messages.append(f"     - 减小学习率")
            error_messages.append(f"     - 检查输入数据")
            error_messages.append(f"     - 添加梯度裁剪")
            print("\n" + "\n".join(error_messages))
            raise ValueError("Loss为NaN，训练异常终止！")
        
        # 检查Inf
        if threshold['inf_check'] and (torch.isinf(torch.tensor(loss_value)) or np.isinf(loss_value)):
            error_messages.append(f"❌ 严重错误：Loss为Inf！")
            error_messages.append(f"   Batch索引: {batch_idx}, Batch大小: {batch_size}")
            error_messages.append(f"   可能原因：")
            error_messages.append(f"     1. 梯度爆炸")
            error_messages.append(f"     2. 数值溢出")
            error_messages.append(f"   建议：")
            error_messages.append(f"     - 减小学习率")
            error_messages.append(f"     - 添加梯度裁剪")
            print("\n" + "\n".join(error_messages))
            raise ValueError("Loss为Inf，训练异常终止！")
        
        # 检查loss是否异常高
        if loss_value > threshold['max_loss']:
            error_messages.append(f"⚠️  警告：Loss异常高！")
            error_messages.append(f"   当前loss: {loss_value:.4f} (阈值: {threshold['max_loss']})")
            error_messages.append(f"   Batch索引: {batch_idx}, Batch大小: {batch_size}")
            error_messages.append(f"   可能原因：")
            error_messages.append(f"     1. 学习率太大，导致训练不稳定")
            error_messages.append(f"     2. 模型初始化问题")
            error_messages.append(f"     3. 数据预处理问题")
            error_messages.append(f"   建议：")
            error_messages.append(f"     - 检查学习率（当前: {self.optimizer.param_groups[0]['lr']}）")
            error_messages.append(f"     - 如果持续出现，考虑减小学习率")
            print("\n" + "\n".join(error_messages))
            # 不抛出异常，只警告
        
        # 检查loss是否为负数（BPR loss应该是正数）
        if loss_value < threshold['min_loss']:
            error_messages.append(f"⚠️  警告：Loss为负数！")
            error_messages.append(f"   当前loss: {loss_value:.4f}")
            error_messages.append(f"   Batch索引: {batch_idx}, Batch大小: {batch_size}")
            error_messages.append(f"   可能原因：")
            error_messages.append(f"     1. 损失函数实现问题")
            error_messages.append(f"     2. 模型输出异常（pos_score < neg_score且差异很大）")
            error_messages.append(f"   建议：")
            error_messages.append(f"     - 检查损失函数实现")
            error_messages.append(f"     - 检查模型输出")
            print("\n" + "\n".join(error_messages))
            # 不抛出异常，只警告
    
    def _monitor_and_clip_gradients(self, batch_idx):
        """
        监测和裁剪梯度
        
        Args:
            batch_idx: batch索引
        
        Returns:
            float: 梯度范数
        """
        # 获取模型参数（处理DataParallel）
        if self.use_multi_gpu:
            model = self.model.module
        else:
            model = self.model
        
        # 计算梯度范数
        total_norm = 0.0
        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
        else:
            total_norm = 0.0
        
        # 检查梯度爆炸
        if total_norm > 100.0:  # 梯度范数超过100认为可能爆炸
            if batch_idx == 0 or batch_idx % 50 == 0:  # 每50个batch或第一个batch提示
                print(f"\n  ⚠️  警告：梯度范数异常大！")
                print(f"     当前梯度范数: {total_norm:.4f} (阈值: 100.0)")
                print(f"     Batch索引: {batch_idx}")
                print(f"     可能原因：梯度爆炸，学习率太大")
                print(f"     建议：减小学习率或启用梯度裁剪")
        
        # 梯度裁剪
        if self.enable_grad_clip and total_norm > self.max_grad_norm:
            # DataParallel和单GPU都使用相同的方法（因为我们已经获取了module）
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            
            if batch_idx == 0 or batch_idx % 100 == 0:  # 每100个batch提示一次
                print(f"\n  ✓ 已应用梯度裁剪: {total_norm:.4f} -> {self.max_grad_norm}")
        
        return total_norm
    
    def _check_model_outputs(self, pos_scores, neg_scores, batch_idx):
        """
        检查模型输出是否正常
        
        Args:
            pos_scores: 正样本分数
            neg_scores: 负样本分数
            batch_idx: batch索引
        """
        with torch.no_grad():
            pos_mean = pos_scores.mean().item()
            neg_mean = neg_scores.mean().item()
            pos_std = pos_scores.std().item()
            neg_std = neg_scores.std().item()
            diff_mean = (pos_scores - neg_scores).mean().item()
            
            # 记录统计信息
            stats = {
                'pos_mean': pos_mean,
                'neg_mean': neg_mean,
                'pos_std': pos_std,
                'neg_std': neg_std,
                'diff_mean': diff_mean,
                'batch_idx': batch_idx
            }
            self.score_stats_history.append(stats)
            
            # 检查异常情况
            warnings = []
            
            # 检查1: pos_score是否小于neg_score（这是不正常的）
            if pos_mean < neg_mean:
                warnings.append(f"⚠️  异常：正样本分数({pos_mean:.4f}) < 负样本分数({neg_mean:.4f})")
                warnings.append(f"     这意味着模型预测错误，loss会持续上升")
                warnings.append(f"     可能原因：")
                warnings.append(f"       1. 模型初始化问题")
                warnings.append(f"       2. 学习率太大，导致模型无法学习正确方向")
                warnings.append(f"       3. 数据标签问题")
            
            # 检查2: 分数差异是否太小
            if diff_mean < 0.1:
                warnings.append(f"⚠️  警告：正负样本分数差异很小({diff_mean:.4f})")
                warnings.append(f"     模型可能无法区分正负样本")
            
            # 检查3: 分数是否在合理范围内（BPR通常期望分数在合理范围）
            if pos_mean < -10 or pos_mean > 10:
                warnings.append(f"⚠️  警告：正样本分数异常({pos_mean:.4f})，可能超出合理范围")
            
            if neg_mean < -10 or neg_mean > 10:
                warnings.append(f"⚠️  警告：负样本分数异常({neg_mean:.4f})，可能超出合理范围")
            
            # 输出警告
            if warnings:
                print(f"\n  Batch {batch_idx} 模型输出检查：")
                print(f"    正样本分数: mean={pos_mean:.4f}, std={pos_std:.4f}")
                print(f"    负样本分数: mean={neg_mean:.4f}, std={neg_std:.4f}")
                print(f"    分数差异: {diff_mean:.4f}")
                for warning in warnings:
                    print(f"    {warning}")
    
    def _check_epoch_loss_trend(self):
        """检查epoch级别的loss趋势"""
        if len(self.loss_history) < self.loss_anomaly_threshold['no_decrease_epochs']:
            return
        
        # 检查最近N个epoch的loss是否完全不下降
        recent_losses = self.loss_history[-self.loss_anomaly_threshold['no_decrease_epochs']:]
        
        # 计算loss变化
        loss_changes = [recent_losses[i] - recent_losses[i-1] for i in range(1, len(recent_losses))]
        
        # 如果所有变化都>=0（即完全不下降或上升）
        if all(change >= 0 for change in loss_changes):
            print(f"\n⚠️  警告：连续 {self.loss_anomaly_threshold['no_decrease_epochs']} 个epoch loss不下降！")
            print(f"   最近 {len(recent_losses)} 个epoch的loss: {[f'{l:.4f}' for l in recent_losses]}")
            print(f"   Loss变化: {[f'{c:+.4f}' for c in loss_changes]}")
            print(f"   可能原因：")
            print(f"     1. 学习率太小，模型无法学习")
            print(f"     2. 模型容量不足")
            print(f"     3. 数据问题")
            print(f"     4. 已经收敛（如果loss已经很低）")
            print(f"   建议：")
            print(f"     - 检查学习率（当前: {self.optimizer.param_groups[0]['lr']}）")
            print(f"     - 检查验证集指标是否提升")
            print(f"     - 如果验证集指标也不提升，考虑调整超参数")
        
        # 检查loss是否持续上升（使用阈值避免微小波动误报）
        threshold = self.loss_anomaly_threshold['rising_threshold']
        recent_changes = loss_changes[-3:] if len(loss_changes) >= 3 else loss_changes
        if len(recent_changes) >= 3 and all(change > threshold for change in recent_changes):
            print(f"\n⚠️  警告：Loss持续上升！")
            print(f"   最近3个epoch的loss: {[f'{l:.6f}' for l in recent_losses[-3:]]}")
            print(f"   Loss变化: {[f'{c:+.6f}' for c in recent_changes]}")
            
            # 检查梯度范数
            if len(self.grad_norm_history) > 0:
                recent_grad_norms = self.grad_norm_history[-100:]  # 最近100个batch
                avg_grad_norm = np.mean(recent_grad_norms)
                max_grad_norm = np.max(recent_grad_norms)
                print(f"   梯度范数: 平均={avg_grad_norm:.4f}, 最大={max_grad_norm:.4f}")
                if max_grad_norm > 100.0:
                    print(f"   ⚠️  检测到梯度爆炸！最大梯度范数: {max_grad_norm:.4f}")
            
            # 检查模型输出统计
            if len(self.score_stats_history) > 0:
                recent_stats = self.score_stats_history[-10:]  # 最近10次检查
                avg_pos_mean = np.mean([s['pos_mean'] for s in recent_stats])
                avg_neg_mean = np.mean([s['neg_mean'] for s in recent_stats])
                avg_diff = np.mean([s['diff_mean'] for s in recent_stats])
                print(f"   模型输出统计（最近10次检查）:")
                print(f"     正样本分数均值: {avg_pos_mean:.4f}")
                print(f"     负样本分数均值: {avg_neg_mean:.4f}")
                print(f"     分数差异均值: {avg_diff:.4f}")
                if avg_pos_mean < avg_neg_mean:
                    print(f"   ❌ 严重：正样本分数 < 负样本分数，这是导致loss上升的根本原因！")
            
            print(f"   可能原因：")
            print(f"     1. 学习率太大，导致训练不稳定")
            print(f"     2. 梯度爆炸（检查梯度范数）")
            print(f"     3. 模型输出异常（pos_score < neg_score）")
            print(f"   建议：")
            print(f"     - 减小学习率（当前: {self.optimizer.param_groups[0]['lr']}）")
            print(f"     - 检查是否已启用梯度裁剪（当前: {self.enable_grad_clip}）")
            print(f"     - 如果持续上升，考虑暂停训练并检查模型")
            
            # 如果loss上升很严重，提供自动调整建议
            avg_rise = np.mean(recent_changes)
            if avg_rise > 0.01:  # 平均每个epoch上升超过0.01
                print(f"\n   ⚠️  严重：Loss平均每个epoch上升 {avg_rise:.6f}")
                print(f"   建议立即采取行动：")
                suggested_lr = self.optimizer.param_groups[0]['lr'] * 0.5
                print(f"     1. 减小学习率到 {suggested_lr:.6f}（当前的一半）")
                print(f"     2. 确保梯度裁剪已启用（max_grad_norm={self.max_grad_norm}）")
                print(f"     3. 检查模型输出范围（见上面的统计信息）")
                print(f"     4. 如果问题持续，考虑重新初始化模型或使用更小的学习率")

    def _train_epoch(self):
        """Train model for one epoch"""
        self.model.train()
        total_loss = 0.
        total_samples = 0

        # Time tracking
        data_loading_time = 0
        forward_time = 0
        backward_time = 0
        optimizer_time = 0

        # 添加batch级别的进度条
        batch_iterator = tqdm(self.train_loader, desc='  Batches', leave=False, ncols=100)
        
        # 监测变量：用于检测训练是否卡住
        batch_start_time = time()
        first_batch_start_time = None
        slow_batch_count = 0
        batch_timeout_threshold = 300  # 5分钟超时阈值（秒）
        slow_batch_threshold = 60  # 1分钟慢batch阈值（秒）
        
        for batch_idx, batch in enumerate(batch_iterator):
            batch_start = time()
            users, pos_items, neg_items = batch
            data_loading_time += time() - batch_start

            try:
                # 监测：记录第一个batch的开始时间
                if batch_idx == 0:
                    first_batch_start_time = time()
                    # 检查是否是动态模式（阶段2）
                    if hasattr(self.model, 'module'):
                        model_check = self.model.module
                    else:
                        model_check = self.model
                    if hasattr(model_check, 'use_cache') and not model_check.use_cache:
                        print(f"\n  ⚠️  检测到动态模式（阶段2），第一个batch需要编码metadata")
                        print(f"     预计等待时间：2-5分钟，请耐心等待...")
                
                # Forward pass
                forward_start = time()
                pos_scores, neg_scores = self.model(batch)
                forward_time += time() - forward_start
                
                # 监测：检查forward pass是否耗时过长
                forward_duration = time() - forward_start
                if forward_duration > slow_batch_threshold:
                    slow_batch_count += 1
                    if slow_batch_count <= 3:  # 只提示前3次
                        print(f"\n  ⚠️  检测到慢batch（batch {batch_idx+1}，forward耗时 {forward_duration:.1f}秒）")
                        if batch_idx == 0:
                            print(f"     这是正常的：第一个batch需要初始化GPT-2编码器")
                        else:
                            print(f"     可能原因：动态模式需要编码大量metadata")
                            print(f"     建议：如果持续很慢，考虑使用缓存模式或减少batch_size")

                # Loss computation and backward pass
                backward_start = time()
                # Use model's calculate_loss method if available, otherwise use default BPR loss
                # 处理DataParallel：需要访问module
                model_for_loss = self.model.module if self.use_multi_gpu else self.model
                if hasattr(model_for_loss, 'calculate_loss'):
                    loss = model_for_loss.calculate_loss(pos_scores, neg_scores)
                else:
                    loss = -(pos_scores - neg_scores).sigmoid().log().mean()
                backward_time += time() - backward_start

                # Optimizer step
                optimizer_start = time()
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度监测和裁剪
                grad_norm = self._monitor_and_clip_gradients(batch_idx)
                self.grad_norm_history.append(grad_norm)
                
                # 检查模型输出（pos_score vs neg_score）
                if batch_idx == 0 or batch_idx % 100 == 0:  # 每100个batch检查一次
                    self._check_model_outputs(pos_scores, neg_scores, batch_idx)
                
                self.optimizer.step()
                optimizer_time += time() - optimizer_start

                # Loss监测：检查loss是否异常
                loss_value = loss.item()
                self._check_loss_anomaly(loss_value, batch_idx, len(users))
                
                total_loss += loss_value * len(users)
                total_samples += len(users)
                
                # 监测：计算整个batch的处理时间
                batch_total_time = time() - batch_start_time
                
                # 监测：检查是否超时（特别是第一个batch）
                if batch_idx == 0 and first_batch_start_time:
                    elapsed = time() - first_batch_start_time
                    if elapsed > batch_timeout_threshold:
                        print(f"\n  ⚠️  第一个batch处理时间超过 {batch_timeout_threshold//60} 分钟")
                        print(f"     当前耗时: {elapsed:.1f}秒")
                        print(f"     诊断信息：")
                        self._diagnose_slow_training(batch_idx, users, elapsed)
                
                # 更新进度条显示当前loss和batch时间
                batch_iterator.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'time': f'{batch_total_time:.1f}s'
                })
                
                # 记录loss历史（用于epoch级别的监测）
                if not hasattr(self, '_current_epoch_losses'):
                    self._current_epoch_losses = []
                self._current_epoch_losses.append(loss_value)
                
                batch_start_time = time()  # 重置batch开始时间
                
            except RuntimeError as e:
                if "out of memory" in str(e) or "CUDA out of memory" in str(e):
                    # 显存不足错误处理
                    print(f"\n⚠️ 显存不足错误！当前batch size: {len(users)}")
                    print("建议解决方案：")
                    print("  1. 减少batch_size（推荐）")
                    print("  2. 使用动态模式（use_cache=False）")
                    print("  3. 使用更少的GPU（设置CUDA_VISIBLE_DEVICES）")
                    print("  4. 使用梯度累积（gradient_accumulation_steps）")
                    
                    # 清理显存
                    torch.cuda.empty_cache()
                    
                    # 重新抛出错误，让用户决定如何处理
                    raise RuntimeError(
                        f"GPU显存不足。当前batch size: {len(users)}, "
                        f"GPU数量: {self.num_gpus}。"
                        f"建议：将batch_size减少到 {len(users) // 2} 或更小。"
                    ) from e
                else:
                    # 其他运行时错误，直接抛出
                    raise

        # Log time costs
        # print(
        #     f"\nEpoch time breakdown:"
        #     f"\n - Data loading: {data_loading_time:.2f}s"
        #     f"\n - Forward pass: {forward_time:.2f}s"
        #     f"\n - Backward pass: {backward_time:.2f}s"
        #     f"\n - Optimizer step: {optimizer_time:.2f}s"
        #     f"\n - Total time: {data_loading_time + forward_time + backward_time + optimizer_time:.2f}s"
        # )

        avg_loss = total_loss / total_samples
        
        # Epoch级别的loss监测
        if hasattr(self, '_current_epoch_losses') and len(self._current_epoch_losses) > 0:
            self.loss_history.append(avg_loss)
            self._check_epoch_loss_trend()
            # 清空当前epoch的loss记录
            self._current_epoch_losses = []
        
        return avg_loss

    @torch.no_grad()
    def _evaluate(self, eval_loader, k=[10, 20, 50]):
        """Evaluate model on validation/test set"""
        self.model.eval()
        
        # 对于评估，如果使用DataParallel，需要访问module来调用recommend方法
        model_for_eval = self.model.module if self.use_multi_gpu else self.model

        metrics = {
            f'NDCG@{k_}': [] for k_ in k
        }
        metrics.update({f'HR@{k_}': [] for k_ in k})

        # 添加评估进度条
        eval_iterator = tqdm(eval_loader, desc='  Evaluating', leave=False, ncols=100)
        for users in eval_iterator:
            # Get user positive items
            pos_items = eval_loader.get_user_eval_pos_items(users.cpu().numpy())

            # Get recommendations
            scores = []
            for user in users:
                score = model_for_eval.recommend(user)
                scores.append(score)
            scores = torch.stack(scores)

            # Calculate metrics for each user
            recommends = scores.topk(max(k), dim=1)[1].cpu().numpy()
            for idx, items in enumerate(pos_items):
                if not items:  # Skip if no positive items
                    continue

                rank_list = recommends[idx]
                hit_list = [(item in items) for item in rank_list]

                # Calculate metrics
                for k_ in k:
                    hit_k = hit_list[:k_]
                    ndcg_k = self._calculate_ndcg(hit_k, min(k_, len(items)))
                    hr_k = np.sum(hit_k) > 0

                    metrics[f'NDCG@{k_}'].append(ndcg_k)
                    metrics[f'HR@{k_}'].append(hr_k)

        # Average metrics
        result = {
            metric: np.mean(values) for metric, values in metrics.items()
        }
        return result

    def _calculate_ndcg(self, hit_list, pos_items):
        """Calculate NDCG metric"""
        dcg = np.sum([int(hit) / np.log2(idx + 2) for idx, hit in enumerate(hit_list)])
        idcg = np.sum([1 / np.log2(idx + 2) for idx in range(pos_items)])
        return dcg / idcg if idcg > 0 else 0

    # def _should_early_stop(self, metric_value):
    #     """Check if training should early stop"""
    #     if metric_value > self.best_valid_score:
    #         self.best_valid_score = metric_value
    #         self.wait_epochs = 0
    #         return False
    #     else:
    #         self.wait_epochs += 1
    #         return self.wait_epochs >= self.early_stop_patience

    def fit(self, save_model=False, model_path='checkpoint.pth'):
        """Train model"""
        for epoch in tqdm(range(self.epochs), desc='Training'):
            # Train
            train_start = time()
            train_loss = self._train_epoch()
            train_time = time() - train_start

            # Evaluate
            if (epoch + 1) % self.eval_step == 0:
                valid_start = time()
                valid_result = self._evaluate(self.eval_loader)
                valid_time = time() - valid_start

                # Log progress
                self.logger.info(
                    f'Epoch {epoch + 1:3d}/{self.epochs:3d} - '
                    f'train_loss: {train_loss:.4f} - '
                    f'valid_ndcg@10: {valid_result["NDCG@10"]:.4f} - '
                    f'valid_hr@10: {valid_result["HR@10"]:.4f} - '
                    f'train_time: {train_time:.2f}s - '
                    f'valid_time: {valid_time:.2f}s'
                )

                # Save best model and Early stopping check
                metric = valid_result["NDCG@10"]
                if metric > self.best_valid_score:
                    # 有提升:更新best,重置等待计数,保存model
                    self.best_valid_score = metric
                    self.best_valid_result = valid_result
                    self.wait_epochs = 0

                    if save_model:
                        # 如果使用DataParallel，保存时需要访问module
                        model_to_save = self.model.module if self.use_multi_gpu else self.model
                        torch.save(model_to_save.state_dict(), model_path)
                        self.best_model_path = model_path
                else:
                    # 无提升:增加等待计数并检查是否达到早停阈值
                    self.wait_epochs += 1
                    if self.wait_epochs >= self.early_stop_patience:
                        self.logger.info("Early stopping triggered")
                        break
                # # Save best model
                # if valid_result["NDCG@10"] > self.best_valid_score:
                #     self.best_valid_score = valid_result["NDCG@10"]
                #     self.best_valid_result = valid_result
                #     if save_model:
                #         torch.save(self.model.state_dict(), model_path)
                #         self.best_model_path = model_path
                #
                # # Early stopping check
                # if self._should_early_stop(valid_result["NDCG@10"]):
                #     self.logger.info("Early stopping triggered")
                #     break

        # Load best model for testing
        if save_model and self.best_model_path is not None:
            # 如果使用DataParallel，加载时需要访问module
            model_to_load = self.model.module if self.use_multi_gpu else self.model
            # 处理可能的DataParallel保存格式（带module.前缀）
            state_dict = torch.load(self.best_model_path, map_location=self.device)
            # 如果保存的state_dict有module.前缀，需要去掉
            if any(key.startswith('module.') for key in state_dict.keys()):
                if not self.use_multi_gpu:
                    # 单GPU加载多GPU保存的模型，需要去掉module.前缀
                    state_dict = {k[7:] if k.startswith('module.') else k: v 
                                 for k, v in state_dict.items()}
            elif self.use_multi_gpu:
                # 多GPU加载单GPU保存的模型，需要添加module.前缀
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
            model_to_load.load_state_dict(state_dict)

        # Final test evaluation
        if self.test_loader is not None:
            test_result = self._evaluate(self.test_loader)
            self.logger.info(f"Test Result: {test_result}")
            return self.best_valid_result, test_result

        return self.best_valid_result

    def evaluate(self, eval_loader):
        """Evaluate model on given dataloader"""
        return self._evaluate(eval_loader)

