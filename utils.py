# coding: utf-8
"""
实用工具函数
"""

import torch
import os


def get_optimal_batch_size(num_gpus=None, base_batch_size=2048, min_batch_size=32):
    """
    根据GPU数量自动调整batch size
    
    Args:
        num_gpus: GPU数量，如果为None则自动检测
        base_batch_size: 单GPU的基础batch size（总batch size）
        min_batch_size: 每个GPU的最小batch size
    
    Returns:
        调整后的batch size（每个GPU应该使用的batch size）
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if num_gpus > 1:
        # 多GPU时，每个GPU的batch size = 总batch size / GPU数量
        per_gpu_batch = base_batch_size // num_gpus
        # 确保至少为min_batch_size
        per_gpu_batch = max(min_batch_size, per_gpu_batch)
        total_batch = per_gpu_batch * num_gpus
        print(f"✓ 使用 {num_gpus} 张GPU，每个GPU的batch size: {per_gpu_batch}, "
              f"总batch size: {total_batch}")
        return per_gpu_batch
    else:
        print(f"✓ 使用单GPU，batch size: {base_batch_size}")
        return base_batch_size


def check_gpu_memory(verbose=True):
    """
    检查GPU显存使用情况
    
    Args:
        verbose: 是否打印详细信息
    
    Returns:
        dict: GPU显存信息
    """
    if not torch.cuda.is_available():
        if verbose:
            print("⚠️ CUDA不可用")
        return {}
    
    memory_info = {}
    num_gpus = torch.cuda.device_count()
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
        total = props.total_memory / 1024**3  # GB
        free = total - reserved
        
        memory_info[i] = {
            'name': props.name,
            'total': total,
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'usage_percent': (reserved / total) * 100
        }
        
        if verbose:
            print(f"GPU {i} ({props.name}):")
            print(f"  总计: {total:.2f}GB")
            print(f"  已分配: {allocated:.2f}GB")
            print(f"  已保留: {reserved:.2f}GB")
            print(f"  可用: {free:.2f}GB")
            print(f"  使用率: {memory_info[i]['usage_percent']:.1f}%")
    
    return memory_info


def estimate_memory_usage(model, batch_size, num_gpus=1, use_cache=True, 
                         n_users=6000, n_items=4000, embed_dim=64):
    """
    估算训练时的显存使用
    
    Args:
        model: 模型对象
        batch_size: 总batch size
        num_gpus: GPU数量
        use_cache: 是否使用缓存模式
        n_users: 用户数量
        n_items: 物品数量
        embed_dim: 嵌入维度
    
    Returns:
        dict: 显存使用估算（GB）
    """
    per_gpu_batch = batch_size // num_gpus if num_gpus > 1 else batch_size
    
    # 模型参数（每个GPU）
    model_params = sum(p.numel() for p in model.parameters()) * 4 / 1024**3  # GB
    
    # 缓存embeddings（如果使用缓存模式）
    cache_memory = 0
    if use_cache:
        user_cache = n_users * embed_dim * 4 / 1024**3  # GB
        item_cache = n_items * embed_dim * 4 / 1024**3  # GB
        cache_memory = user_cache + item_cache
    
    # 激活值（每个GPU）
    # 粗略估算：batch_size * embed_dim * 10（包括中间层）
    activation_memory = per_gpu_batch * embed_dim * 10 * 4 / 1024**3  # GB
    
    # 梯度（与参数相同）
    gradient_memory = model_params
    
    # 优化器状态（Adam需要2倍参数空间）
    optimizer_memory = model_params * 2
    
    total_memory = model_params + cache_memory + activation_memory + gradient_memory + optimizer_memory
    
    return {
        'model_params': model_params,
        'cache_memory': cache_memory,
        'activation_memory': activation_memory,
        'gradient_memory': gradient_memory,
        'optimizer_memory': optimizer_memory,
        'total_per_gpu': total_memory,
        'total_all_gpus': total_memory * num_gpus
    }


def suggest_batch_size(model, num_gpus=None, target_memory_gb=8, 
                       use_cache=True, n_users=6000, n_items=4000, embed_dim=64):
    """
    根据目标显存使用量建议batch size
    
    Args:
        model: 模型对象
        num_gpus: GPU数量
        target_memory_gb: 目标显存使用量（GB）
        use_cache: 是否使用缓存模式
        n_users: 用户数量
        n_items: 物品数量
        embed_dim: 嵌入维度
    
    Returns:
        int: 建议的batch size（每个GPU）
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # 估算固定部分的内存
    memory_est = estimate_memory_usage(model, batch_size=1, num_gpus=num_gpus,
                                     use_cache=use_cache, n_users=n_users,
                                     n_items=n_items, embed_dim=embed_dim)
    
    fixed_memory = (memory_est['model_params'] + memory_est['cache_memory'] + 
                   memory_est['gradient_memory'] + memory_est['optimizer_memory'])
    
    # 计算可用于激活值的内存
    available_memory = target_memory_gb - fixed_memory
    
    if available_memory <= 0:
        print(f"⚠️ 固定内存占用 ({fixed_memory:.2f}GB) 已超过目标 ({target_memory_gb}GB)")
        return 32  # 返回最小batch size
    
    # 估算每个样本的激活值内存
    activation_per_sample = embed_dim * 10 * 4 / 1024**3  # GB
    
    # 计算建议的batch size
    suggested_batch = int(available_memory / activation_per_sample)
    suggested_batch = max(32, min(suggested_batch, 2048))  # 限制在32-2048之间
    
    print(f"✓ 建议batch size（每个GPU）: {suggested_batch}")
    print(f"  固定内存: {fixed_memory:.2f}GB")
    print(f"  可用内存: {available_memory:.2f}GB")
    print(f"  预计总内存: {fixed_memory + suggested_batch * activation_per_sample:.2f}GB")
    
    return suggested_batch


def limit_gpu_memory(fraction=0.9):
    """
    限制GPU显存使用比例
    
    Args:
        fraction: 使用的显存比例（0-1）
    """
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            torch.cuda.set_per_process_memory_fraction(fraction, i)
            print(f"✓ GPU {i}: 限制显存使用为 {fraction*100:.0f}% "
                  f"({total_memory * fraction / 1024**3:.2f}GB)")

