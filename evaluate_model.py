#!/usr/bin/env python
# coding: utf-8
"""
独立的模型评估脚本
用于加载训练好的模型并按照标准流程评估

使用方法:
    python evaluate_model.py --model_path your_model.pth
    
或者指定更多参数:
    python evaluate_model.py \
        --model_path gpt2_recommender_enhanced.pth \
        --data_path /path/to/ml-1m \
        --device cuda \
        --embed_dim 64 \
        --k_list 10 20 50
"""

import torch
import json
import argparse
import os
import sys
from dataset import ML1MDataset
from gpt2_recommender_enhanced import GPT2RecommenderEnhanced
from evaluator import evaluate_model_standard


def check_device(device_str):
    """检查设备是否可用"""
    if device_str == 'cuda':
        if not torch.cuda.is_available():
            print("⚠️  警告: CUDA不可用，将使用CPU进行评估")
            return 'cpu'
    return device_str


def load_model_config(model_path):
    """尝试从模型路径推断配置（如果可能）"""
    # 这里可以扩展，比如从模型文件名或同目录的配置文件中读取
    # 目前返回None，使用默认值
    return None


def main():
    parser = argparse.ArgumentParser(
        description='评估训练好的推荐模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认参数评估
  python evaluate_model.py --model_path my_model.pth
  
  # 指定所有参数
  python evaluate_model.py \\
      --model_path gpt2_recommender_enhanced.pth \\
      --data_path /Users/zhuxuzhou/Downloads/ml-1m \\
      --device cuda \\
      --embed_dim 64 \\
      --use_cache \\
      --use_attention \\
      --k_list 10 20 50 \\
      --output results.json
        """
    )
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型权重文件路径（必需）')
    parser.add_argument('--data_path', type=str, default='/Users/zhuxuzhou/Downloads/ml-1m',
                       help='数据集路径（默认: /Users/zhuxuzhou/Downloads/ml-1m）')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='设备 (cuda/cpu，默认: cuda)')
    parser.add_argument('--embed_dim', type=int, default=64,
                       help='嵌入维度（默认: 64，需与训练时一致）')
    parser.add_argument('--use_cache', action='store_true', default=True,
                       help='是否使用缓存模式（默认: True，需与训练时一致）')
    parser.add_argument('--no_cache', dest='use_cache', action='store_false',
                       help='不使用缓存模式（动态模式）')
    parser.add_argument('--use_attention', action='store_true', default=True,
                       help='是否使用注意力机制（默认: True，需与训练时一致）')
    parser.add_argument('--no_attention', dest='use_attention', action='store_false',
                       help='不使用注意力机制')
    parser.add_argument('--k_list', type=int, nargs='+', default=[10, 20, 50],
                       help='评估的K值列表（默认: 10 20 50）')
    parser.add_argument('--output', type=str, default=None,
                       help='结果保存路径（默认: 自动生成，基于模型文件名）')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 找不到模型文件: {args.model_path}")
        print("请检查文件路径是否正确")
        sys.exit(1)
    
    # 检查数据路径是否存在
    if not os.path.exists(args.data_path):
        print(f"❌ 错误: 找不到数据路径: {args.data_path}")
        print("请检查数据路径是否正确")
        sys.exit(1)
    
    # 检查设备
    device = check_device(args.device)
    
    # 自动生成输出文件名
    if args.output is None:
        model_basename = os.path.splitext(os.path.basename(args.model_path))[0]
        args.output = f'{model_basename}_evaluation_results.json'
    
    # 打印配置信息
    print("="*70)
    print("模型评估脚本")
    print("="*70)
    print(f"模型路径:     {args.model_path}")
    print(f"数据路径:     {args.data_path}")
    print(f"设备:         {device}")
    print(f"嵌入维度:     {args.embed_dim}")
    print(f"缓存模式:     {args.use_cache}")
    print(f"注意力机制:   {args.use_attention}")
    print(f"K值列表:      {args.k_list}")
    print(f"结果保存到:   {args.output}")
    print("="*70)
    
    # 1. 加载数据集
    print("\n[步骤 1/4] 加载数据集...")
    try:
        dataset = ML1MDataset(args.data_path, split_ratio=[0.8, 0.1, 0.1])
        train_data = dataset.get_split_data('train')
        valid_data = dataset.get_split_data('validation')
        test_data = dataset.get_split_data('test')
        print(f"  ✓ 数据集加载成功")
        print(f"    用户数: {dataset.get_user_num()}")
        print(f"    物品数: {dataset.get_item_num()}")
        print(f"    训练集: {len(train_data):,} 条")
        print(f"    验证集: {len(valid_data):,} 条")
        print(f"    测试集: {len(test_data):,} 条")
    except Exception as e:
        print(f"  ✗ 错误: 加载数据集失败")
        print(f"  错误信息: {e}")
        sys.exit(1)
    
    # 2. 创建模型（与训练时相同的配置）
    print("\n[步骤 2/4] 创建模型...")
    try:
        model = GPT2RecommenderEnhanced(
            n_users=dataset.get_user_num(),
            n_items=dataset.get_item_num(),
            embed_dim=args.embed_dim,
            dataset=dataset,
            use_cache=args.use_cache,
            use_attention=args.use_attention,
            freeze_gpt2=False  # 评估时不需要训练
        )
        print(f"  ✓ 模型创建成功")
        print(f"    使用缓存模式: {args.use_cache}")
        print(f"    使用注意力机制: {args.use_attention}")
    except Exception as e:
        print(f"  ✗ 错误: 创建模型失败")
        print(f"  错误信息: {e}")
        sys.exit(1)
    
    # 3. 加载模型权重
    print(f"\n[步骤 3/4] 加载模型权重...")
    print(f"  文件: {args.model_path}")
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        
        # 处理可能的DataParallel保存格式
        if any(key.startswith('module.') for key in state_dict.keys()):
            print("  检测到DataParallel格式，正在转换...")
            state_dict = {k[7:] if k.startswith('module.') else k: v 
                         for k, v in state_dict.items()}
        
        # 尝试加载模型权重
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"  ⚠️  警告: 以下参数未加载: {missing_keys[:5]}...")
            if len(missing_keys) > 5:
                print(f"    （共 {len(missing_keys)} 个参数）")
        
        if unexpected_keys:
            print(f"  ⚠️  警告: 以下参数在模型中不存在: {unexpected_keys[:5]}...")
            if len(unexpected_keys) > 5:
                print(f"    （共 {len(unexpected_keys)} 个参数）")
        
        if not missing_keys and not unexpected_keys:
            print("  ✓ 模型权重加载成功（完全匹配）")
        else:
            print("  ✓ 模型权重加载完成（部分匹配）")
            
    except FileNotFoundError:
        print(f"  ✗ 错误: 找不到模型文件 {args.model_path}")
        print("  请确保模型文件存在，或使用 --model_path 指定正确的路径")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ 错误: 加载模型失败")
        print(f"  错误信息: {e}")
        print("\n可能的原因:")
        print("  1. 模型文件损坏")
        print("  2. 模型配置参数不匹配（embed_dim, use_cache, use_attention等）")
        print("  3. 模型架构与训练时不一致")
        sys.exit(1)
    
    # 将模型移到指定设备
    model = model.to(device)
    model.eval()
    print(f"  ✓ 模型已移至设备: {device}")
    
    # 4. 执行标准评估
    print(f"\n[步骤 4/4] 执行标准评估...")
    print("\n评估标准：")
    print("  1. 对每个用户，只考虑未在训练集+验证集中交互过的item")
    print("  2. 使用模型的predict方法计算user-item分数")
    print("  3. 排序得到top K推荐")
    print("  4. 计算Recall, Precision, NDCG等指标")
    print()
    
    try:
        results = evaluate_model_standard(
            model=model,
            dataset=dataset,
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            device=device,
            k_list=args.k_list,
            verbose=True
        )
    except Exception as e:
        print(f"\n  ✗ 错误: 评估过程失败")
        print(f"  错误信息: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 5. 保存结果
    print(f"\n保存评估结果...")
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({k: float(v) for k, v in results.items()}, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 结果已保存到: {args.output}")
    except Exception as e:
        print(f"  ✗ 错误: 保存结果失败")
        print(f"  错误信息: {e}")
    
    # 打印总结
    print("\n" + "="*70)
    print("评估完成！")
    print("="*70)
    print(f"\n评估结果摘要:")
    for k in args.k_list:
        recall_key = f'Recall@{k}'
        precision_key = f'Precision@{k}'
        ndcg_key = f'NDCG@{k}'
        if recall_key in results:
            print(f"  Top-{k}:")
            print(f"    Recall@{k}:    {results[recall_key]:.4f}")
            if precision_key in results:
                print(f"    Precision@{k}: {results[precision_key]:.4f}")
            if ndcg_key in results:
                print(f"    NDCG@{k}:      {results[ndcg_key]:.4f}")
    print(f"\n详细结果已保存到: {args.output}")
    print("="*70)


if __name__ == '__main__':
    main()

