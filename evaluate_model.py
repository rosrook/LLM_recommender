#!/usr/bin/env python
# coding: utf-8
"""
独立的模型评估脚本
用于加载训练好的模型并按照标准流程评估
"""

import torch
import json
import argparse
from dataset import ML1MDataset
from gpt2_recommender_enhanced import GPT2RecommenderEnhanced
from evaluator import evaluate_model_standard


def main():
    parser = argparse.ArgumentParser(description='评估训练好的模型')
    parser.add_argument('--model_path', type=str, default='gpt2_recommender_enhanced.pth',
                       help='模型权重文件路径')
    parser.add_argument('--data_path', type=str, default='/Users/zhuxuzhou/Downloads/ml-1m',
                       help='数据集路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--embed_dim', type=int, default=64,
                       help='嵌入维度')
    parser.add_argument('--use_cache', action='store_true', default=True,
                       help='是否使用缓存模式')
    parser.add_argument('--use_attention', action='store_true', default=True,
                       help='是否使用注意力机制')
    parser.add_argument('--k_list', type=int, nargs='+', default=[10, 20, 50],
                       help='评估的K值列表，如: --k_list 10 20 50')
    parser.add_argument('--output', type=str, default='test_results_standard.json',
                       help='结果保存路径')
    
    args = parser.parse_args()
    
    print("="*60)
    print("模型评估脚本")
    print("="*60)
    print(f"模型路径: {args.model_path}")
    print(f"数据路径: {args.data_path}")
    print(f"设备: {args.device}")
    print(f"K值列表: {args.k_list}")
    print("="*60)
    
    # 1. 加载数据集
    print("\n[1/4] 加载数据集...")
    dataset = ML1MDataset(args.data_path, split_ratio=[0.8, 0.1, 0.1])
    train_data = dataset.get_split_data('train')
    valid_data = dataset.get_split_data('validation')
    test_data = dataset.get_split_data('test')
    print(f"  用户数: {dataset.get_user_num()}")
    print(f"  物品数: {dataset.get_item_num()}")
    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(valid_data)} 条")
    print(f"  测试集: {len(test_data)} 条")
    
    # 2. 创建模型（与训练时相同的配置）
    print("\n[2/4] 创建模型...")
    model = GPT2RecommenderEnhanced(
        n_users=dataset.get_user_num(),
        n_items=dataset.get_item_num(),
        embed_dim=args.embed_dim,
        dataset=dataset,
        use_cache=args.use_cache,
        use_attention=args.use_attention,
        freeze_gpt2=False  # 评估时不需要训练
    )
    print(f"  使用缓存模式: {args.use_cache}")
    print(f"  使用注意力机制: {args.use_attention}")
    
    # 3. 加载模型权重
    print(f"\n[3/4] 加载模型权重: {args.model_path}")
    try:
        state_dict = torch.load(args.model_path, map_location=args.device)
        
        # 处理可能的DataParallel保存格式
        if any(key.startswith('module.') for key in state_dict.keys()):
            # 如果保存时使用了DataParallel，需要去掉module.前缀
            state_dict = {k[7:] if k.startswith('module.') else k: v 
                         for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        print("  ✓ 模型加载成功")
    except FileNotFoundError:
        print(f"  ✗ 错误: 找不到模型文件 {args.model_path}")
        print("  请确保模型文件存在，或使用 --model_path 指定正确的路径")
        return
    except Exception as e:
        print(f"  ✗ 错误: 加载模型失败")
        print(f"  错误信息: {e}")
        return
    
    # 将模型移到指定设备
    model = model.to(args.device)
    model.eval()
    
    # 4. 执行标准评估
    print(f"\n[4/4] 执行标准评估...")
    print("评估标准：")
    print("  1. 对每个用户，只考虑未在训练集+验证集中交互过的item")
    print("  2. 使用模型的predict方法计算user-item分数")
    print("  3. 排序得到top K推荐")
    print("  4. 计算Recall, Precision, NDCG等指标")
    print()
    
    results = evaluate_model_standard(
        model=model,
        dataset=dataset,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        device=args.device,
        k_list=args.k_list,
        verbose=True
    )
    
    # 5. 保存结果
    print(f"\n保存评估结果到: {args.output}")
    with open(args.output, 'w') as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
    print("  ✓ 结果已保存")
    
    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)


if __name__ == '__main__':
    main()

