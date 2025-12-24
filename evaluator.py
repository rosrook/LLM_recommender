# coding: utf-8
"""
测试集评估模块
按照标准评估流程：
1. 对每个用户，找到未交互过的item（训练集+验证集中未交互的）
2. 计算user/item embedding的预测分数（使用模型的predict方法）
3. 排序得到top K推荐
4. 计算Recall, Precision, NDCG等评价指标
"""

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict


class StandardEvaluator:
    """标准评估器，按照用户指定的评估标准"""
    
    def __init__(self, model, dataset, train_data, valid_data, test_data, 
                 device='cuda', k_list=[10, 20, 50]):
        """
        初始化评估器
        
        Args:
            model: 训练好的模型
            dataset: ML1MDataset对象
            train_data: 训练集数据（DataFrame）
            valid_data: 验证集数据（DataFrame）
            test_data: 测试集数据（DataFrame）
            device: 设备
            k_list: 要评估的K值列表，如[10, 20, 50]
        """
        self.model = model
        self.dataset = dataset
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.device = device
        self.k_list = k_list
        
        # 如果使用DataParallel，需要访问module
        if hasattr(model, 'module'):
            self.model_for_eval = model.module
        else:
            self.model_for_eval = model
        
        self.model_for_eval.eval()
        
        # 构建用户交互历史（训练集+验证集）
        print("构建用户交互历史...")
        self._build_user_interaction_history()
        
        # 获取测试集中的用户和正样本
        print("构建测试集正样本...")
        self._build_test_positive_items()
    
    def _build_user_interaction_history(self):
        """构建每个用户在训练集+验证集中的交互历史"""
        # 合并训练集和验证集
        import pandas as pd
        train_valid_data = pd.concat([self.train_data, self.valid_data], ignore_index=True)
        
        # 为每个用户记录交互过的item集合
        self.user_interacted_items = defaultdict(set)
        
        for _, row in train_valid_data.iterrows():
            user_id = row['user_id']
            item_id = row['movie_id']
            self.user_interacted_items[user_id].add(item_id)
        
        print(f"  构建了 {len(self.user_interacted_items)} 个用户的交互历史")
    
    def _build_test_positive_items(self):
        """构建测试集中每个用户的正样本（真实交互的item）"""
        self.test_positive_items = defaultdict(set)
        
        for _, row in self.test_data.iterrows():
            user_id = row['user_id']
            item_id = row['movie_id']
            self.test_positive_items[user_id].add(item_id)
        
        print(f"  测试集中有 {len(self.test_positive_items)} 个用户有交互")
    
    def get_candidate_items(self, user_id):
        """
        获取用户的候选item（未在训练集+验证集中交互过的item）
        
        Args:
            user_id: 用户ID
        
        Returns:
            torch.Tensor: 候选item的ID列表
        """
        interacted_items = self.user_interacted_items.get(user_id, set())
        all_items = set(range(self.dataset.get_item_num()))
        candidate_items = all_items - interacted_items
        
        return torch.tensor(list(candidate_items), dtype=torch.long, device=self.device)
    
    def predict_scores_for_user(self, user_id, candidate_items, batch_size=1024):
        """
        对用户和候选item计算预测分数
        
        Args:
            user_id: 用户ID
            candidate_items: 候选item的tensor
            batch_size: 批处理大小
        
        Returns:
            torch.Tensor: 预测分数，shape=(len(candidate_items),)
        """
        user_tensor = torch.full((len(candidate_items),), user_id, 
                                dtype=torch.long, device=self.device)
        
        # 分批计算，避免显存不足
        scores = []
        for i in range(0, len(candidate_items), batch_size):
            end_idx = min(i + batch_size, len(candidate_items))
            batch_items = candidate_items[i:end_idx]
            batch_users = user_tensor[i:end_idx]
            
            # 使用模型的predict方法计算分数
            batch_scores = self.model_for_eval.predict(batch_users, batch_items)
            scores.append(batch_scores)
        
        return torch.cat(scores, dim=0)
    
    def evaluate_user(self, user_id, k_list=None):
        """
        评估单个用户
        
        Args:
            user_id: 用户ID
            k_list: K值列表，如果为None则使用self.k_list
        
        Returns:
            dict: 评估指标
        """
        if k_list is None:
            k_list = self.k_list
        
        # 获取候选item（未交互过的）
        candidate_items = self.get_candidate_items(user_id)
        
        if len(candidate_items) == 0:
            # 如果没有候选item，返回0
            return {f'Recall@{k}': 0.0 for k in k_list}
        
        # 获取测试集中的正样本
        positive_items = self.test_positive_items.get(user_id, set())
        
        if len(positive_items) == 0:
            # 如果测试集中没有正样本，返回0
            return {f'Recall@{k}': 0.0 for k in k_list}
        
        # 计算预测分数
        scores = self.predict_scores_for_user(user_id, candidate_items)
        
        # 排序，获取top K推荐
        _, top_indices = torch.topk(scores, k=min(max(k_list), len(candidate_items)))
        top_items = candidate_items[top_indices].cpu().numpy()
        
        # 计算指标
        metrics = {}
        for k in k_list:
            if k > len(top_items):
                k = len(top_items)
            
            top_k_items = set(top_items[:k])
            
            # Recall@K: 推荐的K个item中有多少个是正样本
            recall = len(top_k_items & positive_items) / len(positive_items) if positive_items else 0.0
            metrics[f'Recall@{k}'] = recall
            
            # Precision@K: 推荐的K个item中有多少个是正样本
            precision = len(top_k_items & positive_items) / k if k > 0 else 0.0
            metrics[f'Precision@{k}'] = precision
            
            # NDCG@K
            ndcg = self._calculate_ndcg(top_k_items, positive_items, k)
            metrics[f'NDCG@{k}'] = ndcg
        
        return metrics
    
    def _calculate_ndcg(self, recommended_items, positive_items, k):
        """
        计算NDCG@K
        
        Args:
            recommended_items: 推荐的item集合
            positive_items: 正样本item集合
            k: K值
        
        Returns:
            float: NDCG@K值
        """
        # 构建hit列表（推荐列表中哪些是正样本）
        hit_list = [1 if item in positive_items else 0 for item in list(recommended_items)[:k]]
        
        if not any(hit_list):
            return 0.0
        
        # 计算DCG
        dcg = sum(hit / np.log2(idx + 2) for idx, hit in enumerate(hit_list))
        
        # 计算IDCG（理想情况下的DCG）
        num_positives = min(k, len(positive_items))
        idcg = sum(1.0 / np.log2(idx + 2) for idx in range(num_positives))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate(self, verbose=True):
        """
        在测试集上评估所有用户
        
        Args:
            verbose: 是否显示进度条
        
        Returns:
            dict: 所有用户的平均评估指标
        """
        # 获取测试集中的所有用户
        test_users = self.test_data['user_id'].unique()
        
        # 初始化指标累加器
        all_metrics = defaultdict(list)
        
        # 评估每个用户
        if verbose:
            iterator = tqdm(test_users, desc="评估中")
        else:
            iterator = test_users
        
        for user_id in iterator:
            metrics = self.evaluate_user(user_id)
            for metric_name, value in metrics.items():
                all_metrics[metric_name].append(value)
        
        # 计算平均值
        result = {
            metric_name: np.mean(values) 
            for metric_name, values in all_metrics.items()
        }
        
        return result
    
    def print_results(self, results):
        """打印评估结果"""
        print("\n" + "="*60)
        print("测试集评估结果（标准评估流程）")
        print("="*60)
        
        # 按K值分组显示
        for k in self.k_list:
            print(f"\nTop-{k} 推荐结果:")
            recall_key = f'Recall@{k}'
            precision_key = f'Precision@{k}'
            ndcg_key = f'NDCG@{k}'
            
            if recall_key in results:
                print(f"  Recall@{k}:    {results[recall_key]:.4f}")
            if precision_key in results:
                print(f"  Precision@{k}: {results[precision_key]:.4f}")
            if ndcg_key in results:
                print(f"  NDCG@{k}:      {results[ndcg_key]:.4f}")
        
        print("\n" + "="*60)


def evaluate_model_standard(model, dataset, train_data, valid_data, test_data,
                            device='cuda', k_list=[10, 20, 50], verbose=True):
    """
    便捷函数：按照标准流程评估模型
    
    Args:
        model: 训练好的模型
        dataset: ML1MDataset对象
        train_data: 训练集数据
        valid_data: 验证集数据
        test_data: 测试集数据
        device: 设备
        k_list: K值列表
        verbose: 是否显示详细信息
    
    Returns:
        dict: 评估结果
    """
    evaluator = StandardEvaluator(
        model=model,
        dataset=dataset,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        device=device,
        k_list=k_list
    )
    
    results = evaluator.evaluate(verbose=verbose)
    
    if verbose:
        evaluator.print_results(results)
    
    return results

