# coding: utf-8
"""
Example script for training Enhanced GPT-2 based recommender system
Demonstrates different training strategies
"""

from dataset import ML1MDataset
from dataloader import TrainDataLoader, EvalDataLoader
from gpt2_recommender_enhanced import GPT2RecommenderEnhanced
from trainer import Trainer


def main_strategy1_fast():
    """Strategy 1: Fast prototyping with caching"""
    print("="*60)
    print("Strategy 1: Fast Prototyping (Cached Mode)")
    print("="*60)
    
    dataset = ML1MDataset('/Users/zhuxuzhou/Downloads/ml-1m', split_ratio=[0.8, 0.1, 0.1])
    train_data = dataset.get_split_data('train')
    valid_data = dataset.get_split_data('validation')
    test_data = dataset.get_split_data('test')
    
    # Fast mode: use cache, freeze GPT-2
    model = GPT2RecommenderEnhanced(
        n_users=dataset.get_user_num(),
        n_items=dataset.get_item_num(),
        embed_dim=64,
        dataset=dataset,
        use_cache=True,  # Use cached embeddings
        use_attention=False,  # Simple concatenation
        freeze_gpt2=True,  # Freeze GPT-2
        use_peft=False
    )
    
    train_loader = TrainDataLoader(train_data, batch_size=2048, shuffle=True, device='cuda')
    valid_loader = EvalDataLoader(valid_data, train_data, batch_size=2048, device='cuda')
    test_loader = EvalDataLoader(test_data, train_data, batch_size=2048, device='cuda')
    
    trainer = Trainer(
        model=model,
        train_data=train_loader,
        eval_data=valid_loader,
        test_data=test_loader,
        device='cuda',
        epochs=20,
        lr=1e-3,
        early_stop_patience=5
    )
    
    valid_result, test_result = trainer.fit(save_model=True, model_path='enhanced_fast.pth')
    print("\nFast Mode Results:")
    print(f"Validation: {valid_result}")
    print(f"Test: {test_result}")


def main_strategy2_end2end():
    """Strategy 2: End-to-end fine-tuning (Recommended)"""
    print("="*60)
    print("Strategy 2: End-to-End Fine-tuning (Recommended)")
    print("="*60)
    
    dataset = ML1MDataset('/Users/zhuxuzhou/Downloads/ml-1m', split_ratio=[0.8, 0.1, 0.1])
    train_data = dataset.get_split_data('train')
    valid_data = dataset.get_split_data('validation')
    test_data = dataset.get_split_data('test')
    
    # End-to-end mode: dynamic extraction, attention fusion, fine-tune GPT-2
    model = GPT2RecommenderEnhanced(
        n_users=dataset.get_user_num(),
        n_items=dataset.get_item_num(),
        embed_dim=64,
        dataset=dataset,
        use_cache=False,  # Dynamic extraction
        use_attention=True,  # Cross-attention fusion
        freeze_gpt2=False,  # Fine-tune GPT-2
        use_peft=True,  # Use PEFT for efficient fine-tuning
        contrastive_weight=0.1  # Add contrastive learning
    )
    
    train_loader = TrainDataLoader(train_data, batch_size=1024, shuffle=True, device='cuda')  # Smaller batch for dynamic mode
    valid_loader = EvalDataLoader(valid_data, train_data, batch_size=1024, device='cuda')
    test_loader = EvalDataLoader(test_data, train_data, batch_size=1024, device='cuda')
    
    trainer = Trainer(
        model=model,
        train_data=train_loader,
        eval_data=valid_loader,
        test_data=test_loader,
        device='cuda',
        epochs=50,
        lr=5e-4,  # Lower learning rate for fine-tuning
        weight_decay=1e-5,
        early_stop_patience=10
    )
    
    valid_result, test_result = trainer.fit(save_model=True, model_path='enhanced_end2end.pth')
    print("\nEnd-to-End Mode Results:")
    print(f"Validation: {valid_result}")
    print(f"Test: {test_result}")


def main_strategy3_hybrid():
    """Strategy 3: Hybrid approach (two-stage training)"""
    print("="*60)
    print("Strategy 3: Hybrid Approach (Two-Stage Training)")
    print("="*60)
    
    dataset = ML1MDataset('/Users/zhuxuzhou/Downloads/ml-1m', split_ratio=[0.8, 0.1, 0.1])
    train_data = dataset.get_split_data('train')
    valid_data = dataset.get_split_data('validation')
    test_data = dataset.get_split_data('test')
    
    # Stage 1: Fast training with cache
    print("\n--- Stage 1: Fast Training with Cache ---")
    model = GPT2RecommenderEnhanced(
        n_users=dataset.get_user_num(),
        n_items=dataset.get_item_num(),
        embed_dim=64,
        dataset=dataset,
        use_cache=True,
        use_attention=True,
        freeze_gpt2=True
    )
    
    train_loader = TrainDataLoader(train_data, batch_size=2048, shuffle=True, device='cuda')
    valid_loader = EvalDataLoader(valid_data, train_data, batch_size=2048, device='cuda')
    test_loader = EvalDataLoader(test_data, train_data, batch_size=2048, device='cuda')
    
    trainer = Trainer(
        model=model,
        train_data=train_loader,
        eval_data=valid_loader,
        test_data=test_loader,
        device='cuda',
        epochs=15,
        lr=1e-3,
        early_stop_patience=5
    )
    
    trainer.fit(save_model=False)
    print(f"Stage 1 Best Validation: {trainer.best_valid_result}")
    
    # Stage 2: Fine-tuning with dynamic extraction
    print("\n--- Stage 2: Fine-tuning with Dynamic Extraction ---")
    model.use_cache = False  # Switch to dynamic mode
    model.freeze_gpt2 = False  # Unfreeze GPT-2
    # Re-enable gradients for GPT-2
    for param in model.gpt2_encoder.parameters():
        param.requires_grad = True
    
    trainer = Trainer(
        model=model,
        train_data=train_loader,
        eval_data=valid_loader,
        test_data=test_loader,
        device='cuda',
        epochs=30,
        lr=5e-4,  # Lower learning rate
        early_stop_patience=10
    )
    
    valid_result, test_result = trainer.fit(save_model=True, model_path='enhanced_hybrid.pth')
    print("\nHybrid Mode Final Results:")
    print(f"Validation: {valid_result}")
    print(f"Test: {test_result}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        strategy = sys.argv[1]
        if strategy == '1':
            main_strategy1_fast()
        elif strategy == '2':
            main_strategy2_end2end()
        elif strategy == '3':
            main_strategy3_hybrid()
        else:
            print("Usage: python example_enhanced.py [1|2|3]")
            print("  1: Fast prototyping")
            print("  2: End-to-end fine-tuning (recommended)")
            print("  3: Hybrid approach")
    else:
        # Run recommended strategy by default
        print("Running recommended strategy (End-to-End Fine-tuning)...")
        print("Use 'python example_enhanced.py [1|2|3]' to choose different strategies\n")
        main_strategy2_end2end()

