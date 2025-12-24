# coding: utf-8
"""
Example script for training GPT-2 based recommender system
"""

from dataset import ML1MDataset
from dataloader import TrainDataLoader, EvalDataLoader
from recommender import GPT2Recommender
from trainer import Trainer


def main():
    # Load dataset
    print("Loading ML-1M dataset...")
    dataset = ML1MDataset('/Users/zhuxuzhou/Downloads/ml-1m', split_ratio=[0.8, 0.1, 0.1])
    print(dataset)
    
    # Get split datasets
    train_data = dataset.get_split_data('train')
    valid_data = dataset.get_split_data('validation')
    test_data = dataset.get_split_data('test')
    
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Validation samples: {len(valid_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create GPT-2 based recommender model
    print("\nInitializing GPT-2 Recommender...")
    model = GPT2Recommender(
        n_users=dataset.get_user_num(),
        n_items=dataset.get_item_num(),
        embed_dim=64,
        dataset=dataset,
        gpt2_model_name='gpt2',
        use_peft=True,  # Use PEFT for efficient fine-tuning
        freeze_gpt2=False  # Set to True if you want to freeze GPT-2 and only train the recommendation head
    )
    
    # Create dataloaders
    print("\nCreating data loaders...")
    train_loader = TrainDataLoader(train_data, batch_size=2048, shuffle=True, device='cuda')
    valid_loader = EvalDataLoader(valid_data, train_data, batch_size=2048, device='cuda')
    test_loader = EvalDataLoader(test_data, train_data, batch_size=2048, device='cuda')
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_data=train_loader,
        eval_data=valid_loader,
        test_data=test_loader,
        device='cuda',
        epochs=50,
        batch_size=2048,
        optimizer='adam',
        lr=1e-3,
        weight_decay=1e-5,
        eval_step=1,
        early_stop_patience=10
    )
    
    # Train model
    print("\nStarting training...")
    valid_result, test_result = trainer.fit(save_model=True, model_path='gpt2_recommender.pth')
    
    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"{'='*50}")
    print(f"\nBest Validation Result:")
    for metric, value in valid_result.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nTest Result:")
    for metric, value in test_result.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()

