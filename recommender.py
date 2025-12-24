import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from gpt2_encoder import GPT2MetadataEncoder, format_user_metadata, format_item_metadata


class AbstractRecommender(nn.Module, ABC):
    """Abstract base class for recommender models"""

    def __init__(self, n_users, n_items, embed_dim):
        """
        Initialize base recommender

        Args:
            n_users (int): Number of users in the dataset
            n_items (int): Number of items in the dataset
            embed_dim (int): Dimension of embeddings
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim

        # Initialize user and item embeddings
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        # Initialize embeddings with Xavier uniform
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    @abstractmethod
    def forward(self, batch_data):
        """
        Forward pass to compute prediction scores

        Args:
            batch_data (torch.Tensor): Batch data from dataloader containing [users, pos_items, neg_items]

        Returns:
            tuple: (pos_scores, neg_scores) predicted scores for positive and negative samples
        """
        pass

    @abstractmethod
    def calculate_loss(self, pos_scores, neg_scores):
        """
        Calculate loss for training

        Args:
            pos_scores (torch.FloatTensor): Predicted scores for positive samples
            neg_scores (torch.FloatTensor): Predicted scores for negative samples

        Returns:
            torch.FloatTensor: Computed loss value
        """
        pass

    @torch.no_grad()
    def recommend(self, user_id, k=None):
        """
        Generate item recommendations for a user

        Args:
            user_id (int): User ID to generate recommendations for
            k (int, optional): Number of items to recommend. If None, returns scores for all items

        Returns:
            torch.FloatTensor: Predicted scores for items (shape: n_items)
        """
        self.eval()
        user_tensor = torch.LongTensor([user_id]).to(self.device)
        all_items = torch.arange(self.n_items).to(self.device)
        # Get scores for all items
        scores = self.predict(user_tensor.repeat(len(all_items)), all_items)

        if k is not None:
            _, indices = torch.topk(scores, k)
            return all_items[indices]

        return scores

    def predict(self, user_ids, item_ids):
        """
        Predict scores for given user-item pairs

        Args:
            user_ids (torch.LongTensor): User IDs
            item_ids (torch.LongTensor): Item IDs

        Returns:
            torch.FloatTensor: Predicted scores
        """
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)

        return (user_embeds * item_embeds).sum(dim=-1)

    def get_user_embedding(self, user_id):
        """Get embedding for a user"""
        return self.user_embedding(torch.LongTensor([user_id]).to(self.device))

    def get_item_embedding(self, item_id):
        """Get embedding for an item"""
        return self.item_embedding(torch.LongTensor([item_id]).to(self.device))

    @property
    def device(self):
        """Get device model is on"""
        return next(self.parameters()).device


class NCFRecommender(AbstractRecommender):
    def __init__(self, n_users, n_items, embed_dim):
        super().__init__(n_users, n_items, embed_dim)
        self.mlp_layers = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, batch_data):
        user_ids, pos_item_ids, neg_item_ids = batch_data
        user_embeds = self.user_embedding(user_ids)
        pos_item_embeds = self.item_embedding(pos_item_ids)
        neg_item_embeds = self.item_embedding(neg_item_ids)

        concated_embeds = torch.cat([user_embeds, pos_item_embeds], dim=-1)
        concated_embeds_neg = torch.cat([user_embeds, neg_item_embeds], dim=-1)
        mlp_output = self.mlp_layers(concated_embeds)
        mlp_output_neg = self.mlp_layers(concated_embeds_neg)

        return mlp_output, mlp_output_neg

    def calculate_loss(self, pos_scores, neg_scores):
        return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

    @torch.no_grad()
    def recommend(self, user_id, k=None, batch_size_eval=1024):

        self.eval()
        device = self.device

        # get user index as int
        if isinstance(user_id, torch.Tensor):
            user_idx = int(user_id.detach().cpu().item())
        else:
            user_idx = int(user_id)

        # accumulate scores in chunks
        scores_chunks = []
        start = 0
        while start < self.n_items:
            end = min(start + batch_size_eval, self.n_items)
            items = torch.arange(start, end, device=device, dtype=torch.long)
            users = torch.full((len(items),), user_idx, device=device, dtype=torch.long)

            user_embeds = self.user_embedding(users)  # (batch, dim)
            item_embeds = self.item_embedding(items)  # (batch, dim)
            concated_embeds = torch.cat([user_embeds, item_embeds], dim=-1)  # (batch, 2*dim)
            output = self.mlp_layers(concated_embeds).view(-1)  # (batch,)
            scores_chunks.append(output)
            start = end

        scores = torch.cat(scores_chunks, dim=0)  # (n_items,)

        if k is None:
            return scores
        else:
            _, idxs = torch.topk(scores, k)
            return idxs


class GPT2Recommender(AbstractRecommender):
    """
    Recommender model that combines ID embeddings with GPT-2 encoded metadata embeddings
    """
    
    def __init__(self, n_users, n_items, embed_dim, dataset, gpt2_model_name='gpt2', 
                 use_peft=True, metadata_dim=None, freeze_gpt2=False):
        """
        Initialize GPT-2 based recommender
        
        Args:
            n_users (int): Number of users
            n_items (int): Number of items
            embed_dim (int): Dimension of ID embeddings
            dataset: ML1MDataset object to access metadata
            gpt2_model_name (str): GPT-2 model name
            use_peft (bool): Whether to use PEFT for GPT-2 fine-tuning
            metadata_dim (int): Dimension of metadata embeddings. If None, uses GPT-2 default (768)
            freeze_gpt2 (bool): Whether to freeze GPT-2 parameters (only train recommendation head)
        """
        super().__init__(n_users, n_items, embed_dim)
        
        self.dataset = dataset
        self.freeze_gpt2 = freeze_gpt2
        
        # Initialize GPT-2 encoder (will be moved to device later)
        # Use 'cuda' as default, will be moved to correct device when model.to(device) is called
        self.gpt2_encoder = GPT2MetadataEncoder(
            model_name=gpt2_model_name,
            use_peft=use_peft,
            device='cuda'  # Will be updated when model is moved to device
        )
        
        # GPT-2 embedding dimension
        self.metadata_dim = metadata_dim if metadata_dim else self.gpt2_encoder.gpt2_dim
        
        # Precompute metadata embeddings for all users and items
        # Note: These will be computed on CPU first, then moved to device
        print("Precomputing user metadata embeddings...")
        self.user_metadata_embeddings = self._precompute_user_metadata_embeddings()
        
        print("Precomputing item metadata embeddings...")
        self.item_metadata_embeddings = self._precompute_item_metadata_embeddings()
        
        # Projection layer to align metadata embedding dimension with embed_dim if needed
        if self.metadata_dim != embed_dim:
            self.user_meta_proj = nn.Linear(self.metadata_dim, embed_dim)
            self.item_meta_proj = nn.Linear(self.metadata_dim, embed_dim)
        else:
            self.user_meta_proj = nn.Identity()
            self.item_meta_proj = nn.Identity()
        
        # MLP layers for prediction
        # Input: concatenated [user_id_embed, user_meta_embed, item_id_embed, item_meta_embed]
        # Total input dim: embed_dim * 4
        self.mlp_layers = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Freeze GPT-2 if specified
        if freeze_gpt2:
            for param in self.gpt2_encoder.parameters():
                param.requires_grad = False
    
    def _precompute_user_metadata_embeddings(self):
        """Precompute GPT-2 embeddings for all user metadata"""
        user_meta_df = self.dataset.get_user_meta()
        user_texts = []
        
        for user_id in range(self.n_users):
            if user_id in user_meta_df.index:
                user_row = user_meta_df.loc[user_id]
                text = format_user_metadata(user_row)
            else:
                # Default text for users without metadata
                text = "User profile: Gender is Unknown, Age group is Unknown, Occupation is Unknown."
            user_texts.append(text)
        
        # Encode all user texts
        embeddings = self.gpt2_encoder.encode_text(user_texts)
        # Move to device (will be set correctly after model.to(device) is called)
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cuda'
        return embeddings.to(device)
    
    def _precompute_item_metadata_embeddings(self):
        """Precompute GPT-2 embeddings for all item metadata"""
        item_meta_df = self.dataset.get_item_meta()
        item_texts = []
        
        for item_id in range(self.n_items):
            if item_id in item_meta_df.index:
                item_row = item_meta_df.loc[item_id]
                text = format_item_metadata(item_row)
            else:
                # Default text for items without metadata
                text = "Movie: Unknown. Genres: Unknown."
            item_texts.append(text)
        
        # Encode all item texts
        embeddings = self.gpt2_encoder.encode_text(item_texts)
        # Move to device (will be set correctly after model.to(device) is called)
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cuda'
        return embeddings.to(device)
    
    def forward(self, batch_data):
        """
        Forward pass to compute prediction scores
        
        Args:
            batch_data (tuple): Batch data tuple containing (users, pos_items, neg_items)
                               Each element is a torch.Tensor of shape (batch_size,)
            
        Returns:
            tuple: (pos_scores, neg_scores) predicted scores for positive and negative samples
        """
        user_ids, pos_item_ids, neg_item_ids = batch_data
        device = self.device
        
        # Ensure metadata embeddings are on the correct device
        if self.user_metadata_embeddings.device != device:
            self.user_metadata_embeddings = self.user_metadata_embeddings.to(device)
        if self.item_metadata_embeddings.device != device:
            self.item_metadata_embeddings = self.item_metadata_embeddings.to(device)
        
        # Get ID embeddings
        user_id_embeds = self.user_embedding(user_ids)  # (batch_size, embed_dim)
        pos_item_id_embeds = self.item_embedding(pos_item_ids)  # (batch_size, embed_dim)
        neg_item_id_embeds = self.item_embedding(neg_item_ids)  # (batch_size, embed_dim)
        
        # Get metadata embeddings (precomputed)
        user_meta_embeds = self.user_metadata_embeddings[user_ids]  # (batch_size, metadata_dim)
        pos_item_meta_embeds = self.item_metadata_embeddings[pos_item_ids]  # (batch_size, metadata_dim)
        neg_item_meta_embeds = self.item_metadata_embeddings[neg_item_ids]  # (batch_size, metadata_dim)
        
        # Project metadata embeddings to embed_dim if needed
        user_meta_embeds = self.user_meta_proj(user_meta_embeds)  # (batch_size, embed_dim)
        pos_item_meta_embeds = self.item_meta_proj(pos_item_meta_embeds)  # (batch_size, embed_dim)
        neg_item_meta_embeds = self.item_meta_proj(neg_item_meta_embeds)  # (batch_size, embed_dim)
        
        # Concatenate ID and metadata embeddings
        pos_concat = torch.cat([
            user_id_embeds,
            user_meta_embeds,
            pos_item_id_embeds,
            pos_item_meta_embeds
        ], dim=-1)  # (batch_size, embed_dim * 4)
        
        neg_concat = torch.cat([
            user_id_embeds,
            user_meta_embeds,
            neg_item_id_embeds,
            neg_item_meta_embeds
        ], dim=-1)  # (batch_size, embed_dim * 4)
        
        # Pass through MLP to get scores (0-1 range)
        pos_scores = self.mlp_layers(pos_concat)  # (batch_size, 1)
        neg_scores = self.mlp_layers(neg_concat)  # (batch_size, 1)
        
        return pos_scores, neg_scores
    
    def calculate_loss(self, pos_scores, neg_scores):
        """
        Calculate BPR loss
        
        Args:
            pos_scores (torch.FloatTensor): Predicted scores for positive samples
            neg_scores (torch.FloatTensor): Predicted scores for negative samples
            
        Returns:
            torch.FloatTensor: Computed loss value
        """
        # BPR loss: -log(sigmoid(pos_score - neg_score))
        # Since scores are already in [0, 1], we can use them directly
        # But for BPR, we want to maximize pos_score - neg_score
        diff = pos_scores - neg_scores
        loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
        return loss
    
    def predict(self, user_ids, item_ids):
        """
        Predict scores for given user-item pairs
        
        Args:
            user_ids (torch.LongTensor): User IDs
            item_ids (torch.LongTensor): Item IDs
            
        Returns:
            torch.FloatTensor: Predicted scores
        """
        # Get ID embeddings
        user_id_embeds = self.user_embedding(user_ids)
        item_id_embeds = self.item_embedding(item_ids)
        
        # Get metadata embeddings
        user_meta_embeds = self.user_metadata_embeddings[user_ids]
        item_meta_embeds = self.item_metadata_embeddings[item_ids]
        
        # Project metadata embeddings
        user_meta_embeds = self.user_meta_proj(user_meta_embeds)
        item_meta_embeds = self.item_meta_proj(item_meta_embeds)
        
        # Concatenate
        concat_embeds = torch.cat([
            user_id_embeds,
            user_meta_embeds,
            item_id_embeds,
            item_meta_embeds
        ], dim=-1)
        
        # Predict
        scores = self.mlp_layers(concat_embeds).squeeze(-1)
        return scores
    
    @torch.no_grad()
    def recommend(self, user_id, k=None, batch_size_eval=1024):
        """
        Generate item recommendations for a user
        
        Args:
            user_id (int): User ID to generate recommendations for
            k (int, optional): Number of items to recommend
            batch_size_eval (int): Batch size for evaluation
            
        Returns:
            torch.Tensor: Recommended item indices or scores
        """
        self.eval()
        device = self.device
        
        # Get user index as int
        if isinstance(user_id, torch.Tensor):
            user_idx = int(user_id.detach().cpu().item())
        else:
            user_idx = int(user_id)
        
        # Accumulate scores in chunks
        scores_chunks = []
        start = 0
        while start < self.n_items:
            end = min(start + batch_size_eval, self.n_items)
            items = torch.arange(start, end, device=device, dtype=torch.long)
            users = torch.full((len(items),), user_idx, device=device, dtype=torch.long)
            
            scores_batch = self.predict(users, items)
            scores_chunks.append(scores_batch)
            start = end
        
        scores = torch.cat(scores_chunks, dim=0)  # (n_items,)
        
        if k is None:
            return scores
        else:
            _, idxs = torch.topk(scores, k)
            return idxs


if __name__ == '__main__':
    # Example usage
    from dataset import ML1MDataset
    from dataloader import TrainDataLoader, EvalDataLoader
    from trainer import Trainer

    # Load dataset
    dataset = ML1MDataset('ml-1m')

    # Create model
    model = NCFRecommender(
        n_users=dataset.get_user_num(),
        n_items=dataset.get_item_num(),
        embed_dim=64
    )

    # Get split datasets
    train_data = dataset.get_split_data('train')
    valid_data = dataset.get_split_data('validation')
    test_data = dataset.get_split_data('test')

    # Create dataloaders
    train_loader = TrainDataLoader(train_data, batch_size=2048, shuffle=True)
    valid_loader = EvalDataLoader(valid_data, train_data, batch_size=2048)
    test_loader = EvalDataLoader(test_data, train_data, batch_size=2048)

    trainer = Trainer(
        model=model,
        train_data=train_loader,
        eval_data=valid_loader,
        test_data=test_loader,
        epochs=100,
    )

    valid_result, test_result = trainer.fit(save_model=False)
    print(f"Best Validation Result: {valid_result}")
    print(f"Test Result: {test_result}")

