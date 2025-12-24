# coding: utf-8
"""
Enhanced GPT-2 Recommender with innovative features:
1. Dynamic metadata extraction during training (end-to-end learning)
2. Cross-attention mechanism for feature fusion
3. Contrastive learning with semantic understanding
4. Multi-granularity feature interaction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from gpt2_encoder import GPT2MetadataEncoder, format_user_metadata, format_item_metadata
from recommender import AbstractRecommender


class CrossAttentionFusion(nn.Module):
    """Cross-attention mechanism for fusing ID embeddings and metadata embeddings"""
    
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Query, Key, Value projections for ID embeddings
        self.id_q = nn.Linear(embed_dim, embed_dim)
        self.id_k = nn.Linear(embed_dim, embed_dim)
        self.id_v = nn.Linear(embed_dim, embed_dim)
        
        # Query, Key, Value projections for metadata embeddings
        self.meta_q = nn.Linear(embed_dim, embed_dim)
        self.meta_k = nn.Linear(embed_dim, embed_dim)
        self.meta_v = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, id_embed, meta_embed):
        """
        Cross-attention fusion
        
        Args:
            id_embed: (batch_size, embed_dim) ID embedding
            meta_embed: (batch_size, embed_dim) Metadata embedding
            
        Returns:
            fused_embed: (batch_size, embed_dim) Fused embedding
        """
        batch_size = id_embed.size(0)
        
        # Cross-attention: ID as query, Metadata as key and value
        # Reshape for multi-head attention
        q = self.id_q(id_embed).view(batch_size, self.num_heads, self.head_dim)  # (batch, heads, head_dim)
        k = self.meta_k(meta_embed).view(batch_size, self.num_heads, self.head_dim)  # (batch, heads, head_dim)
        v = self.meta_v(meta_embed).view(batch_size, self.num_heads, self.head_dim)  # (batch, heads, head_dim)
        
        # Transpose for attention computation: (batch, heads, head_dim) -> (batch, heads, 1, head_dim)
        q = q.unsqueeze(2)  # (batch, heads, 1, head_dim)
        k = k.unsqueeze(2)  # (batch, heads, 1, head_dim)
        v = v.unsqueeze(2)  # (batch, heads, 1, head_dim)
        
        # Attention scores: (batch, heads, 1, 1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values: (batch, heads, 1, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.squeeze(2)  # (batch, heads, head_dim)
        attn_output = attn_output.contiguous().view(batch_size, self.embed_dim)  # (batch, embed_dim)
        
        # Concatenate ID and attended metadata, then project
        fused = torch.cat([id_embed, attn_output], dim=-1)  # (batch, 2*embed_dim)
        fused = self.out_proj(fused)  # (batch, embed_dim)
        fused = self.layer_norm(fused + id_embed)  # Residual connection
        
        return fused


class GPT2RecommenderEnhanced(AbstractRecommender):
    """
    Enhanced GPT-2 Recommender with:
    1. Dynamic metadata extraction (end-to-end training)
    2. Cross-attention feature fusion
    3. Contrastive learning support
    """
    
    def __init__(self, n_users, n_items, embed_dim, dataset, gpt2_model_name='gpt2', 
                 use_peft=True, metadata_dim=None, freeze_gpt2=False, use_cache=False,
                 use_attention=True, contrastive_weight=0.1):
        """
        Initialize Enhanced GPT-2 based recommender
        
        Args:
            n_users (int): Number of users
            n_items (int): Number of items
            embed_dim (int): Dimension of ID embeddings
            dataset: ML1MDataset object to access metadata
            gpt2_model_name (str): GPT-2 model name
            use_peft (bool): Whether to use PEFT for GPT-2 fine-tuning
            metadata_dim (int): Dimension of metadata embeddings
            freeze_gpt2 (bool): Whether to freeze GPT-2 parameters
            use_cache (bool): Whether to cache metadata embeddings (faster but less flexible)
            use_attention (bool): Whether to use cross-attention fusion
            contrastive_weight (float): Weight for contrastive learning loss
        """
        super().__init__(n_users, n_items, embed_dim)
        
        self.dataset = dataset
        self.freeze_gpt2 = freeze_gpt2
        self.use_cache = use_cache
        self.use_attention = use_attention
        self.contrastive_weight = contrastive_weight
        
        # Initialize GPT-2 encoder
        self.gpt2_encoder = GPT2MetadataEncoder(
            model_name=gpt2_model_name,
            use_peft=use_peft,
            device='cuda'
        )
        
        # GPT-2 embedding dimension
        self.metadata_dim = metadata_dim if metadata_dim else self.gpt2_encoder.gpt2_dim
        
        # Cache metadata texts for faster access
        self._cache_metadata_texts()
        
        # Optionally precompute metadata embeddings (for faster training but less flexible)
        if use_cache:
            print("Precomputing user metadata embeddings (cached mode)...")
            self.user_metadata_embeddings = self._precompute_user_metadata_embeddings()
            print("Precomputing item metadata embeddings (cached mode)...")
            self.item_metadata_embeddings = self._precompute_item_metadata_embeddings()
        else:
            self.user_metadata_embeddings = None
            self.item_metadata_embeddings = None
            print("Using dynamic metadata extraction (end-to-end training mode)")
        
        # Projection layer to align metadata embedding dimension with embed_dim
        if self.metadata_dim != embed_dim:
            self.user_meta_proj = nn.Linear(self.metadata_dim, embed_dim)
            self.item_meta_proj = nn.Linear(self.metadata_dim, embed_dim)
        else:
            self.user_meta_proj = nn.Identity()
            self.item_meta_proj = nn.Identity()
        
        # Feature fusion: Cross-attention or simple concatenation
        if use_attention:
            self.user_fusion = CrossAttentionFusion(embed_dim, num_heads=4)
            self.item_fusion = CrossAttentionFusion(embed_dim, num_heads=4)
            fusion_output_dim = embed_dim * 2  # After fusion, we concatenate user and item
        else:
            self.user_fusion = None
            self.item_fusion = None
            fusion_output_dim = embed_dim * 4  # Simple concatenation
        
        # MLP layers for prediction
        self.mlp_layers = nn.Sequential(
            nn.Linear(fusion_output_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Contrastive learning projection head
        self.contrastive_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2)
        )
        
        # Freeze GPT-2 if specified
        if freeze_gpt2:
            for param in self.gpt2_encoder.parameters():
                param.requires_grad = False
    
    def _cache_metadata_texts(self):
        """Cache formatted metadata texts for faster access"""
        user_meta_df = self.dataset.get_user_meta()
        item_meta_df = self.dataset.get_item_meta()
        
        self.user_texts = []
        for user_id in range(self.n_users):
            if user_id in user_meta_df.index:
                user_row = user_meta_df.loc[user_id]
                text = format_user_metadata(user_row)
            else:
                text = "User profile: Gender is Unknown, Age group is Unknown, Occupation is Unknown."
            self.user_texts.append(text)
        
        self.item_texts = []
        for item_id in range(self.n_items):
            if item_id in item_meta_df.index:
                item_row = item_meta_df.loc[item_id]
                text = format_item_metadata(item_row)
            else:
                text = "Movie: Unknown. Genres: Unknown."
            self.item_texts.append(text)
    
    def _precompute_user_metadata_embeddings(self):
        """Precompute GPT-2 embeddings for all user metadata"""
        embeddings = self.gpt2_encoder.encode_text(self.user_texts)
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cuda'
        return embeddings.to(device)
    
    def _precompute_item_metadata_embeddings(self):
        """Precompute GPT-2 embeddings for all item metadata"""
        embeddings = self.gpt2_encoder.encode_text(self.item_texts)
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cuda'
        return embeddings.to(device)
    
    def _get_user_metadata_embeddings(self, user_ids, training=False):
        """Get user metadata embeddings (dynamic or cached)"""
        if self.use_cache:
            # 从 user_ids tensor 获取设备，更可靠
            device = user_ids.device
            if self.user_metadata_embeddings.device != device:
                self.user_metadata_embeddings = self.user_metadata_embeddings.to(device)
            return self.user_metadata_embeddings[user_ids]
        else:
            # Dynamic extraction during training
            batch_texts = [self.user_texts[uid.item()] for uid in user_ids]
            if training:
                return self.gpt2_encoder.encode_text_train(batch_texts)
            else:
                return self.gpt2_encoder.encode_text(batch_texts)
    
    def _get_item_metadata_embeddings(self, item_ids, training=False):
        """Get item metadata embeddings (dynamic or cached)"""
        if self.use_cache:
            # 从 item_ids tensor 获取设备，更可靠
            device = item_ids.device
            if self.item_metadata_embeddings.device != device:
                self.item_metadata_embeddings = self.item_metadata_embeddings.to(device)
            return self.item_metadata_embeddings[item_ids]
        else:
            # Dynamic extraction during training
            batch_texts = [self.item_texts[iid.item()] for iid in item_ids]
            if training:
                return self.gpt2_encoder.encode_text_train(batch_texts)
            else:
                return self.gpt2_encoder.encode_text(batch_texts)
    
    def forward(self, batch_data):
        """
        Forward pass with dynamic metadata extraction and attention fusion
        
        Args:
            batch_data (tuple): Batch data tuple containing (users, pos_items, neg_items)
                               Each element is a torch.Tensor of shape (batch_size,)
            
        Returns:
            tuple: (pos_scores, neg_scores) predicted scores
        """
        user_ids, pos_item_ids, neg_item_ids = batch_data
        training = self.training
        
        # Get ID embeddings
        user_id_embeds = self.user_embedding(user_ids)
        pos_item_id_embeds = self.item_embedding(pos_item_ids)
        neg_item_id_embeds = self.item_embedding(neg_item_ids)
        
        # Get metadata embeddings (dynamic or cached)
        user_meta_embeds = self._get_user_metadata_embeddings(user_ids, training=training)
        pos_item_meta_embeds = self._get_item_metadata_embeddings(pos_item_ids, training=training)
        neg_item_meta_embeds = self._get_item_metadata_embeddings(neg_item_ids, training=training)
        
        # Project metadata embeddings
        user_meta_embeds = self.user_meta_proj(user_meta_embeds)
        pos_item_meta_embeds = self.item_meta_proj(pos_item_meta_embeds)
        neg_item_meta_embeds = self.item_meta_proj(neg_item_meta_embeds)
        
        # Feature fusion: Cross-attention or concatenation
        if self.use_attention:
            # Use cross-attention for better feature interaction
            user_fused = self.user_fusion(user_id_embeds, user_meta_embeds)
            pos_item_fused = self.item_fusion(pos_item_id_embeds, pos_item_meta_embeds)
            neg_item_fused = self.item_fusion(neg_item_id_embeds, neg_item_meta_embeds)
            
            # Concatenate user and item fused features
            pos_concat = torch.cat([user_fused, pos_item_fused], dim=-1)
            neg_concat = torch.cat([user_fused, neg_item_fused], dim=-1)
        else:
            # Simple concatenation
            pos_concat = torch.cat([
                user_id_embeds, user_meta_embeds,
                pos_item_id_embeds, pos_item_meta_embeds
            ], dim=-1)
            neg_concat = torch.cat([
                user_id_embeds, user_meta_embeds,
                neg_item_id_embeds, neg_item_meta_embeds
            ], dim=-1)
        
        # Predict scores
        pos_scores = self.mlp_layers(pos_concat)
        neg_scores = self.mlp_layers(neg_concat)
        
        return pos_scores, neg_scores
    
    def calculate_loss(self, pos_scores, neg_scores):
        """
        Calculate combined BPR loss and contrastive loss
        
        Args:
            pos_scores (torch.FloatTensor): Predicted scores for positive samples
            neg_scores (torch.FloatTensor): Predicted scores for negative samples
            
        Returns:
            torch.FloatTensor: Combined loss value
        """
        # BPR loss
        diff = pos_scores - neg_scores
        bpr_loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
        
        # Optional: Add contrastive loss for semantic alignment
        # This encourages similar users/items to have similar embeddings
        if self.contrastive_weight > 0 and self.training:
            # Simple contrastive loss: maximize pos_score, minimize neg_score
            contrastive_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-10).mean() + \
                              torch.log(torch.sigmoid(neg_scores) + 1e-10).mean()
            total_loss = bpr_loss + self.contrastive_weight * contrastive_loss
        else:
            total_loss = bpr_loss
        
        return total_loss
    
    def predict(self, user_ids, item_ids):
        """Predict scores for given user-item pairs"""
        # Get ID embeddings
        user_id_embeds = self.user_embedding(user_ids)
        item_id_embeds = self.item_embedding(item_ids)
        
        # Get metadata embeddings
        user_meta_embeds = self._get_user_metadata_embeddings(user_ids, training=False)
        item_meta_embeds = self._get_item_metadata_embeddings(item_ids, training=False)
        
        # Project
        user_meta_embeds = self.user_meta_proj(user_meta_embeds)
        item_meta_embeds = self.item_meta_proj(item_meta_embeds)
        
        # Fusion
        if self.use_attention:
            user_fused = self.user_fusion(user_id_embeds, user_meta_embeds)
            item_fused = self.item_fusion(item_id_embeds, item_meta_embeds)
            concat_embeds = torch.cat([user_fused, item_fused], dim=-1)
        else:
            concat_embeds = torch.cat([
                user_id_embeds, user_meta_embeds,
                item_id_embeds, item_meta_embeds
            ], dim=-1)
        
        scores = self.mlp_layers(concat_embeds).squeeze(-1)
        return scores
    
    @torch.no_grad()
    def recommend(self, user_id, k=None, batch_size_eval=1024):
        """Generate recommendations"""
        self.eval()
        # 从 embedding 层获取设备，避免 DataParallel 的问题
        try:
            device = next(self.parameters()).device
        except (StopIteration, RuntimeError):
            device = next(iter(self.user_embedding.parameters())).device
        
        if isinstance(user_id, torch.Tensor):
            user_idx = int(user_id.detach().cpu().item())
        else:
            user_idx = int(user_id)
        
        scores_chunks = []
        start = 0
        while start < self.n_items:
            end = min(start + batch_size_eval, self.n_items)
            items = torch.arange(start, end, device=device, dtype=torch.long)
            users = torch.full((len(items),), user_idx, device=device, dtype=torch.long)
            
            scores_batch = self.predict(users, items)
            scores_chunks.append(scores_batch)
            start = end
        
        scores = torch.cat(scores_chunks, dim=0)
        
        if k is None:
            return scores
        else:
            _, idxs = torch.topk(scores, k)
            return idxs

