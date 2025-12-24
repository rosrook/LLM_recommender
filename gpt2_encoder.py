# coding: utf-8

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np


class GPT2MetadataEncoder(nn.Module):
    """GPT-2 based encoder for extracting metadata embeddings"""
    
    def __init__(self, model_name='gpt2', use_peft=True, peft_config=None, device='cuda'):
        """
        Initialize GPT-2 metadata encoder
        
        Args:
            model_name (str): Name of the GPT-2 model
            use_peft (bool): Whether to use PEFT for fine-tuning
            peft_config (LoraConfig): PEFT configuration. If None, uses default LoRA config
            device (str): Device to run on
        """
        super().__init__()
        self.device = device
        
        # Load GPT-2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Apply PEFT if enabled
        if use_peft:
            if peft_config is None:
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    inference_mode=False,
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    target_modules=["c_attn", "c_proj"]
                )
            self.gpt2_model = get_peft_model(base_model, peft_config)
        else:
            self.gpt2_model = base_model
        
        self.gpt2_model.to(device)
        self.gpt2_model.eval()  # Set to eval mode initially, can be set to train mode when fine-tuning
        
        # GPT-2 hidden dimension (768 for GPT-2 base)
        self.gpt2_dim = base_model.config.n_embd
        
    def encode_text(self, texts, batch_size=32):
        """
        Encode text metadata to embeddings using GPT-2
        
        Args:
            texts (list): List of text strings to encode
            batch_size (int): Batch size for encoding
            
        Returns:
            torch.Tensor: Encoded embeddings of shape (len(texts), gpt2_dim)
        """
        self.gpt2_model.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize texts
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(self.device)
                
                # Get hidden states
                outputs = self.gpt2_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # List of hidden states
                
                # Use the last hidden state and mean pool over sequence length
                # hidden_states[-1] shape: (batch_size, seq_len, hidden_dim)
                last_hidden = hidden_states[-1]
                # Mean pool: average over sequence dimension
                batch_embeddings = torch.mean(last_hidden, dim=1)  # (batch_size, hidden_dim)
                
                embeddings.append(batch_embeddings)
        
        return torch.cat(embeddings, dim=0)
    
    def encode_text_train(self, texts, batch_size=32):
        """
        Encode text metadata during training (allows gradients)
        
        Args:
            texts (list): List of text strings to encode
            batch_size (int): Batch size for encoding
            
        Returns:
            torch.Tensor: Encoded embeddings of shape (len(texts), gpt2_dim)
        """
        self.gpt2_model.train()
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize texts
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            # Get hidden states (with gradients)
            outputs = self.gpt2_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Use the last hidden state and mean pool
            last_hidden = hidden_states[-1]
            batch_embeddings = torch.mean(last_hidden, dim=1)
            
            embeddings.append(batch_embeddings)
        
        return torch.cat(embeddings, dim=0)
    
    def forward(self, texts, training=False):
        """
        Forward pass for encoding texts
        
        Args:
            texts (list): List of text strings
            training (bool): Whether in training mode
            
        Returns:
            torch.Tensor: Encoded embeddings
        """
        if training:
            return self.encode_text_train(texts)
        else:
            return self.encode_text(texts)


def format_user_metadata(user_meta_row):
    """
    Format user metadata row to text string
    
    Args:
        user_meta_row: pandas Series with user metadata
        
    Returns:
        str: Formatted text string
    """
    gender = user_meta_row.get('gender', 'Unknown')
    age = user_meta_row.get('age', 'Unknown')
    occupation = user_meta_row.get('occupation', 'Unknown')
    
    text = f"User profile: Gender is {gender}, Age group is {age}, Occupation is {occupation}."
    return text


def format_item_metadata(item_meta_row):
    """
    Format item metadata row to text string
    
    Args:
        item_meta_row: pandas Series with item metadata
        
    Returns:
        str: Formatted text string
    """
    title = item_meta_row.get('title', 'Unknown')
    genres = item_meta_row.get('genres', 'Unknown')
    
    # Handle genres (might be separated by |)
    if isinstance(genres, str):
        genres = genres.replace('|', ', ')
    
    text = f"Movie: {title}. Genres: {genres}."
    return text

