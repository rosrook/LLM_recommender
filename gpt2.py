from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# run 'export HF_ENDPOINT=https://hf-mirror.com' in the terminal before running the script

# Load the model and tokenizer
# 非本地加载gpt2
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 本地加载gpt2
# local_dir="你的本地路径/gpt2_cache"
# model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=local_dir, local_files_only=True)
# tokenizer=GPT2Tokenizer.from_pretrained("gpt2", cache_dir=local_dir, local_files_only=True)

# Set the model to evaluation mode
model.eval()

# build the input
text = "Hello, how are you?"
input = tokenizer(text, return_tensors="pt")

# get the hidden states
with torch.no_grad():
    outputs = model(**input, output_hidden_states=True)
    hidden_states = outputs.hidden_states # list of 13 tensors of shape (1, 6, 768)
    # use the first hidden state as the embedding
    embedding = hidden_states[0] # size (1, 6, 768)
    # mean pool the embedding
    embedding = torch.squeeze(torch.mean(embedding, dim=1), dim=0) # size (768)