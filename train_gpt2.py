from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


# GPT Building Blocks

class CausalSelfAttention(nn.Module):
    """
    Multiheaded Causal Self Attention
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, query, value projections for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularizaton
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # mask - called bias by OpenAI/HF
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size() # Batch size, sequence length, embedding dimensionality (token embedding length)

        # We will first calculate query, key and values for ALL HEADS
        # nh is "number of heads", hs is "head size", and C (num of channels) = nh * ns
        # e.g in GPT-2 (124M), n_heads = 12, hs = 64, so nh * ns = C = 768 channels in Transformer
        # each head outputs sequence of vectors of a size 64, so in multiheaded self attention the last step is to conncatinate outputs from all the heads
        # by doing this conncationation we effecively restore the dimensions of original embeddings and preserve
        # different information we obtained from each head
        
        # Joined Q, K, V values for all the heads
        # attention(Q,K,V) = softmax((Q @ K^T) * 1/sqrt(n_embed)) @ V
        # For a single head -> x @ Wq = Q, x @ Wk = 0, x @ Wv = V
        # In multiheaded attention Wq holds weights for all the heads, same goes for Wk, Wv
        # So q, k , v here are q for all the heads, k for all the heads, v for all the heads 
        # Since x is (B, T, C) and it passes through linear layer with weight matrix of (C, 3 * C)
        # qkv is (B, T, 3 * C)
        qkv = self.c_attn(x)

        # Each of them will be of the size (B, T, C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Now we want to retain the notion of separate q, k, v per head
        # using .view() we transform (B, T, C) into (B, T, nh, n_embed / n_heads)
        # We transpose (B, T, nh, hs) into (B, nh, T, hs) so this is the tensor that holds Q
        # It hold data for B batches, each batch has 12 heads, each sequence is of length T, and each query is of hs length
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # We repeat for all the others
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # We calulate the attention - we transponse only the last two dims of k
        # (B, nh, T, hs) @ (B, nh, hs, T) = (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # For every batch, and for every head in a batch, we will look at all T sequences of length T
        # Since bias is tensor (1, 1, bs, bs) where (bs, bs) is a lower triangular matrix
        # We will overlay it over each (T, T) matrix in (B, nh, T, T) tensor and turn all the 0 to -inf
        # This will essentially mask the future tokens so the earlyer ones can't attend to them
        # We use -inf and not 0 because this will have benefits for softmax
        # This is also called autoregressive mask 
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    
        # Now we apply softmax along each row in each sequence T
        # So if we look at row i in TxT matrix and pluck it out
        # We will see how much attention token at position i gives to all the other tokens is sequence T
        # So once we apply softmax on a row basis we normalized attention that each token i pays to all the other tokens in the sequence and it sums to 1
        att = F.softmax(att, dim=-1)
        
        # Now we need to do att @ V so we obtain modifed values of embeddings based on the attention
        # (B, nh, T, T) @ (B, nh, T, hs) = (B, nh, T, hs)
        y = att @ v

        # Alternatively previous 4 lines coud have been commented out and we could use flash attention
        # y = F.scaled_dot_product-attention(q, k, v is_causal=True)

        # Now we first transpose this tensor so we get (B, T, nh, hs)
        # We than need to use contiguous() function to allocate continous block of memory for our tensor
        # This will ensure us that tensor's data is stored in memory in sequential, row-major order.
        # Now we can perform reshaping of our tensor, where we concat outputs of each head
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Finally we just project the output through the linear layer
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor):
        # Forward pass in gpt2 defers from original transfomer since it uses layer norms befor attention and mlp 
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    """
    This dataclass will be used to store the configuration parameters for GPT model
    """
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens - 50k BPE + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimesnion


class GPT(nn.Module):
     
    def __init__(self, config):
        super().__init__()
        self.config = config
        # This is how HF organizes their gpt2 implementation
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # lm_head shares the weight matrix with wte
        # This is quite clever optimization as it turns out it helps with regularization and reduces the number of trainable parameters
        # Additionally this does not seem correct at the first glance since wte = nn.Embedding(vocab_size, n_embd) and lm_head = nn.Linear(n_embd, vocab_size)
        # But since lm_head is linear it will be saved as (vocab_size, n_embd) actually and that matches with dimensions of wte
        self.transformer.wte.weight = self.lm_head.weight
    
    def forward(self, idx, targets: None):
        # idx is as tensor containg batches B of sequences T, (B, T) - T is a sequence of token ids - idx = [[14, 2435, 1234], [345, 43567, 8067]]
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # Now we create a tensor [0, 1, 2, ..., T-1] - this basically marks positions inside our sequence
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # For each position in a sequence we will get the positional embedding
        # Since we have T elements in the sequence we will obtain positional emedding for each element so it is a tensor (T, n_embd)
        # Since all the batches have sequences that are of length T we pluck out the positional embedding just once
        pos_emb = self.transformer.wpe(pos)
        # In this step we create token embeddings for each id in a sequence for each batch - this results in (B, T, n_embd) tensor
        tok_emb = self.transformer.wte(idx) 
        # Now we need to combine these two - since pos_e (T, n_embd) and tok_e (B, T, n_embd) PyTorch will do the broadcasting for us
        # It basically adds a dimension B to pos_e to match the dims of tok_e
        x = tok_emb + pos_emb

        # Now we propagete the x which is (B, T, C) though Transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # Now we do the LayerNorm one last time
        x = self.transformer.ln_f(x)

        # We pass x which is (B, T, n_embd) through LM head which will basically do the calssification and predict
        # the next most probable token
        # The output here is (B, T, vocab_size) where in the last dimension we have 50257 logits - one for each possible token in vocab
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss 


    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained gpt {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600)
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # Creating from-scratch model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


# Initialization and inference

num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
print("Model loaded 🤖")
model.eval()
model.to('mps')

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens: list[int] = enc.encode("Hello, I'm a language model,")
tokens: torch.Tensor = torch.tensor(tokens, dtype=torch.long) # Converting list[int] to torch.Tensor
# We will add a dimension so we get (B, T) = (5, 8)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to('mps')

torch.manual_seed(42)
torch.mps.manual_seed(42)

# Until the sequence length is equal to max_length
print("Happy decoding!")
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        # We keep only the logits of the last element in the sequence
        logits = logits[:, -1,:]
        # We apply softmax to logits to get probabilites
        # logits (B, 1, vocab_size)
        probs = F.softmax(logits, dim=-1)
        # We will do the topk sampling of 50
        # So for each batch we will take 50 most probable tokens
        # topk_probs,here becomes (B, 50), topk_indexes is (B, 50)
        # it will receive probs (B, 1, vocab_size) and select 50 elements from the last dimension [vocab_size] that have the highest probability
        topk_probs, topk_indeces = torch.topk(probs, 50, dim=-1)
        # Select a token from topk probabilites
        # ix (B, 1)
        ix = torch.multinomial(topk_probs, 1)
        # For each batch in (B, 50) we will extract 1 token with index ix from 50 possible 
        xcol = torch.gather(topk_indeces, -1, ix)
        # We concat the predicted token id to the rest of the sequence
        # x (B, T+1)
        x = torch.cat([x, xcol], dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)


# Training

# Initialize model with random params - check the way the original authors did it
# model = GPT(GPTConfig()) 
# Adjust model.forward() to optionally receive y (training data (B, T)) and return logits and loss

# Create a Dataloader which will deal with batching and creating training and test splits - data has to be moved to GPU - I will do this in the training loop
# data_loader = DataLoader(batches, sequence_length, dataset)

# Write a standard training loop
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# for i in range(epochs):
#   get through all the x, y batches in every epoch
#   x, y = data_loader.next_batch()
#   x, y = x.to("mps"), y.to("mps") I am interested who will te tr
#   optimizer.zero_grad()
#   logits, loss = model(x , y)
#   loss.backward() accumulate gradients
#   optimizer.step() perform one step of optimization

# Explore how to run training on multiple gpu-s
# Datasets for Pretraining
# Evals

