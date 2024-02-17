import torch
import numpy as np
from torch import nn


# ---------------------------- Input Embeddings ---------------------------------- #

class InputEmbeddings(nn.Module):

    def __init__(self, d_model:int, vocab_size:int,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embeddings=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embeddings(x) * np.sqrt(self.d_model)
    
# -------------------------------------------------------------------------------- #
    


# ---------------------------- Positional Encoding ------------------------------- #
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len: int, dropout:float,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        # Create a matrix of shape (seq_len,d_model)
        pe=torch.zeros(seq_len,d_model)

        # Create a vector of shape (seq_len,1)
        position=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div=torch.exp(torch.arange(0,d_model,2).float()*(np.log(10000.0)/d_model))
        # Apply the sine function to even postion of the vector
        pe[:,0::2]=torch.sin(position*div)
        # Apply the cos function to odd postion of the vector
        pe[:,1::2]=torch.cos(position*div)

        # (1,seq_len,d_model)
        pe=pe.unsqueeze(0)

        self.register_buffer('pe',pe)


    def forward(self,x):
        x=x+(self.pe[:,:x.shape[1],:]).required_grad_(False)
        return self.dropout(x)
    
# -------------------------------------------------------------------------------- #


# ---------------------------- Layer Normalization ------------------------------- #
    
class LayerNormalization(nn.Module):
    
    def __init__(self,eps:float=10**-6,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1))
        self.beta=nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps)+self.beta
    
# -------------------------------------------------------------------------------- #


# ----------------------------    Feed Forward     ------------------------------- #
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int , d_ff:int , dropout:float ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # W1, B1
        self.linear_1=nn.Linear(d_model,d_ff) 
        self.dropout=nn.Dropout(dropout)
        # W2, B2
        self.linear_2=nn.Linear(d_ff,d_model)

    def forward(self,x):
        # (Batch, seq_len, d_model)  --> (Batch, seq_len, d_ff)  --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# -------------------------------------------------------------------------------- #
    


# ----------------------------    Multi Head Attention     ----------------------- #
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model=d_model
        self.h=h
        
        assert d_model%h==0, "d_model is not divisible by h"

        self.d_k=d_model//h

        # Wq
        self.w_q=nn.Linear(d_model,d_model) 
        
        # Wk
        self.w_k=nn.Linear(d_model,d_model) 

        # Wv
        self.w_v=nn.Linear(d_model,d_model)
        
        # Wo
        self.w_o=nn.Linear(d_model,d_model)

        self.dropout=nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k=query.shape[-1]

        # (Batch, h, seq_len, d_k)  -->  (Batch, h, seq_len, seq_len)
        attention_scores=(query@key.transpose(-2,-1))/np.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)

        # (Batch, h, seq_len, seq_len)
        attention_scores=attention_scores.softmax(dim=-1) 

        if dropout is not None:
            attention_scores=dropout(attention_scores)
    
        return (attention_scores@value),attention_scores

    def forward(self,q,k,v,mask):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        query = self.w_q(q) 
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k)
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v)

        # (Batch, seq_len, d_model) --> (Batch,seq_len,h,d_k) --> (Batch,h,seq_len,d_k)
        query = query.view(query.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2)
        
        # (Batch, seq_len, d_model) --> (Batch,seq_len,h,d_k) --> (Batch,h,seq_len,d_k)
        key = key.view(key.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2)
        
        # (Batch, seq_len, d_model) --> (Batch,seq_len,h,d_k) --> (Batch,h,seq_len,d_k)
        value = value.view(value.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2)

        x,self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        # (Batch,h,seq_len,d_k) --> (Batch,seq_len,h,d_k)  --> (Batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)

        return self.w_o(x)

# -------------------------------------------------------------------------------- #


# ----------------------------    Residual Connection     ------------------------ #
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout:float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()

    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
# -------------------------------------------------------------------------------- #
    


# ----------------------------    Encoder Block     ------------------------------ #
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self,x,src_mask):
        x=self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x=self.residual_connections[1](x,self.feed_forward_block)
        return x
    
# -------------------------------------------------------------------------------- #
                                       

# ----------------------------        Encoder       ------------------------------ #

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

# -------------------------------------------------------------------------------- #



# ----------------------------        Decoder Block      ------------------------- #

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock, cross_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock, dropout:float,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.self_attention_block=self_attention_block
        self.cross_attention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x=self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x=self.residual_connections[1](x, lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connections[2](x, self.feed_forward_block)

        return x

# -------------------------------------------------------------------------------- #


# ----------------------------        Decoder       ------------------------------ #

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)

        return self.norm(x)
    
# -------------------------------------------------------------------------------- #
    

# ----------------------------        Projection Layer       --------------------- #
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.proj=nn.Linear(d_model,vocab_size)

    def forward(self,x):
        # (Batch, seq_len, d_model)  -->  (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)

# -------------------------------------------------------------------------------- #