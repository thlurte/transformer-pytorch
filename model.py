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