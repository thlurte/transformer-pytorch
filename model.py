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
    

