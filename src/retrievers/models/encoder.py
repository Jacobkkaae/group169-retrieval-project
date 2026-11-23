import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from transformers import BertTokenizer
from src.retrievers.models.positional_encoding import PositionalEncoding 

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim =    emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.setup_model()

    def setup_model(self):
        """
        Instantiate the model.
        """

        # Bert vocab size is 30522
        self.embedder = nn.Embedding(num_embeddings=30522, embedding_dim = self.emb_dim)

        self.positional_encoder = PositionalEncoding(d_model = self.emb_dim, max_len = 1000)

        self.model = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.emb_dim, nhead=self.n_heads, dim_feedforward=self.feed_forward_dim, 
                dropout=0.1, activation='gelu'
            ),num_layers=self.n_layers
        )

    def pool(self, x):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension 
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        if self.cfg.model.pooling_type == 'cls':
            raise NotImplementedError()
        
        elif self.cfg.model.pooling_type == 'max':
            return torch.max(x, dim=1).values
        
        elif self.cfg.model.pooling_type == 'mean':
            return torch.mean(x, dim=1).values

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Run a forward pass on the model
        """

        embedding = self.embedder(x)

        positional_embedding = self.positional_encoder(embedding)


        result = self.model(positional_embedding)


        return result
