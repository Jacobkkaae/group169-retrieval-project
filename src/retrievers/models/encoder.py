import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from transformers import BertTokenizer

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
        self.embedder = nn.Embedding(num_embeddings=BertTokenizer.from_pretrained('bert-base-uncased').vocab_size, embedding_dim = self.emb_dim)

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


        print("Before embedding: Number of embeddings: ", len(embedding))
        print("Before embedding: Embedding dimension: ", len(embedding[0]))
        print("Before embedding: First ten numbers of first embeding: ", embedding[0][0:11])

        result = self.model(embedding)

        print("After embedding: Number of embeddings: ", len(result))
        print("After embedding: Embedding dimension: ", len(result[0]))
        print("After embedding: First ten numbers of first embeding: ", result[0][0:11])
        print()


        return result
