from src.retrievers.models.encoder import TransformerEncoder
import torch.nn as nn

class DPR:
    def __init__(self, cfg):
        self.cfg = cfg
        self.__load_models()
    
    def __load_models(self):

        self.query_encoder = TransformerEncoder(n_layers = self.cfg.n_layers, n_heads = self.cfg.n_heads,
                                           emb_dim = self.cfg.emb_dim, hidden_dim = self.cfg.hidden_dim,
                                           feed_forward_dim = self.cfg.feed_forward_dim)
        self.document_encoder = TransformerEncoder(n_layers = self.cfg.n_layers, n_heads = self.cfg.n_heads,
                                           emb_dim = self.cfg.emb_dim, hidden_dim = self.cfg.hidden_dim,
                                           feed_forward_dim = self.cfg.feed_forward_dim)
        
        query_weights_path = self.cfg.get('query_encoder_weights_path', None)
        if self.cfg.get('query_encoder_weights_path', None):
            self.query_encoder.load_state_dict(query_weights_path)

        document_weights_path = self.cfg.get('document_encoder_weights_path', None)
        if self.cfg.get('document_encoder_weights_path', None):
            self.document_encoder.load_state_dict(document_weights_path)

    

        

