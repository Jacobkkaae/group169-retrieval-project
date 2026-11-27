from src.retrievers.models.encoder import TransformerEncoder
import torch.nn as nn
from src.retrievers.utils import tokenize, getDocumentText, createIndexDict
import torch
from transformers import BertTokenizer
import torch.nn.functional as F

class DPR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.__load_models()

    
    def __load_models(self):

        self.query_encoder = TransformerEncoder(n_layers = self.cfg.n_layers, n_heads = self.cfg.n_heads,
                                           emb_dim = self.cfg.emb_dim, hidden_dim = self.cfg.hidden_dim,
                                           feed_forward_dim = self.cfg.feed_forward_dim, pooling=self.cfg.pooling_type, max_token_len= self.cfg.max_token_length)
        self.document_encoder = TransformerEncoder(n_layers = self.cfg.n_layers, n_heads = self.cfg.n_heads,
                                           emb_dim = self.cfg.emb_dim, hidden_dim = self.cfg.hidden_dim,
                                           feed_forward_dim = self.cfg.feed_forward_dim, pooling=self.cfg.pooling_type, max_token_len= self.cfg.max_token_length)
        
        # self.query_encoder.to(self.cfg.device)
        # print("Query on: ", self.query_encoder.device)
        # self.document_encoder.to(self.cfg.device)
        # print("Document on: ", self.document_encoder.device)
        
        
        query_weights_path = self.cfg.get('query_encoder_weights_path', None)
        if self.cfg.get('query_encoder_weights_path', None):
            self.query_encoder.load_state_dict(query_weights_path)
        else:
            self.query_encoder.apply(self.init_encoder)

        document_weights_path = self.cfg.get('document_encoder_weights_path', None)
        if self.cfg.get('document_encoder_weights_path', None):
            self.document_encoder.load_state_dict(document_weights_path)
        else:
            self.document_encoder.apply(self.init_encoder)

    def run_inference(self, question_input_ids, question_attention_mask,
            text_input_ids, text_attention_mask, k_value = None):
        embedded_questions = self.query_encoder.forward(question_input_ids, question_attention_mask)
        embedded_documents = self.document_encoder.forward(text_input_ids, text_attention_mask)

        embedded_questions = F.normalize(embedded_questions, dim=-1)
        embedded_documents = F.normalize(embedded_documents, dim=-1)
    
        question_dot_products = self.compute_dot_products(embedded_questions, embedded_documents) / 0.05

        if k_value != None:
             pass
        
        return question_dot_products

    def tokenize_questions_and_docs(self, all_queries, examples, doc_id_to_index, corpus):
        question_order = []
        document_order = []

        tokenized_questions = []
        tokenized_documents = []
         
        for index, query in enumerate(all_queries):
                    question_order.append(query.id)
                    tokenized_questions.append(tokenize(query.text, self.tokenizer, self.cfg.max_token_length))
                    if index == examples-1:
                        break

        for question_index, question in enumerate(all_queries):
                for document_index, document_id in enumerate(question.relevant_document_ids):
                    document_order.append(document_id)
                    document_text = getDocumentText(doc_id_to_index, document_id, corpus)
                    tokenized_documents.append(tokenize(document_text, self.tokenizer, self.cfg.max_token_length))
                
                if question_index == examples-1:
                    break

        tokenized_questions_tensor = torch.tensor(tokenized_questions)
        
        tokenized_documents_tensor = torch.tensor(tokenized_documents)
        

        return tokenized_questions_tensor, tokenized_documents_tensor, question_order, document_order

    def embed_questions_and_documents(self, tokenized_questions_tensor, tokenized_documents_tensor):
        embedded_questions = []
        embedded_documents = []
         
        for index, tokenized_question in enumerate(tokenized_questions_tensor):
            single_tokenized_question_tensor = tokenized_questions_tensor[index]
            question_embedding = self.query_encoder(single_tokenized_question_tensor)
            embedded_questions.append(question_embedding)

        for index, tokenized_document in enumerate(tokenized_documents_tensor):
            single_tokenized_document_tensor = tokenized_documents_tensor[index]
            document_embedding = self.document_encoder(single_tokenized_document_tensor)
            embedded_documents.append(document_embedding)

        return embedded_questions, embedded_documents

    def compute_dot_products(self, embedded_questions, embedded_documents):

        question_dot_products = torch.zeros(len(embedded_questions), len(embedded_documents), device=self.cfg.device)
        
        for question_index, embedded_question in enumerate(embedded_questions):
            for document_index, embedded_document in enumerate(embedded_documents):
                question_dot_products[question_index][document_index] =  torch.dot(embedded_question, embedded_document)
        
        return question_dot_products

    def init_encoder(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            if module._qkv_same_embed_dim:
                nn.init.xavier_uniform_(module.in_proj_weight)
            else:
                nn.init.xavier_uniform_(module.q_proj_weight)
                nn.init.xavier_uniform_(module.k_proj_weight)
                nn.init.xavier_uniform_(module.v_proj_weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)
