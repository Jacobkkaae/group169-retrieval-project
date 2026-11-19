from src.retrievers.models.encoder import TransformerEncoder
import torch.nn as nn
from src.training.trainer import Trainer
from src.retrievers.utils import tokenize, getDocumentText, createIndexDict
import torch
from transformers import BertTokenizer

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

    def run_inference(self, all_queries, corpus):
        doc_id_to_index = createIndexDict(corpus)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        examples = 2
        
        question_order = []
        document_order = []

        tokenized_questions = []
        tokenized_documents = []


        for index, query in enumerate(all_queries):
            question_order.append(query.id)
            tokenized_questions.append(tokenize(query.text, tokenizer))
            if index > examples:
                break

        tokenized_questions_tensor = torch.tensor(tokenized_questions)


        for question_index, question in enumerate(all_queries):
            for document_index, document_id in enumerate(question.relevant_document_ids):
                document_order.append(document_id)
                document_text = getDocumentText(doc_id_to_index, document_id, corpus)
                tokenized_documents.append(tokenize(document_text, tokenizer))
            
            if question_index > examples:
                break

        tokenized_documents_tensor = torch.tensor(tokenized_documents)

        embedded_questions = []
        embedded_documents = []

        for index, tokenized_question in enumerate(tokenized_questions):
            single_tokenized_question_tensor = tokenized_questions_tensor[index]
            question_embedding = self.query_encoder(single_tokenized_question_tensor)
            embedded_questions.append(question_embedding)

        for index, tokenized_document in enumerate(tokenized_documents):
                    single_tokenized_document_tensor = tokenized_documents_tensor[index]
                    document_embedding = self.document_encoder(single_tokenized_document_tensor)
                    embedded_documents.append(document_embedding)

        return embedded_questions, embedded_documents



