from src.retrievers.models.encoder import TransformerEncoder
import torch.nn as nn
from src.retrievers.utils import tokenize, getDocumentText, createIndexDict
import torch
from transformers import BertTokenizer

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
        
        query_weights_path = self.cfg.get('query_encoder_weights_path', None)
        if self.cfg.get('query_encoder_weights_path', None):
            self.query_encoder.load_state_dict(query_weights_path)

        document_weights_path = self.cfg.get('document_encoder_weights_path', None)
        if self.cfg.get('document_encoder_weights_path', None):
            self.document_encoder.load_state_dict(document_weights_path)

    def run_inference(self, all_queries, corpus, k_value, examples):
        doc_id_to_index = createIndexDict(corpus)
        
        tokenized_questions_tensor, tokenized_documents_tensor, question_order, document_order = self.tokenize_questions_and_docs(all_queries=all_queries, 
                                                                                                                    examples=examples, doc_id_to_index= doc_id_to_index, corpus = corpus)

        embedded_questions, embedded_documents = self.embed_questions_and_documents(tokenized_questions_tensor, tokenized_documents_tensor)

        question_cosine_scores = self.compute_cosine_scores(embedded_questions, embedded_documents)

        top_k_documents, top_k_cosine_scores = self.compute_top_k_docs(question_cosine_scores, k_value, document_order)
        
        print(top_k_documents)


        return top_k_documents, top_k_cosine_scores

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

    def compute_cosine_scores(self, embedded_questions, embedded_documents):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        question_cosine_scores = []
                
        for embedded_question in embedded_questions:
            temp_cosine_scores = []
            for embedded_document in embedded_documents:
                result = cos(embedded_question, embedded_document)
                temp_cosine_scores.append(result)
                

            question_cosine_scores.append(temp_cosine_scores)

        return question_cosine_scores

    def compute_top_k_docs(self, question_cosine_scores, k_value, document_order):
        top_k_documents = []
        top_k_sorted_document_cosine_scores = []

        for document_cosine_scores in question_cosine_scores:
            document_cosine_scores_tensor = torch.tensor(document_cosine_scores)
            sorted_document_cosine_scores, sort_indices = torch.sort(document_cosine_scores_tensor, descending=True)
            temp_top_k_sorted_document_cosine_scores = sorted_document_cosine_scores[:k_value]
            top_k_sorted_document_cosine_scores.append(temp_top_k_sorted_document_cosine_scores)
            values, indices = torch.topk(sort_indices, k_value, largest=False)
            temp_top_k_documents = []
            
            for idx in indices:
                temp_top_k_documents.append(document_order[idx])

            top_k_documents.append(temp_top_k_documents)

        return top_k_documents, top_k_sorted_document_cosine_scores
