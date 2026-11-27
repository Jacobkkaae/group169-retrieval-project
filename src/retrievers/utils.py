from src.retrievers.base_retriever import Retriever
from src.data_utils.query import Query
from transformers import BertTokenizer


def retrieveQueryAndGetRelevancies(model: Retriever, queries: list[Query], k: int):
    """
    @param model: Retrieval model
    @param queries: A list of queries.
    @param k: Top-k relevant documents to consider.

    Returns a list of lists of booleans that are True when the retrieved document matches the ground truth of that query.
    """
    N = len(queries)
    retrieved_documents = model.Lookup(queries=[query.getQuery() for query in queries], k=k)
    relevancies = []
    for i in range(N):
        query_relevancies = []
        for document in retrieved_documents[i]:
            if queries[i].isDocumentRelevant(document):
                query_relevancies.append(True)
            else:
                query_relevancies.append(False)
        relevancies.append(query_relevancies)
    return relevancies

def tokenize(sentence: str, tokenizer, max_length):      
        
        tokens = tokenizer(sentence, padding = "max_length", max_length = max_length)

        tokens = tokens["input_ids"]

        return tokens

def createIndexDict(corpus):
    doc_id_to_index = {}
    for index, document in enumerate(corpus.all_documents['document_id']):
            for doc_id in document:
                doc_id_to_index[doc_id] =  index

    return doc_id_to_index

def getDocumentText(doc_id_to_index, document_id, corpus):
    idx = doc_id_to_index[document_id]
    subIdx = corpus.all_documents['document_id'][idx].index(document_id)
    return corpus.all_documents['text'][idx][subIdx]


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

def run_inference(self, all_queries, corpus, k_value, examples):
    doc_id_to_index = createIndexDict(corpus)
    
    tokenized_questions_tensor, tokenized_documents_tensor, question_order, document_order = self.tokenize_questions_and_docs(all_queries=all_queries, 
                                                                                                                examples=examples, doc_id_to_index= doc_id_to_index, corpus = corpus)

    embedded_questions, embedded_documents = self.embed_questions_and_documents(tokenized_questions_tensor, tokenized_documents_tensor)

    question_cosine_scores = self.compute_cosine_scores(embedded_questions, embedded_documents)

    top_k_documents, top_k_cosine_scores = self.compute_top_k_docs(question_cosine_scores, k_value, document_order)
    
    print(top_k_documents)

    return top_k_documents, top_k_cosine_scores