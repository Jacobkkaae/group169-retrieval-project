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

def tokenize(sentence: str, tokenizer):      
        
        tokens = tokenizer(sentence, padding = "max_length", max_length = 1000)

        print(tokens["input_ids"])

        return tokens["input_ids"]

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

     