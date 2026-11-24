from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
import math
from collections import Counter
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(TFIDF, self).__init__(documents, index_path)
        self.corpus_vocabulary = self.GetCorpusVocabulary()
        # Create a mapping from terms to indices
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}
        self.document_frequencies = self.GetDocumentFrequencies()
        self.idf = self.GetInverseDocumentFrequencies()

        # Only run if we are in TFIDF class
        if type(self) == TFIDF:
            self.tfidf_matrix = self.GetDocumentsTFIDFVectors()

    def PreprocessText(self, text: str) -> list[str]:
        """
        Remove unwanted charatcters from text and lowercase
        """
        for char in '[]()".\'':
            text = text.replace(char, '')
        return text.lower().split()

    def PreprocessQuery(self, query: str) -> list[str]:
        return [t for t in query.lower().split() if t.isalpha()]

    def GetQueryVocabulary(self, query: str) -> list[str]:
        """
        Returns all unique words in a query
        """
        return list(set(self.PreprocessQuery(query)))

    def GetDocumentVocabulary(self, document: Document) -> list[str]:
        """
        Returns all unique words in a document
        """
        return list(set(self.PreprocessText(document.GetText())))
    
    def GetCorpusVocabulary(self) -> set[str]:
        """
        Returns all unique words in ALL documents.
        """
        # Use map function for more efficient processing
        processed_texts = map(lambda doc: set(self.PreprocessText(doc.GetText())), self.index.GetDocuments())
        corpus_vocabulary = set().union(*processed_texts)
        return corpus_vocabulary

    def GetDocumentFrequencies(self) -> dict[str, int]:
        """
        Returns a dict with terms and counts of how many documents the term occurs in (counted once per document).
        """
        document_frequencies = {}
        for document in self.index.GetDocuments():
            for term in self.GetDocumentVocabulary(document):
                document_frequencies[term] = document_frequencies.get(term, 0) + 1
        return document_frequencies


    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        n_documents = len(self.index.GetDocuments())
        return {term: math.log(n_documents / df) for term, df in self.document_frequencies.items()}


    def GetDocumentTermCounts(self, document: Document):
        """
        Returns a dict on the form {term: num_occurences} for an input Document.
        """
        return dict(Counter(self.PreprocessText(document.GetText())))

    def GetDocumentsTFIDFVectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the td-idf score of the j'th term in the i'th document.
        """
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)

        rows, cols, values = [], [], []
        for i, document in enumerate(self.index.GetDocuments()):
            counts = self.GetDocumentTermCounts(document)
            if not counts:
                continue
            max_count = max(counts.values())
            for term, count in counts.items():
                tf = count / max_count
                rows.append(i)
                cols.append(self.term_to_index[term])
                values.append(tf * self.idf[term])

        # Convert to CSR format for efficient row slicing
        tfidf_matrix = csr_matrix((values, (rows, cols)), shape=(n_documents, n_terms), dtype=np.float32)
        return tfidf_matrix
    
    def QueryToVector(self, queries: list[str]):
        """ Converts a query or batch of queries into a sparse vector matching the vocab size with counts of how many times the term occurs.
        
        Args:
            query (str): The query to convert
        
        Returns:
            csr_matrix: A sparse vector representation of the query
        """
        query_vector = lil_matrix((len(queries), len(self.corpus_vocabulary)), dtype=np.float32)

        for i, query in enumerate(queries):
            for term in self.PreprocessQuery(query):
                j = self.term_to_index.get(term)
                if j is not None:
                    query_vector[i, j] += 1

        return query_vector.tocsr()

    #@time_func
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document

            num_queries x num_documents where (i, j) = ∑_i TF-IDF(q_i, D_j; D) 
        
        """
        # Convert the queries to a vector
        query_vectors = self.QueryToVector(queries) # n_queries x vocab size
        scores = query_vectors.dot(self.tfidf_matrix.T).toarray() # n_queries x n_documents
        return scores
    