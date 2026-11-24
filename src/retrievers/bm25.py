import math
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from src.retrievers.tf_idf import TFIDF
from src.data_utils.document import Document

class BM25(TFIDF):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b
        self.bm25_idf = self.GetBM25InverseDocumentFrequencies()
        self.bm25_matrix = self.GetBM25Matrix()

    def GetDocumentLengths(self):
        document_lengths = {}
        average_document_length = 0
        for document in self.index.GetDocuments():
            length = len(self.PreprocessText(document.GetText()))
            document_lengths[document] = length
            average_document_length += length
        return document_lengths, average_document_length/len(self.index.GetDocuments())
    
    def GetBM25InverseDocumentFrequencies(self) -> dict[str, float]:
        n_documents = len(self.index.GetDocuments())
        return {term: math.log((n_documents + 1) / (df + 0.5)) for term, df in self.document_frequencies.items()}

    def GetBM25Matrix(self):
        documents = self.index.GetDocuments()
        document_lengths, average_document_length = self.GetDocumentLengths()

        rows, cols, values = [], [], []
        for i, document in enumerate(documents):
            length_norm = self.k1 * (1 - self.b + self.b * document_lengths[document] / average_document_length)
            for term, f in self.GetDocumentTermCounts(document).items():
                tf = f * (self.k1 + 1) / (f + length_norm)
                rows.append(i)
                cols.append(self.term_to_index[term])
                values.append(tf * self.bm25_idf[term])

        return csr_matrix((values, (rows, cols)), shape=(len(documents), len(self.corpus_vocabulary)), dtype=np.float32)

    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document
        """
        query_vector = self.QueryToVector(queries)
        scores = query_vector.dot(self.bm25_matrix.T).toarray()
        return scores