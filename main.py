import hydra
from src.data_utils.corpus import Corpus
from src.retrievers.tf_idf import TFIDF
from src.retrievers.bm25 import BM25
from src.retrievers.dpr import DPR
from src.data_utils.query import Query
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate
from src.retrievers.base_retriever import Retriever
from src.training.trainer import Trainer
from src.retrievers.utils import tokenize, getDocumentText, createIndexDict
import torch

@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')

    ### Prepare queries
    train_queries = [Query(corpus.train['question'][idx], 
                           id=corpus.train['query_id'][idx], 
                           relevant_document_ids=corpus.relevant_ids[corpus.train['query_id'][idx]]
                           ) 
                        for idx in range(corpus.train.num_rows)
                    ]
    print("train queries: ", len(train_queries))

    val_queries =   [Query(corpus.val['question'][idx], 
                         id=corpus.val['query_id'][idx], 
                         relevant_document_ids=corpus.relevant_ids[corpus.val['query_id'][idx]]
                           ) 
                        for idx in range(corpus.val.num_rows)
                    ]

    print("val queries: ", len(val_queries))

    all_queries = train_queries + val_queries 

    if cfg.dataset.name == 'fiqa':
        test_queries =  [Query(corpus.test['question'][idx], 
                            id=corpus.test['query_id'][idx], 
                            relevant_document_ids=corpus.relevant_ids[corpus.test['query_id'][idx]]
                            ) 
                            for idx in range(corpus.test.num_rows)
                        ]

        print("test queries: ", len(test_queries))
    
        all_queries += test_queries

    print("all queries amount: ", len(all_queries))


    
    retriever: Retriever = None
    
    ### Initialize retriever
    if cfg.retriever.method == 'tf-idf':
        retriever = TFIDF(documents=corpus.all_documents)
    elif cfg.retriever.method == 'bm25':
        retriever = BM25(documents=corpus.all_documents)
    elif cfg.retriever.method == 'dpr':
        retriever = DPR(cfg=cfg.model)
        k_value = 3
        examples = 10

        top_k_documents = retriever.run_inference(all_queries, corpus, k_value, examples)

        for index, question in enumerate(top_k_documents):
            print("most relevant documents for question ", index, ": ", question)
            print("actual most relevant documents for question ", index, ": ", all_queries[index].relevant_document_ids)



        

        #if not cfg.model.get('query_encoder_weights_path', None):
            # trainer = Trainer(corpus, train_queries)
            # print("query encoder: ", retriever.query_encoder)
            # print("document encoder: ", retriever.document_encoder)
            # trainer.train(query_encoder=retriever.query_encoder, document_encoder= retriever.document_encoder, cfg = cfg)
        #else:
            
                    

    
        
    raise NotImplementedError

    ### Retrieve top-k relevant queries
    retriever_train_relevancies = retrieveQueryAndGetRelevancies(retriever, train_queries, k=cfg.top_k)
    retriever_val_relevancies = retrieveQueryAndGetRelevancies(retriever, val_queries, k=cfg.top_k)
    retriever_test_relevancies = retrieveQueryAndGetRelevancies(retriever, test_queries, k=cfg.top_k)

    ### Evaluate
    train_scores = Evaluate(retriever_train_relevancies, train_queries)
    val_scores = Evaluate(retriever_val_relevancies, val_queries)
    test_scores = Evaluate(retriever_test_relevancies, test_queries)

    print(f'Train results: {train_scores}')
    print(f'Val results: {val_scores}')
    print(f'Test results: {test_scores}')

if __name__ == '__main__':
    main()