import torch
import torch.nn as nn
from src.training.loss import DPRLoss
from src.training.dataset_real import Dataset_real
from torch.utils.data import DataLoader
from transformers import BertTokenizer

class Trainer:
    def __init__(self, corpus, train_queries):
        self.criterion = DPRLoss()
        self.corpus = corpus
        self.train_queries = train_queries
        #self.optimizer_query_encoder = torch.optim.Adam(query_encoder.parameters(), lr=0.001, momentum=0.9)
        #self.optimizer_document_encoder = torch.optim.Adam(document_encoder.parameters(), lr=0.001, momentum=0.9)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    def train(self, query_encoder: nn.Module, document_encoder: nn.Module, cfg):

        training_dataset = Dataset_real(self.corpus.ds['train'])

        dataloader = DataLoader(training_dataset, batch_size=1,
                        shuffle=True, num_workers=0)


        for index, example in enumerate(dataloader): 
            print("example question: ", example['question'])

            token_ids_question = self.tokenize(sentence = example['question'], max_length=cfg.model.max_token_length)
            token_ids_question_tensor = torch.tensor(token_ids_question)
            query_result = query_encoder.forward(token_ids_question_tensor)

            tokenized_texts = []
            print("Length of example text: ", len(example['text']))
            for text in example['text']: 
                print("Example text: ", text)
                token_ids_document = self.tokenize(sentence = text, max_length=cfg.model.max_token_length)
                tokenized_texts.append(token_ids_document)

            print("tokenized texts: ", len(tokenized_texts))
            print("Amount of token IDS in first tokenized text: ", len(tokenized_texts[0]))
            
            token_ids_question_tensor = torch.tensor(tokenized_texts)
            print("Tokenized texts shape: ", token_ids_question_tensor.shape)
            document_result = document_encoder.forward(token_ids_question_tensor)



            raise NotImplementedError()

    
    def tokenize(self, sentence, max_length: int):        
        
        tokens = self.tokenizer(sentence, padding = "max_length", max_length = max_length)

        return tokens["input_ids"][0]

        


    