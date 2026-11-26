import torch
import torch.nn as nn
from src.training.loss import DPRLoss
from src.training.dataset_real import Dataset_real
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import random
from src.retrievers.models.encoder import TransformerEncoder
from src.retrievers.dpr import DPR

class Trainer:
    def __init__(self, retriever: DPR, corpus, cfg):
        self.criterion = DPRLoss()
        self.corpus = corpus
        self.cfg = cfg
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.retriever = retriever
        self.optimizer = torch.optim.AdamW(self.retriever.parameters(), lr=0.0001)


    def train(self):

        dataset = Dataset_real(self.corpus.ds['train'])

        dataloader = DataLoader(dataset, batch_size=self.cfg.model.batch_size,
                        shuffle=True, num_workers=0, collate_fn=self.collate_function)

        loss_fn = nn.CrossEntropyLoss(reduce=True)
        


        for index in range(5): 

            example = next(iter(dataloader))

            tokenized_questions_tensor = torch.tensor(example['questions'])
            tokenized_texts_tensor = torch.tensor(example['texts'])
            query_ids = example['query_ids']
            doc_ids = example['doc_ids']


            embedded_questions = self.retriever.query_encoder.forward(tokenized_questions_tensor)
            embedded_documents = self.retriever.document_encoder.forward(tokenized_texts_tensor)

            question_cosine_scores = self.retriever.compute_cosine_scores(embedded_questions, embedded_documents)

            targets = torch.arange(self.cfg.model.batch_size)

            loss = loss_fn(question_cosine_scores, targets)

            print("question_cosine_scores: ", question_cosine_scores)
            print("targets: ", targets)
            print("loss: ", loss.item())

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()





    def collate_function(self, batch):
        
        tokenized_questions = []
        query_ids = []
        tokenized_texts = []
        document_ids = []

        for index, example in enumerate(batch):
            randomly_chosen_doc = random.randint(0, len(example['text'])-1)
            tokenized_questions.append(self.tokenize(example['question'], self.cfg.model.max_token_length))
            query_ids.append(example['query_id'])
            tokenized_texts.append(self.tokenize(example['text'][randomly_chosen_doc], self.cfg.model.max_token_length)) #brug rand i stedt
            document_ids.append(example['document_id'][randomly_chosen_doc])

        return { "questions": tokenized_questions, "texts": tokenized_texts, "query_ids": query_ids, "doc_ids":document_ids }

    def tokenize(self, sentence, max_length: int):        
        
        tokens = self.tokenizer(sentence, padding = "max_length", max_length = max_length)

        return tokens["input_ids"]

        


    