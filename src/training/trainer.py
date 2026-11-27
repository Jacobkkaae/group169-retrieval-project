import torch
import torch.nn as nn
from src.training.loss import DPRLoss
from src.training.dataset_real import Dataset_real
from torch.utils.data import DataLoader
import random
from src.retrievers.models.encoder import TransformerEncoder
from src.retrievers.dpr import DPR
from datasets import Dataset
from src.training.collator import RetrievalCollator
from transformers import get_linear_schedule_with_warmup, BertTokenizerFast

class Trainer:
    def __init__(self, retriever: DPR, corpus, cfg):
        self.criterion = DPRLoss()
        self.corpus = corpus
        self.cfg = cfg
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.retriever = retriever
        self.optimizer = torch.optim.AdamW(self.retriever.parameters(), lr=0.0001)


    def train(self):
        device = torch.device(self.cfg.model.device)

        dataset = self.corpus.ds['train']

        example = dataset[0]
        print("Fast tokenizer:", self.tokenizer.is_fast)
        print("Texts per question:", len(example['text']))
        print("Characters in first text:", len(example['text'][0]))
        print(dataset.features)

        dataset = dataset.map(self.pretokenize_function, batched=True, batch_size=1000)

        collator = RetrievalCollator(self.tokenizer)
        
        dataloader = DataLoader(dataset, batch_size=self.cfg.model.batch_size, shuffle=True, drop_last=True,
                        num_workers=4, persistent_workers=True, collate_fn=collator, pin_memory=False)

        loss_fn = nn.CrossEntropyLoss()
        
        epochs = 10

        total_steps = len(dataloader) * epochs

        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=200, num_training_steps=total_steps)

        #example = next(iter(dataloader))

        for epoch in range(epochs):
            for iteration, example in enumerate(dataloader):
            #for iteration in range (400):

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    question_input_ids = example['questions']['input_ids'].to(device, non_blocking=True)
                    question_attention_mask = example['questions']['attention_mask'].to(device, non_blocking=True)
                    text_input_ids = example['texts']['input_ids'].to(device, non_blocking=True)
                    text_attention_mask = example['texts']['attention_mask'].to(device, non_blocking=True)

                    query_ids = example['query_ids']
                    doc_ids = example['doc_ids']

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    question_dot_products = self.retriever.run_inference(question_input_ids, question_attention_mask,
                                                                         text_input_ids, text_attention_mask)

                targets = torch.arange(question_dot_products.size(0), device=device)

                loss = loss_fn(question_dot_products.float(), targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                accuracy = (question_dot_products.argmax(dim=1) == targets).float().mean()

                if iteration % 10 == 0:
                    print("At iteration: ", iteration)
                    print(f"Loss: {loss.item():.4f}")
                    print("Accuracy: ", accuracy.item())

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")



    def pretokenize_function(self, examples):

        tokenized_questions = self.tokenizer(examples['question'], truncation=True, max_length=64)['input_ids']
        tokenized_texts = self.tokenizer(examples['text'], truncation=True, max_length=512)['input_ids']

        return { "tokenized_question": tokenized_questions, "tokenized_text": tokenized_texts }
    # def collate_function(self, batch):

    #     tokenized_questions = []
    #     query_ids = []
    #     tokenized_texts = []
    #     document_ids = []

    #     for example in batch:
    #         randomly_chosen_doc = random.randrange(len(example['tokenized_texts']))
    #         tokenized_questions.append(example['tokenized_question'])
    #         query_ids.append(example['query_id'])
    #         tokenized_texts.append(example['tokenized_texts'][randomly_chosen_doc])
    #         document_ids.append(example['document_id'][randomly_chosen_doc])

    #     padded_questions = self.tokenizer.pad({ "input_ids": tokenized_questions }, return_tensors="pt")
    #     padded_texts = self.tokenizer.pad({ "input_ids": tokenized_texts }, return_tensors="pt")

    #     return { "questions": padded_questions, "texts": padded_texts, "query_ids": query_ids, "doc_ids": document_ids }

    def tokenize(self, sentence, max_length: int):        
        
        tokens = self.tokenizer(sentence, padding = "max_length", max_length = max_length)

        return tokens["input_ids"]

        


    