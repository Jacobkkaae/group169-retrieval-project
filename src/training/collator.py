import random

class RetrievalCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        tokenized_questions = []
        query_ids = []
        tokenized_texts = []
        document_ids = []

        for example in batch:
            tokenized_questions.append(example['tokenized_question'])
            query_ids.append(example['query_id'])
            tokenized_texts.append(example['tokenized_text'])
            document_ids.append(example['document_id'][0])

        padded_questions = self.tokenizer.pad({ "input_ids": tokenized_questions }, return_tensors="pt")
        padded_texts = self.tokenizer.pad({ "input_ids": tokenized_texts }, return_tensors="pt")

        return { "questions": padded_questions, "texts": padded_texts, "query_ids": query_ids, "doc_ids": document_ids }
