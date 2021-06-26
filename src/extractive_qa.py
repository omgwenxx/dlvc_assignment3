from transformers import DistilBertTokenizer
from transformers import DistilBertForQuestionAnswering, AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import AlbertTokenizer, TFAlbertModel
from allennlp.common.file_utils import cached_path
import numpy as np
import time
import torch
from util import *
import enum

class Models(enum.Enum):
   DistilBERT = 1
   BERT = 2
   AlBERT = 3
   roBERTa = 4

class ExtractiveQA:
    def __init__(self, model: Models):
        self.questions = {}
        self.answers = []
        self.answers_queries = []
        self.results = []
        self.documents = {}
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("Using "+self.device)

        if model == Models.BERT:
            print("Initiliazing BERT model")
            self.name = "BERT"
            self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
            self.model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad").to(self.device)
        elif model == Models.DistilBERT:
            print("Initiliazing DistilBERT model")
            self.name = "DistilBERT"
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
            self.model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad").to(
                self.device)
        elif model == Models.AlBERT:
            print("Initiliazing AlBERT model")
            self.name = "AlBERT"
            #self.tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
            #self.model = TFAlbertModel.from_pretrained("albert-xxlarge-v2")
        elif model == Models.roBERTa:
            print("Initiliazing roBERTa model")
            self.name = "roBERTa"
            self.tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
            self.model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2").to(self.device)


    def load_data(self, file_path: str):
        '''
        read in FiRA gold-label dataset
        queryid documentid relevance-grade query-text document-text text-selection (multiple answers possible, split with tab)
        '''
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            queryid = []
            documentid = []
            contexts = []
            questions = []
            answers = []
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                queryid.append(line_parts[0])
                documentid.append(line_parts[1])
                questions.append(line_parts[3])  # query_question
                contexts.append(line_parts[4])  # equivalent to our document text
                answers.append(line_parts[6:])  # answer_text

        return queryid, documentid, contexts, questions, answers

    def run_inference(self):
        '''
        Run inference and return answers.
        '''
        queryid, documentid, document_text, questions, answers = self.load_data("./data/fira.qrels.qa-tuples.tsv")

        for queryid, documentid, question, text in zip(queryid, documentid, questions, document_text):
            inputs = self.tokenizer(question, text, add_special_tokens=True, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"].tolist()[0]
            answer_start_scores, answer_end_scores = self.model(**inputs)
            answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
            answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            self.results.append([queryid, documentid, question, answer])

    def get_results(self):
        return self.results

    def get_answer(self, question, text):
        inputs = self.tokenizer(question, text, add_special_tokens=True, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"].tolist()[0]
        answer_start_scores, answer_end_scores = self.model(**inputs)
        answer_start = torch.argmax(
            answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(
            answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        return answer

    def save_results(self):
        with open("./results/"+self.name+".tsv", "w", encoding="utf-8") as output:
            tsv_writer = csv.writer(output, delimiter='\t')
            for queryid, documentid, question, answer in self.results:
                tsv_writer.writerow([queryid, documentid, question, answer])

class Eval:
    def __init__(self):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.answers = {}
        self.results = {}
        self.scores = []
        self.load_answers()

    def load_answers(self):
        '''
        read in FiRA gold-label dataset
        queryid documentid relevance-grade text-selection (multiple answers possible, split with tab)
        '''
        print("Loading files from " + "../data/fira.qrels.qa-tuples.tsv")
        with open(cached_path("../data/fira.qrels.qa-tuples.tsv"), "r", encoding="utf8") as data_file:
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                queryid = line_parts[0]
                documentid = line_parts[1]
                answers = line_parts[6:]  # answer_text

                self.answers[str(queryid)] = [documentid, answers]

    def load_results(self, file_path: str):
        '''
        read in FiRA gold-label answers dataset
        queryid documentid relevance-grade text-selection (multiple answers possible, split with tab)
        '''
        print("Loading files from " + file_path)
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                queryid = line_parts[0]
                documentid = line_parts[1]
                answers = line_parts[3]  # answer_text

                self.results[queryid] = [documentid, answers]

    def compute_evaluation(self):
        '''
        Computes the average how many words overlap for each result returns by the extractice qa model and the answers provided
        by the fira.qrels.qa-tuples.tsv dataset.
        '''
        f1_scores = []
        exact_match_scores = []
        print(f"Comparing {len(self.results.keys())} elements")

        for queryid in self.results.keys():
            answers = self.answers[queryid][1]
            result = self.results[queryid][1]
            query_f1_score = []
            query_exact_match_score = []
            for answer in answers:
                query_f1_score.append(compute_f1(answer, result))
                query_exact_match_score.append(compute_exact(answer, result))
            f1_scores.append(max(query_f1_score))
            exact_match_scores.append(max(query_exact_match_score))

        return np.average(np.array(f1_scores)), np.average(np.array(exact_match_scores)), np.count_nonzero(np.array(f1_scores) == 0)

if __name__ == "__main__":
    evaluate = False
    inference = True

    if inference:
        qa = ExtractiveQA(Models.AlBERT)
        startTime = time.time()
        qa.run_inference()
        executionTime = (time.time() - startTime)
        print('Execution time of inference in seconds: ' + str(executionTime))
        qa.save_results()

    if evaluate:
        eval = Eval()
        print("Results for DistilBERT")
        eval.load_results("../results/DistilBERT.tsv")
        f1_distil, match_distil, zeros_distil = eval.compute_evaluation()
        print(f"Average F1-Score of {round(f1_distil, 3)}")
        print(f"Average Exact-Match-Score of {round(match_distil, 3)}")
        print(f"Total number of zero elements: {round(zeros_distil, 3)}\n")

        eval.load_results("../results/BERT.tsv")
        f1_bert, match_bert, zeros_bert = eval.compute_evaluation()
        print("BERT results")
        print(f"Average F1-Score of {round(f1_bert, 3)}")
        print(f"Average Exact-Match-Score of {round(match_bert, 3)}")
        print(f"Total number of zero elements: {round(zeros_bert, 3)}\n")

