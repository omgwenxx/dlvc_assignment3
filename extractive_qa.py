import collections
from data_loading import *
from transformers import DistilBertTokenizer
from transformers import DistilBertForQuestionAnswering, JsonPipelineDataFormat, AutoTokenizer, \
    AutoModelForQuestionAnswering
import torch
import csv
import re
import string
import time
from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data.dataloader import PyTorchDataLoader
prepare_environment(Params({})) # sets the seeds to be fixed

import torch

from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from data_loading import *
from model_knrm import *
from model_conv_knrm import *
from model_tk import *
from util import *
from allennlp.nn.util import move_to_device
from util import *

# PART 2
def compute_f1(a_gold, a_pred):
    '''
    F1 = 2*precision*recall/(precision+recall)
    Precision = tp/(tp+fp)
    Recall = tp/(tp+fn)
    TP =  number of tokens* that are shared between the correct answer and the prediction
    FP = number of tokens that are in the prediction but not in the correct answer
    FN = number of tokens that are in the correct answer but not in the prediction.
    https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    '''
    gold_toks = normalize_answer(a_gold).split(" ")
    pred_toks = normalize_answer(a_pred).split(" ")
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if (len(gold_toks) == 0 or len(pred_toks) == 0):
        # If either no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_exact(a_gold, a_pred):
    '''
    Computes exact match, either 1 if match, 0 otherwise.
    '''
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def save_results(self, path: str):
    with open(path, "w", encoding="utf-8") as output:
        tsv_writer = csv.writer(output, delimiter='\t')
        for queryid, documentid, question, answer in self.results:
            tsv_writer.writerow([queryid, documentid, question, answer])

class ExtractiveQA:
    def __init__(self):
        self.reranking_results = {}
        self.questions = {}
        self.answers = []
        self.answers_queries = []
        self.results = []
        self.documents = {}
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("Using " + self.device)

        ##### BERT MODEL #####
        #self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        #self.model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad").to(self.device)

        ##### DistilBERT MODEL #####
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
        self.model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad").to(
            self.device)

        self.load_answers()


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
                contexts.append(line_parts[4])  # equivalent to our document text
                questions.append(line_parts[3])  # query_question
                answers.append(line_parts[6:])  # answer_text
                self.questions[line_parts[0]] = line_parts[3]

        return queryid, documentid, contexts, questions, answers

    def load_answers(self):
        '''
        read in FiRA gold-label dataset
        queryid documentid relevance-grade text-selection (multiple answers possible, split with tab)
        '''
        print("Loading files from " + "fira.qrels.qa-answers.tsv")
        with open(cached_path("../data/fira.qrels.qa-answers.tsv"), "r", encoding="utf8") as data_file:
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                queryid = line_parts[0]
                self.answers_queries.append(queryid)

    def load_reranking_results(self):
        print("Loading files from " + "../results/reranking_knrm.txt")
        with open(cached_path("../results/reranking_knrm.txt"), "r", encoding="utf8") as data_file:
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                queryid = line_parts[0]
                documentid = line_parts[1]
                self.reranking_results[str(queryid)] = documentid

    def load_documents(self):
        with open(cached_path("../data/msmarco_tuples.test.tsv"), "r", encoding="utf8") as data_file:
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                documentid = line_parts[1]
                text = line_parts[3:]  # answer_text
                self.documents[documentid] = str(text)

    def run_inference(self):
        '''
        Run inference using DistilBERT and return answers.
        '''
        queryid, documentid, document_text, questions, answers = self.load_data("../data/fira.qrels.qa-tuples.tsv")

        for queryid, documentid, question, text in zip(queryid, documentid, questions, document_text):
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
            self.results.append([queryid, documentid, question, answer])

    def get_reranking_results(self):
        queryid, documentid, document_text, questions, answers = self.load_data("../data/fira.qrels.qa-tuples.tsv")
        for queryid in self.answers_queries:
            if queryid not in self.reranking_results.keys():
                continue
            question = self.questions[queryid]
            documentid = self.reranking_results[queryid]
            if documentid not in self.documents.keys():
                continue
            text = self.documents[documentid]
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
            self.results.append([queryid, documentid, question, answer])

    def predict_answer(self, queryid, documentid, question, text):
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
        # print(f"Question: {question}")
        # print(f"Answer: {answer}")
        return answer

    def save_results(self, path: str):
        with open(path, "w", encoding="utf-8") as output:
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
        self.reranking_results = {}
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
    inference = False
    compute_reranking = False
    evaluate_reranking = True

    if evaluate_reranking:
        eval = Eval()
        print("Results for KNRM Reranking")
        eval.load_results("../results/knrm_results.tsv")
        f1, match, zeros = eval.compute_evaluation()
        print(f"Average F1-Score of {round(f1, 3)}")
        print(f"Average Exact-Match-Score of {round(match, 3)}")
        print(f"Total number of zero elements: {round(zeros, 3)}\n")

    if compute_reranking:
        qa = ExtractiveQA()
        qa.load_reranking_results()
        qa.load_documents()
        qa.get_reranking_results()
        qa.save_results("../results/knrm_results.tsv")

    if inference:
        startTime = time.time()
        qa = ExtractiveQA()
        qa.run_inference()
        executionTime = (time.time() - startTime)
        print('Execution time in seconds: ' + str(executionTime))
        qa.save_results("./results/bert_runtime_check")

    if evaluate:
        eval = Eval()
        print("Results for DistilBERT")
        eval.load_results("./results/extractive_qa_results_distilled_squad.tsv")
        f1_distil, match_distil, zeros_distil = eval.compute_evaluation()
        print(f"Average F1-Score of {round(f1_distil, 3)}")
        print(f"Average Exact-Match-Score of {round(match_distil, 3)}")
        print(f"Total number of zero elements: {round(zeros_distil, 3)}\n")

        eval.load_results("./results/extractive_qa_results_bert.tsv")
        f1_bert, match_bert, zeros_bert = eval.compute_evaluation()
        print("BERT results")
        print(f"Average F1-Score of {round(f1_bert, 3)}")
        print(f"Average Exact-Match-Score of {round(match_bert, 3)}")
        print(f"Total number of zero elements: {round(zeros_bert, 3)}\n")

