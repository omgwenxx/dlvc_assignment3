import collections
import re
import string
import csv


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