import csv
from core_metrics import *

def get_qrels(filepath):
    qrels = {}
    with open(filepath, mode='r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            qid = int(row[0])
            did = int(row[2])
            if qid not in qrels:
                qrels[qid] = []
            qrels[qid].append(did)
    return qrels

def get_metrics(pred_results_seb, qrel_path):
    qrels_seb = load_qrels(qrel_path)

    for i, q_id in enumerate(pred_results_seb):
        # sorted_vals contains tuples with (id, predValue) --> has to be sorted by [1]
        sorted_vals_seb = sorted(pred_results_seb[str(q_id)], key=lambda x: x[1], reverse=True)
        pred_results_seb[str(q_id)] = [i[0] for i in sorted_vals_seb]  # Hofstaetters metrics need a list instead of tuples

    return calculate_metrics_plain(pred_results_seb, qrels_seb, binarization_point=1.0, return_per_query=False)

def save_evalresults(results, path: str):
    with open(path, "w+", encoding="utf-8") as f:
        for key in results.keys():
            f.write(f"Metric {key}: {round(results[key],3)}")
            f.write('\n')