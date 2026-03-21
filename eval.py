"""
PromptAutopsy — Retrieval Evaluation
=====================================
Runs all 18 eval queries through retrieve()
Computes Precision@K, Recall@K, F1@K at K=3 and K=5
Prints a report by failure mode + overall scores
"""

from retrieve import retrieve
from eval_dataset import EVAL_DATA, precision_at_k, recall_at_k, f1_at_k, failure_mode_precision
from collections import defaultdict

K_VALUES = [3, 5]

def run_eval():
    results = []
    for item in EVAL_DATA:
        item_result = {
            "id": item["id"],
            "failure_mode": item["failure_mode"],
            "query": item["query"],
            "scores": {}
        }
        for k in K_VALUES:
            nodes = retrieve(item["query"], k)
            retrieved_sources = [node.metadata["source"] for node in nodes]
            retrieved_modes = [node.metadata["failure_mode"] for node in nodes]
            precision = precision_at_k(retrieved_sources, item["relevant_sources"], k)
            r = recall_at_k(retrieved_sources, item["relevant_sources"], k)
            f1 = f1_at_k(precision, r)
            fm = failure_mode_precision(retrieved_modes, item["relevant_failure_modes"],k)

            item_result["scores"][k] = {
                "precision"            : precision,
                "recall"               : r,
                "f1"                   : f1,
                "failure_mode_precision": fm
            }
        results.append(item_result)
    return results

def print_report(results: list):
    by_mode = defaultdict(list)
    for item in results:
        for k in K_VALUES:
            by_mode[item["failure_mode"]].append({
                "k": k,
                "precision": item["scores"][k]["precision"],
                "recall": item["scores"][k]["recall"],
                "f1": item["scores"][k]["f1"],
                "failure_mode_precision": item["scores"][k]["failure_mode_precision"]
            })
    
    for k in K_VALUES:
        print(f"\n{'='*55}")
        print(f"  K = {k}")
        print(f"{'='*55}")
        print(f"{'Failure Mode':<28} {'P@K':>6} {'R@K':>6} {'F1':>6} {'FM@K':>6}")
        print(f"{'-'*55}")

        all_p, all_r, all_f1, all_fm = [], [], [], []
        for mode,scores in by_mode.items():
            k_scores = [s for s in scores if s['k'] == k]
            avg_p = sum(s["precision"] for s in k_scores) / len(k_scores)
            avg_r = sum(s["recall"] for s in k_scores) / len(k_scores)
            avg_f1 = sum(s["f1"] for s in k_scores) / len(k_scores)
            avg_fm = sum(s["failure_mode_precision"] for s in k_scores) / len(k_scores)

            print(f"{mode:<28} {avg_p:>6.2f} {avg_r:>6.2f} {avg_f1:>6.2f} {avg_fm:>6.2f}")

            all_p.append(avg_p)
            all_r.append(avg_r)
            all_f1.append(avg_f1)
            all_fm.append(avg_fm)

        print(f"{'-'*55}")
        print(f"{'OVERALL':<28} {sum(all_p)/len(all_p):>6.2f} {sum(all_r)/len(all_r):>6.2f} {sum(all_f1)/len(all_f1):>6.2f} {sum(all_fm)/len(all_fm):>6.2f}")

if __name__ == "__main__":
    print("\n🔬 PromptAutopsy — Retrieval Evaluation")
    print("Running 18 queries across 5 failure modes...\n")
    results = run_eval()
    print_report(results)
    

    



