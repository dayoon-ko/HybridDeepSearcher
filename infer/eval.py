import re
import json
import argparse
import os
import time
import numpy as np
import string
import glob
from transformers import AutoTokenizer
from collections import Counter

# Load tokenizer for token counting
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

def replace_search_result(text):
    pattern = r'<\|begin_search_results\|>([\s\S]*?)<\|end_search_results\|>'
    matches = re.findall(pattern, text)
    for match in matches:
        text = text.replace(match, "")
    return text

def find_search_calls(item):
    num_search_queries = 0
    num_search_cycles = 0
    output_history_set = set()
    output_history = item["output_history"]
    search_counts = item["search_count"]
    for output, search_count in zip(output_history, search_counts):
        output_history_set.add(output)
        pattern = r'<\|begin_search_queries\|>([\s\S]*?)<\|end_search_queries\|>'
        matches = re.findall(pattern, output)
        if matches:
            num_search_queries += search_count
    num_search_cycles = len(output_history_set)
    return num_search_queries, num_search_cycles

def extract_answer(output):
    # \boxed{...} 
    pattern = r'\\boxed\{(.*)\}'
    matches = re.findall(pattern, output)
    if matches:
        pred = matches[-1]
        pred = pred.strip("()").strip("{").strip("}")
        pred = pred.replace("\\text", "").replace("{", "").replace("}", "")
        return pred
    return ''

def normalize_answer(text):
    text = text.lower()
    text = " ".join(text.strip().split())
    return text

def normalize_token(token):
    for i in ["begin{aligned}", "end{aligned}", "\\", "&", "{", "}", "(", ")", ":", "\"", "'", ".", ",", ";", "!", "?", "\\text"]:
        token = token.replace(i, "")
    return token

def compute_metrics(pred, gt, dataset_name="musique"):
    # normalize
    pred = normalize_answer(pred)
    gt = normalize_answer(gt)
    em = int(pred == gt)
    acc = int(gt in pred)
    pred_tokens = pred.split()
    gt_tokens = gt.split()
    if dataset_name == "fanoutqa":
        pred_tokens = [normalize_token(token) for token in pred_tokens]
        pred_tokens = [token for token in pred_tokens if token != ""]
        gt_tokens = [normalize_token(token) for token in gt_tokens]
        gt_tokens = [token for token in gt_tokens if token != ""]
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        f1 = 0
    else:
        precision = 1.0 * num_same / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = 1.0 * num_same / len(gt_tokens) if len(gt_tokens) > 0 else 0
        if (precision + recall) == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall) / (precision + recall)
    math_equal = em

    return {
        "em": em * 100,
        "acc": acc * 100,
        "f1": f1 * 100,
        "math_equal": math_equal
    }


if __name__ == "__main__":

    # Parse command-line arguments for flexibility
    parser = argparse.ArgumentParser(description="Evaluate model outputs with optional backoff.")
    parser.add_argument('--output_dir', '-o', type=str, default="results", help='Directory of the model output JSON file.')
    parser.add_argument('--datasets', nargs='+', help='Dataset names')
    args = parser.parse_args()

    for dataset in args.datasets:
        output_dir = f"{args.output_dir}/{dataset}"
        output_path = f"{output_dir}/output.json"
        output_metrics_path = f"{output_path}.metrics.json"
        
        if not os.path.exists(output_path):
            all_outputs = []
            for i in glob.glob(f"{output_dir}/output_*.json"):
                with open(i, "r", encoding='utf-8') as f:
                    all_outputs.extend(json.load(f))
            with open(output_path, "w", encoding='utf-8') as f:
                json.dump(all_outputs, f, indent=4)
        
        with open(output_path, "r", encoding='utf-8') as f:
            outputs = json.load(f)
            
        overall_metric = {
            "f1": [],
            "num_valid_answer": 0,
            "generated_tokens": [],
            "search_calls": [],
            "search_cycles": [],
            "accumulated_f1_turns": {i: [] for i in range(1, 11)},
            "accumulated_f1_calls": {i: [] for i in range(2, 17)},
            "accumulated_f1_generated_tokens": {i: [] for i in range(300, 1501, 300)},
        }   
        
        for item in outputs:
            
            # Get answer
            if dataset == "musique":
                answer = item["answer"]
            elif dataset == "fanoutqa":
                answer = str(item["answer"])
            elif dataset == "frames":
                answer = item["answer"]
            elif dataset == "med_browse_comp":
                if item["answer"] is None:
                    answer = "DATE: NA"
                else:
                    answer = item["answer"]
            elif dataset == "browse_comp":
                answer = item["Answer"]
                
            # Get response and prediction
            response = item["output"]
            pred = item["generated_answer"]
            
            # Count tokens (remove search results for token counting)
            message_for_tokens = replace_search_result(response)
            tokens = tokenizer.encode(message_for_tokens)
            generated_tokens = len(tokens)
                
            metric = compute_metrics(pred, answer, dataset)
            num_search_queries, num_search_cycles = find_search_calls(item)
            
            if len(response) > 0:
                overall_metric["num_valid_answer"] += 1
            overall_metric["f1"].append(metric["f1"])
            overall_metric["search_calls"].append(num_search_queries)
            overall_metric["search_cycles"].append(num_search_cycles)
            overall_metric["generated_tokens"].append(generated_tokens)
            
            # accumulated_f1_interval_turns (by search cycles, intervals [1,2,4,8])
            num_search_cycles_clamped = max(1, min(10, num_search_cycles))
            for i in range(1, num_search_cycles_clamped):
                overall_metric["accumulated_f1_turns"][i].append(0)
            for i in range(num_search_cycles_clamped, 11):
                overall_metric["accumulated_f1_turns"][i].append(metric["f1"])
            
            # accumulated_f1_calls (by search calls, 2-16)
            calls_clamped = max(2, min(16, num_search_queries))
            for i in range(2, calls_clamped):
                overall_metric["accumulated_f1_calls"][i].append(0)
            for i in range(calls_clamped, 17):
                overall_metric["accumulated_f1_calls"][i].append(metric["f1"])
            
            # accumulated_f1_generated_tokens (by generated tokens, 0-2999)
            tokens_clamped = max(300, min(1500, generated_tokens))
            tokens_clamped = tokens_clamped // 300 * 300
            for i in range(300, tokens_clamped, 300):
                overall_metric["accumulated_f1_generated_tokens"][i].append(0)
            for i in range(tokens_clamped, 1501, 300): # 3001
                overall_metric["accumulated_f1_generated_tokens"][i].append(metric["f1"])
                
        # Calculate averages and totals for new metrics
        for k in ["generated_tokens", "search_calls", "search_cycles"]:
            if k in overall_metric and type(overall_metric[k]) == list:
                values = overall_metric[k]
                overall_metric[f"{k}_avg"] = sum(values) / len(values) if len(values) > 0 else 0
                overall_metric[f"{k}_total"] = sum(values)
        del overall_metric["generated_tokens"]
        del overall_metric["search_calls"]
        del overall_metric["search_cycles"]
        
        for k in range(1, 11):
            if len(overall_metric["accumulated_f1_turns"][k]) > 0:
                overall_metric["accumulated_f1_turns"][k] = sum(overall_metric["accumulated_f1_turns"][k]) / len(outputs)
            else:
                overall_metric["accumulated_f1_turns"][k] = 0
                
        # Calculate accumulated_f1_calls (2-16)
        for k in range(2, 17):
            if len(overall_metric["accumulated_f1_calls"][k]) > 0:
                overall_metric["accumulated_f1_calls"][k] = sum(overall_metric["accumulated_f1_calls"][k]) / len(outputs)
            else:
                overall_metric["accumulated_f1_calls"][k] = 0
        
        # Calculate accumulated_f1_generated_tokens (by generated tokens, 300-1500)
        for k in range(300, 1501, 300):
            if len(overall_metric["accumulated_f1_generated_tokens"][k]) > 0:
                overall_metric["accumulated_f1_generated_tokens"][k] = sum(overall_metric["accumulated_f1_generated_tokens"][k]) / len(outputs)
            else:
                overall_metric["accumulated_f1_generated_tokens"][k] = 0
                
        # Calculate averages for existing metrics
        for k, v in overall_metric.items():
            if type(v) == list and k in ["f1"]:
                overall_metric[k] = sum(v) / len(v)
                
        # AUC-AT 
        overall_metric["auc_at"] = sum(overall_metric["accumulated_f1_turns"].values()) / len(overall_metric["accumulated_f1_turns"])
        overall_metric["auc_ac"] = sum(overall_metric["accumulated_f1_calls"].values()) / len(overall_metric["accumulated_f1_calls"])
        overall_metric["auc_al"] = sum(overall_metric["accumulated_f1_generated_tokens"].values()) / len(overall_metric["accumulated_f1_generated_tokens"]) * 500
        
        with open(output_metrics_path, "w", encoding="utf-8") as f:
            json.dump(overall_metric, f, indent=4)
        
        # Print key metrics
        print("Evaluation Results:")
        for k in ["f1", "auc_at", "auc_ac", "auc_al", "generated_tokens_avg"]:
            if k in overall_metric:
                print(f"{k}: {overall_metric[k]:.2f}")
        
