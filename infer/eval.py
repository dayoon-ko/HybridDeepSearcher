"""
Evaluation Script for HybridDeepSearcher

Computes metrics including:
- F1 Score
- Exact Match (EM)
- Accuracy
- AUC-AT (Area Under Curve - Accumulated F1 by Turns)
- AUC-AC (Area Under Curve - Accumulated F1 by Calls)
- AUC-AL (Area Under Curve - Accumulated F1 by Length)
"""

import re
import os
import json
import glob
import argparse

from transformers import AutoTokenizer
from collections import Counter
from typing import Dict, Any, List, Tuple


# ============================================================================
# Constants
# ============================================================================

TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Metric configuration
MAX_SEARCH_TURNS = 10
MIN_SEARCH_CALLS = 2
MAX_SEARCH_CALLS = 16
TOKEN_BINS = list(range(300, 1501, 300))


# ============================================================================
# Text Processing
# ============================================================================

def replace_search_result(text: str) -> str:
    """Remove search result blocks from text for token counting."""
    pattern = r'<\|begin_search_results\|>([\s\S]*?)<\|end_search_results\|>'
    matches = re.findall(pattern, text)
    for match in matches:
        text = text.replace(match, "")
    return text


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    text = text.lower()
    text = " ".join(text.strip().split())
    return text


def normalize_token(token: str) -> str:
    """Normalize a single token by removing special characters."""
    chars_to_remove = [
        "begin{aligned}", "end{aligned}", "\\", "&", 
        "{", "}", "(", ")", ":", "\"", "'", 
        ".", ",", ";", "!", "?", "\\text"
    ]
    for char in chars_to_remove:
        token = token.replace(char, "")
    return token


# ============================================================================
# Answer Extraction
# ============================================================================

def extract_answer(output: str) -> str:
    """Extract answer from \\boxed{...} format."""
    pattern = r'\\boxed\{(.*)\}'
    matches = re.findall(pattern, output)
    if matches:
        pred = matches[-1]
        pred = pred.strip("()").strip("{").strip("}")
        pred = pred.replace("\\text", "").replace("{", "").replace("}", "")
        return pred
    return ''


def get_ground_truth_answer(item: Dict[str, Any], dataset: str) -> str:
    """Get ground truth answer based on dataset format."""
    if dataset == "musique":
        return item["answer"]
    elif dataset == "fanoutqa":
        return str(item["answer"])
    elif dataset == "frames":
        return item["answer"]
    elif dataset == "med_browse_comp":
        return item["answer"] if item["answer"] is not None else "DATE: NA"
    elif dataset == "browse_comp":
        return item["Answer"]
    else:
        return item.get("answer", "")


# ============================================================================
# Search Statistics
# ============================================================================

def find_search_calls(item: Dict[str, Any]) -> Tuple[int, int]:
    """
    Count search queries and cycles from item history.
    
    Returns:
        Tuple of (num_search_queries, num_search_cycles)
    """
    num_search_queries = 0
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


# ============================================================================
# Metric Computation
# ============================================================================

def compute_metrics(pred: str, gt: str, dataset_name: str = "musique") -> Dict[str, float]:
    """
    Compute evaluation metrics between prediction and ground truth.
    
    Args:
        pred: Predicted answer string.
        gt: Ground truth answer string.
        dataset_name: Dataset name for special handling.
    
    Returns:
        Dictionary with em, acc, f1, and math_equal scores (scaled to 100).
    """
    pred = normalize_answer(pred)
    gt = normalize_answer(gt)
    
    em = int(pred == gt)
    acc = int(gt in pred)
    
    pred_tokens = pred.split()
    gt_tokens = gt.split()
    
    # Special token normalization for fanoutqa
    if dataset_name == "fanoutqa":
        pred_tokens = [normalize_token(t) for t in pred_tokens if normalize_token(t)]
        gt_tokens = [normalize_token(t) for t in gt_tokens if normalize_token(t)]
    
    # Compute F1
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        f1 = 0
    else:
        precision = num_same / len(pred_tokens) if pred_tokens else 0
        recall = num_same / len(gt_tokens) if gt_tokens else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "em": em * 100,
        "acc": acc * 100,
        "f1": f1 * 100,
        "math_equal": em
    }


# ============================================================================
# Accumulated Metrics
# ============================================================================

def initialize_accumulated_metrics() -> Dict[str, Any]:
    """Initialize the accumulated metrics structure."""
    return {
        "f1": [],
        "num_valid_answer": 0,
        "generated_tokens": [],
        "search_calls": [],
        "search_cycles": [],
        "accumulated_f1_turns": {i: [] for i in range(1, MAX_SEARCH_TURNS + 1)},
        "accumulated_f1_calls": {i: [] for i in range(MIN_SEARCH_CALLS, MAX_SEARCH_CALLS + 1)},
        "accumulated_f1_generated_tokens": {i: [] for i in TOKEN_BINS},
    }


def update_accumulated_f1_turns(
    overall_metric: Dict[str, Any], 
    num_search_cycles: int, 
    f1: float
) -> None:
    """Update accumulated F1 by search turns."""
    num_search_cycles_clamped = max(1, min(MAX_SEARCH_TURNS, num_search_cycles))
    
    for i in range(1, num_search_cycles_clamped):
        overall_metric["accumulated_f1_turns"][i].append(0)
    for i in range(num_search_cycles_clamped, MAX_SEARCH_TURNS + 1):
        overall_metric["accumulated_f1_turns"][i].append(f1)


def update_accumulated_f1_calls(
    overall_metric: Dict[str, Any], 
    num_search_queries: int, 
    f1: float
) -> None:
    """Update accumulated F1 by search calls."""
    calls_clamped = max(MIN_SEARCH_CALLS, min(MAX_SEARCH_CALLS, num_search_queries))
    
    for i in range(MIN_SEARCH_CALLS, calls_clamped):
        overall_metric["accumulated_f1_calls"][i].append(0)
    for i in range(calls_clamped, MAX_SEARCH_CALLS + 1):
        overall_metric["accumulated_f1_calls"][i].append(f1)


def update_accumulated_f1_tokens(
    overall_metric: Dict[str, Any], 
    generated_tokens: int, 
    f1: float
) -> None:
    """Update accumulated F1 by generated tokens."""
    tokens_clamped = max(TOKEN_BINS[0], min(TOKEN_BINS[-1], generated_tokens))
    tokens_clamped = (tokens_clamped // 300) * 300
    
    for i in TOKEN_BINS:
        if i < tokens_clamped:
            overall_metric["accumulated_f1_generated_tokens"][i].append(0)
        else:
            overall_metric["accumulated_f1_generated_tokens"][i].append(f1)


def finalize_metrics(overall_metric: Dict[str, Any], num_outputs: int) -> None:
    """Finalize metrics by computing averages."""
    # Compute averages for list metrics
    for k in ["generated_tokens", "search_calls", "search_cycles"]:
        if k in overall_metric and isinstance(overall_metric[k], list):
            values = overall_metric[k]
            overall_metric[f"{k}_avg"] = sum(values) / len(values) if values else 0
            overall_metric[f"{k}_total"] = sum(values)
        if k in overall_metric:
            del overall_metric[k]
    
    # Finalize accumulated F1 by turns
    for k in range(1, MAX_SEARCH_TURNS + 1):
        values = overall_metric["accumulated_f1_turns"][k]
        overall_metric["accumulated_f1_turns"][k] = sum(values) / num_outputs if values else 0
    
    # Finalize accumulated F1 by calls
    for k in range(MIN_SEARCH_CALLS, MAX_SEARCH_CALLS + 1):
        values = overall_metric["accumulated_f1_calls"][k]
        overall_metric["accumulated_f1_calls"][k] = sum(values) / num_outputs if values else 0
    
    # Finalize accumulated F1 by tokens
    for k in TOKEN_BINS:
        values = overall_metric["accumulated_f1_generated_tokens"][k]
        overall_metric["accumulated_f1_generated_tokens"][k] = sum(values) / num_outputs if values else 0
    
    # Average F1
    if overall_metric["f1"]:
        overall_metric["f1"] = sum(overall_metric["f1"]) / len(overall_metric["f1"])
    
    # Compute AUC metrics
    overall_metric["auc_at"] = (
        sum(overall_metric["accumulated_f1_turns"].values()) / 
        len(overall_metric["accumulated_f1_turns"])
    )
    overall_metric["auc_ac"] = (
        sum(overall_metric["accumulated_f1_calls"].values()) / 
        len(overall_metric["accumulated_f1_calls"])
    )
    overall_metric["auc_al"] = (
        sum(overall_metric["accumulated_f1_generated_tokens"].values()) / 
        len(overall_metric["accumulated_f1_generated_tokens"]) * 500
    )


# ============================================================================
# Data Loading
# ============================================================================

def load_outputs(output_dir: str, output_path: str) -> List[Dict[str, Any]]:
    """Load outputs from file, merging batch files if needed."""
    if not os.path.exists(output_path):
        all_outputs = []
        for batch_file in glob.glob(f"{output_dir}/output_*.json"):
            with open(batch_file, "r", encoding='utf-8') as f:
                all_outputs.extend(json.load(f))
        
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(all_outputs, f, indent=4)
    
    with open(output_path, "r", encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_dataset(dataset: str, output_dir: str) -> Dict[str, Any]:
    """
    Evaluate a single dataset.
    
    Args:
        dataset: Dataset name.
        output_dir: Base output directory.
    
    Returns:
        Dictionary of computed metrics.
    """
    dataset_output_dir = f"{output_dir}/{dataset}"
    output_path = f"{dataset_output_dir}/output.json"
    
    outputs = load_outputs(dataset_output_dir, output_path)
    overall_metric = initialize_accumulated_metrics()
    
    for item in outputs:
        # Get ground truth and prediction
        answer = get_ground_truth_answer(item, dataset)
        response = item["output"]
        pred = item["generated_answer"]
        
        # Count tokens
        message_for_tokens = replace_search_result(response)
        tokens = TOKENIZER.encode(message_for_tokens)
        generated_tokens = len(tokens)
        
        # Compute metrics
        metric = compute_metrics(pred, answer, dataset)
        num_search_queries, num_search_cycles = find_search_calls(item)
        
        # Update overall metrics
        if len(response) > 0:
            overall_metric["num_valid_answer"] += 1
        
        overall_metric["f1"].append(metric["f1"])
        overall_metric["search_calls"].append(num_search_queries)
        overall_metric["search_cycles"].append(num_search_cycles)
        overall_metric["generated_tokens"].append(generated_tokens)
        
        # Update accumulated metrics
        update_accumulated_f1_turns(overall_metric, num_search_cycles, metric["f1"])
        update_accumulated_f1_calls(overall_metric, num_search_queries, metric["f1"])
        update_accumulated_f1_tokens(overall_metric, generated_tokens, metric["f1"])
    
    # Finalize metrics
    finalize_metrics(overall_metric, len(outputs))
    
    return overall_metric


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate model outputs with optional backoff."
    )
    parser.add_argument(
        '--output_dir', '-o', 
        type=str, 
        default="results", 
        help='Directory of the model output JSON file.'
    )
    parser.add_argument(
        '--datasets', 
        nargs='+', 
        help='Dataset names'
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        print(f"\n{'='*50}")
        print(f"Evaluating: {dataset}")
        print('='*50)
        
        overall_metric = evaluate_dataset(dataset, args.output_dir)
        
        # Save metrics
        output_metrics_path = f"{args.output_dir}/{dataset}/output.json.metrics.json"
        with open(output_metrics_path, "w", encoding="utf-8") as f:
            json.dump(overall_metric, f, indent=4)
        
        # Print key metrics
        print("\nEvaluation Results:")
        for k in ["f1", "auc_at", "auc_ac", "auc_al", "generated_tokens_avg"]:
            if k in overall_metric:
                print(f"  {k}: {overall_metric[k]:.2f}")


if __name__ == "__main__":
    main()
