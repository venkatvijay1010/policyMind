"""
Evaluation metrics for PolicyMind.
"""
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    mrr: float  # Mean Reciprocal Rank
    hit_rate: float


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute similarity between two texts using word overlap.
    Returns a score between 0 and 1.
    """
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    jaccard = len(intersection) / len(union)
    recall = len(intersection) / len(words1)  # How much of expected is covered
    
    return 0.5 * jaccard + 0.5 * recall


def compute_retrieval_metrics(
    retrieved_ids: List[int],
    relevant_ids: List[int],
    k: int = 10
) -> EvaluationMetrics:
    """
    Compute retrieval metrics for a single query.
    
    Args:
        retrieved_ids: List of retrieved document IDs (ranked)
        relevant_ids: List of relevant document IDs (ground truth)
        k: Number of top results to consider
        
    Returns:
        EvaluationMetrics with precision, recall, F1, MRR, hit rate
    """
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    # True positives
    tp = len(retrieved_set & relevant_set)
    
    # Precision@k
    precision = tp / k if k > 0 else 0.0
    
    # Recall@k
    recall = tp / len(relevant_set) if relevant_set else 1.0
    
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Accuracy (binary: did we retrieve at least one relevant doc?)
    accuracy = 1.0 if tp > 0 else 0.0
    
    # Mean Reciprocal Rank
    mrr = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in relevant_set:
            mrr = 1.0 / i
            break
    
    # Hit Rate (same as accuracy for single query)
    hit_rate = accuracy
    
    return EvaluationMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        accuracy=accuracy,
        mrr=round(mrr, 4),
        hit_rate=hit_rate
    )


def aggregate_metrics(metrics_list: List[EvaluationMetrics]) -> EvaluationMetrics:
    """
    Aggregate metrics across multiple queries.
    """
    if not metrics_list:
        return EvaluationMetrics(0, 0, 0, 0, 0, 0)
    
    n = len(metrics_list)
    
    return EvaluationMetrics(
        precision=round(sum(m.precision for m in metrics_list) / n, 4),
        recall=round(sum(m.recall for m in metrics_list) / n, 4),
        f1_score=round(sum(m.f1_score for m in metrics_list) / n, 4),
        accuracy=round(sum(m.accuracy for m in metrics_list) / n, 4),
        mrr=round(sum(m.mrr for m in metrics_list) / n, 4),
        hit_rate=round(sum(m.hit_rate for m in metrics_list) / n, 4)
    )


def compute_answer_quality(
    expected_answer: str,
    actual_answer: str,
    key_terms: Optional[List[str]] = None
) -> dict:
    """
    Compute quality metrics for generated answers.
    
    Args:
        expected_answer: Ground truth answer
        actual_answer: Generated answer
        key_terms: Important terms that should appear in answer
        
    Returns:
        Dict with similarity, key_term_coverage, length_ratio
    """
    similarity = compute_text_similarity(expected_answer, actual_answer)
    
    # Key term coverage
    key_term_coverage = 1.0
    if key_terms:
        actual_lower = actual_answer.lower()
        found = sum(1 for term in key_terms if term.lower() in actual_lower)
        key_term_coverage = found / len(key_terms)
    
    # Length ratio (penalize too short or too long)
    expected_len = len(expected_answer.split())
    actual_len = len(actual_answer.split())
    
    if expected_len > 0:
        ratio = actual_len / expected_len
        # Ideal ratio is 1.0, penalize deviations
        length_score = max(0, 1 - abs(1 - ratio) * 0.5)
    else:
        length_score = 1.0 if actual_len == 0 else 0.5
    
    return {
        "similarity": round(similarity, 4),
        "key_term_coverage": round(key_term_coverage, 4),
        "length_score": round(length_score, 4),
        "overall_score": round((similarity * 0.5 + key_term_coverage * 0.3 + length_score * 0.2), 4)
    }


def compute_latency_metrics(latencies_ms: List[int]) -> dict:
    """
    Compute latency statistics.
    """
    if not latencies_ms:
        return {
            "min": 0, "max": 0, "mean": 0, "median": 0, "p95": 0, "p99": 0
        }
    
    arr = np.array(latencies_ms)
    
    return {
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "mean": round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2)
    }
