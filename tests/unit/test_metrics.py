"""
Unit tests for evaluation metrics.
"""
import pytest
from app.evaluation.metrics import (
    compute_text_similarity,
    compute_retrieval_metrics,
    aggregate_metrics,
    compute_answer_quality,
    compute_latency_metrics
)


class TestTextSimilarity:
    """Tests for text similarity computation."""
    
    def test_identical_texts(self):
        """Test similarity of identical texts."""
        text = "This is a test sentence"
        score = compute_text_similarity(text, text)
        assert score == 1.0
    
    def test_completely_different_texts(self):
        """Test similarity of completely different texts."""
        text1 = "apple banana cherry"
        text2 = "dog elephant frog"
        score = compute_text_similarity(text1, text2)
        assert score == 0.0
    
    def test_partial_overlap(self):
        """Test similarity with partial word overlap."""
        text1 = "the quick brown fox"
        text2 = "the quick red dog"
        score = compute_text_similarity(text1, text2)
        assert 0 < score < 1
        assert score > 0.3  # Some overlap
    
    def test_empty_texts(self):
        """Test handling of empty texts."""
        assert compute_text_similarity("", "") == 0.0
        assert compute_text_similarity("hello", "") == 0.0
        assert compute_text_similarity("", "hello") == 0.0


class TestRetrievalMetrics:
    """Tests for retrieval metrics computation."""
    
    def test_perfect_retrieval(self):
        """Test metrics when all retrieved docs are relevant."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = [1, 2, 3, 4, 5]
        
        metrics = compute_retrieval_metrics(retrieved, relevant, k=5)
        
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.mrr == 1.0
    
    def test_no_relevant_retrieved(self):
        """Test metrics when no relevant docs are retrieved."""
        retrieved = [6, 7, 8, 9, 10]
        relevant = [1, 2, 3, 4, 5]
        
        metrics = compute_retrieval_metrics(retrieved, relevant, k=5)
        
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.mrr == 0.0
    
    def test_partial_retrieval(self):
        """Test metrics with partial retrieval."""
        retrieved = [1, 2, 6, 7, 8]  # 2 out of 5 relevant
        relevant = [1, 2, 3, 4, 5]
        
        metrics = compute_retrieval_metrics(retrieved, relevant, k=5)
        
        assert metrics.precision == 0.4  # 2/5
        assert metrics.recall == 0.4  # 2/5
        assert metrics.mrr == 1.0  # First result is relevant
    
    def test_mrr_calculation(self):
        """Test Mean Reciprocal Rank calculation."""
        retrieved = [6, 7, 1, 8, 9]  # First relevant at position 3
        relevant = [1, 2, 3]
        
        metrics = compute_retrieval_metrics(retrieved, relevant, k=5)
        
        assert metrics.mrr == pytest.approx(1/3, rel=0.01)


class TestAnswerQuality:
    """Tests for answer quality computation."""
    
    def test_perfect_answer(self):
        """Test quality metrics for perfect answer."""
        expected = "The coverage limit is 50000 rupees"
        actual = "The coverage limit is 50000 rupees"
        
        result = compute_answer_quality(expected, actual)
        
        assert result["similarity"] == 1.0
        assert result["overall_score"] >= 0.9
    
    def test_answer_with_key_terms(self):
        """Test key term coverage."""
        expected = "Maternity coverage is 50000"
        actual = "The maternity benefit provides coverage of 50000"
        key_terms = ["maternity", "50000"]
        
        result = compute_answer_quality(expected, actual, key_terms)
        
        assert result["key_term_coverage"] == 1.0
    
    def test_missing_key_terms(self):
        """Test detection of missing key terms."""
        expected = "Maternity coverage is 50000"
        actual = "Some coverage is provided"
        key_terms = ["maternity", "50000"]
        
        result = compute_answer_quality(expected, actual, key_terms)
        
        assert result["key_term_coverage"] == 0.0


class TestLatencyMetrics:
    """Tests for latency statistics."""
    
    def test_latency_statistics(self):
        """Test computation of latency statistics."""
        latencies = [100, 200, 300, 400, 500]
        
        result = compute_latency_metrics(latencies)
        
        assert result["min"] == 100
        assert result["max"] == 500
        assert result["mean"] == 300
        assert result["median"] == 300
    
    def test_empty_latencies(self):
        """Test handling of empty latency list."""
        result = compute_latency_metrics([])
        
        assert result["min"] == 0
        assert result["max"] == 0
        assert result["mean"] == 0
    
    def test_percentiles(self):
        """Test percentile calculation."""
        latencies = list(range(1, 101))  # 1-100
        
        result = compute_latency_metrics(latencies)
        
        assert result["p95"] >= 95
        assert result["p99"] >= 99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
