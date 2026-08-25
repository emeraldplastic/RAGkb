"""
Hybrid Search Service for RAG KB.
Combines vector search with keyword search for improved retrieval accuracy.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result with combined scores."""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    vector_score: float
    keyword_score: float
    combined_score: float
    rank: int

@dataclass
class SearchConfig:
    """Configuration for hybrid search."""
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    top_k: int = 10
    min_score_threshold: float = 0.3
    rerank: bool = True
    rerank_top_n: int = 20

class HybridSearchService:
    """Service for hybrid vector + keyword search."""
    
    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self.search_history: List[Dict[str, Any]] = []
    
    def hybrid_search(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        config: Optional[SearchConfig] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and keyword results.
        
        Args:
            query: Search query
            vector_results: Results from vector similarity search
            keyword_results: Results from keyword search
            config: Optional search configuration
        
        Returns:
            List of SearchResult objects ranked by combined score
        """
        search_config = config or self.config
        
        # Normalize scores
        normalized_vector = self._normalize_scores(
            [r.get('score', 0) for r in vector_results]
        )
        normalized_keyword = self._normalize_scores(
            [r.get('score', 0) for r in keyword_results]
        )
        
        # Combine results
        combined_results = []
        seen_doc_ids = set()
        
        # Process vector results
        for i, result in enumerate(vector_results):
            doc_id = result.get('document_id', result.get('id', str(i)))
            if doc_id in seen_doc_ids:
                continue
            
            seen_doc_ids.add(doc_id)
            
            combined_score = (
                normalized_vector[i] * search_config.vector_weight
            )
            
            combined_results.append(SearchResult(
                document_id=doc_id,
                content=result.get('content', ''),
                metadata=result.get('metadata', {}),
                vector_score=normalized_vector[i],
                keyword_score=0.0,
                combined_score=combined_score,
                rank=0
            ))
        
        # Process keyword results and merge
        for i, result in enumerate(keyword_results):
            doc_id = result.get('document_id', result.get('id', str(i)))
            
            if doc_id in seen_doc_ids:
                # Update existing result with keyword score
                for existing in combined_results:
                    if existing.document_id == doc_id:
                        existing.keyword_score = normalized_keyword[i]
                        existing.combined_score = (
                            existing.vector_score * search_config.vector_weight +
                            existing.keyword_score * search_config.keyword_weight
                        )
                        break
            else:
                seen_doc_ids.add(doc_id)
                
                combined_score = (
                    normalized_keyword[i] * search_config.keyword_weight
                )
                
                combined_results.append(SearchResult(
                    document_id=doc_id,
                    content=result.get('content', ''),
                    metadata=result.get('metadata', {}),
                    vector_score=0.0,
                    keyword_score=normalized_keyword[i],
                    combined_score=combined_score,
                    rank=0
                ))
        
        # Filter by threshold
        filtered_results = [
            r for r in combined_results
            if r.combined_score >= search_config.min_score_threshold
        ]
        
        # Sort by combined score
        filtered_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(filtered_results):
            result.rank = i + 1
        
        # Rerank if enabled
        if search_config.rerank and len(filtered_results) > 1:
            filtered_results = self._rerank_results(
                query, filtered_results[:search_config.rerank_top_n]
            )
        
        # Limit to top_k
        final_results = filtered_results[:search_config.top_k]
        
        # Log search
        self._log_search(query, len(final_results))
        
        return final_results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]
    
    def _rerank_results(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Rerank results based on query relevance.
        
        Args:
            query: Original search query
            results: Search results to rerank
        
        Returns:
            Reranked list of SearchResult objects
        """
        query_terms = self._extract_terms(query)
        
        for result in results:
            content_lower = result.content.lower()
            
            # Calculate term overlap
            term_matches = sum(
                1 for term in query_terms
                if term.lower() in content_lower
            )
            
            # Boost score based on term matches
            term_boost = term_matches / len(query_terms) if query_terms else 0
            
            # Apply boost
            result.combined_score = result.combined_score * (1 + term_boost * 0.2)
        
        # Re-sort
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Re-assign ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def _extract_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from query."""
        # Remove special characters and split
        cleaned = re.sub(r'[^\w\s]', ' ', query)
        terms = cleaned.split()
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        meaningful_terms = [
            term for term in terms
            if term.lower() not in stop_words and len(term) > 2
        ]
        
        return meaningful_terms
    
    def _log_search(self, query: str, result_count: int):
        """Log search for analytics."""
        self.search_history.append({
            'query': query,
            'result_count': result_count,
            'timestamp': datetime.now(),
            'config': {
                'vector_weight': self.config.vector_weight,
                'keyword_weight': self.config.keyword_weight,
                'top_k': self.config.top_k
            }
        })
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics."""
        if not self.search_history:
            return {}
        
        total_searches = len(self.search_history)
        avg_results = sum(
            h['result_count'] for h in self.search_history
        ) / total_searches
        
        # Most common queries
        query_counts = {}
        for history in self.search_history:
            query = history['query']
            query_counts[query] = query_counts.get(query, 0) + 1
        
        top_queries = sorted(
            query_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'total_searches': total_searches,
            'average_results_per_search': avg_results,
            'top_queries': top_queries,
            'config_used': {
                'vector_weight': self.config.vector_weight,
                'keyword_weight': self.config.keyword_weight
            }
        }
    
    def optimize_weights(
        self,
        relevance_feedback: List[Dict[str, Any]]
    ) -> SearchConfig:
        """
        Optimize search weights based on relevance feedback.
        
        Args:
            relevance_feedback: List of feedback with query, results, and relevance scores
        
        Returns:
            Optimized SearchConfig
        """
        if not relevance_feedback:
            return self.config
        
        # Simple optimization: increase weight for better performing method
        vector_performance = 0
        keyword_performance = 0
        
        for feedback in relevance_feedback:
            results = feedback.get('results', [])
            relevance = feedback.get('relevance_scores', {})
            
            for result in results:
                doc_id = result.get('document_id')
                rel_score = relevance.get(doc_id, 0)
                
                if result.get('vector_score', 0) > result.get('keyword_score', 0):
                    vector_performance += rel_score
                else:
                    keyword_performance += rel_score
        
        total = vector_performance + keyword_performance
        if total > 0:
            new_vector_weight = vector_performance / total
            new_keyword_weight = 1 - new_vector_weight
            
            # Smooth transition
            self.config.vector_weight = (
                self.config.vector_weight * 0.7 + new_vector_weight * 0.3
            )
            self.config.keyword_weight = 1 - self.config.vector_weight
        
        logger.info(f"Optimized weights: vector={self.config.vector_weight:.2f}, keyword={self.config.keyword_weight:.2f}")
        
        return self.config

# Global hybrid search service instance
hybrid_search_service = HybridSearchService()

def test_hybrid_search():
    """Test the hybrid search service."""
    service = HybridSearchService()
    
    # Mock vector results
    vector_results = [
        {'document_id': 'doc1', 'content': 'Machine learning algorithms', 'score': 0.9},
        {'document_id': 'doc2', 'content': 'Deep learning models', 'score': 0.8},
        {'document_id': 'doc3', 'content': 'Neural networks', 'score': 0.7}
    ]
    
    # Mock keyword results
    keyword_results = [
        {'document_id': 'doc2', 'content': 'Deep learning models', 'score': 0.95},
        {'document_id': 'doc4', 'content': 'AI applications', 'score': 0.85}
    ]
    
    # Perform hybrid search
    results = service.hybrid_search(
        query="machine learning",
        vector_results=vector_results,
        keyword_results=keyword_results
    )
    
    print(f"Hybrid Search Results:")
    for result in results:
        print(f"{result.rank}. {result.document_id}: {result.combined_score:.3f} (vector: {result.vector_score:.3f}, keyword: {result.keyword_score:.3f})")
    
    # Get statistics
    stats = service.get_search_statistics()
    print(f"\nSearch Statistics: {stats}")

if __name__ == "__main__":
    test_hybrid_search()
