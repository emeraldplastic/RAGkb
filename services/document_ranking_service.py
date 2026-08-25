"""
Document Ranking Service for RAG KB.
Advanced document ranking with relevance scoring and result diversification.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

@dataclass
class RankedDocument:
    """Represents a ranked document with detailed scores."""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    diversity_score: float
    freshness_score: float
    authority_score: float
    final_score: float
    rank: int

@dataclass
class RankingConfig:
    """Configuration for document ranking."""
    relevance_weight: float = 0.5
    diversity_weight: float = 0.2
    freshness_weight: float = 0.15
    authority_weight: float = 0.15
    enable_diversification: bool = True
    max_similar_results: int = 3
    freshness_decay_days: int = 365

class DocumentRankingService:
    """Service for advanced document ranking and diversification."""
    
    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()
        self.ranking_history: List[Dict[str, Any]] = []
    
    def rank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        config: Optional[RankingConfig] = None
    ) -> List[RankedDocument]:
        """
        Rank documents based on multiple scoring factors.
        
        Args:
            query: Search query
            documents: List of documents to rank
            config: Optional ranking configuration
        
        Returns:
            List of RankedDocument objects sorted by final score
        """
        ranking_config = config or self.config
        
        ranked_docs = []
        
        for doc in documents:
            doc_id = doc.get('document_id', doc.get('id', ''))
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            # Calculate individual scores
            relevance_score = self._calculate_relevance_score(query, content, metadata)
            diversity_score = 0.0  # Will be calculated after initial ranking
            freshness_score = self._calculate_freshness_score(metadata)
            authority_score = self._calculate_authority_score(metadata)
            
            # Initial final score (without diversity)
            initial_score = (
                relevance_score * ranking_config.relevance_weight +
                freshness_score * ranking_config.freshness_weight +
                authority_score * ranking_config.authority_weight
            )
            
            ranked_doc = RankedDocument(
                document_id=doc_id,
                content=content,
                metadata=metadata,
                relevance_score=relevance_score,
                diversity_score=diversity_score,
                freshness_score=freshness_score,
                authority_score=authority_score,
                final_score=initial_score,
                rank=0
            )
            
            ranked_docs.append(ranked_doc)
        
        # Sort by initial score
        ranked_docs.sort(key=lambda x: x.final_score, reverse=True)
        
        # Apply diversification if enabled
        if ranking_config.enable_diversification:
            ranked_docs = self._apply_diversification(ranked_docs, ranking_config)
        
        # Recalculate final scores with diversity
        for doc in ranked_docs:
            doc.final_score = (
                doc.relevance_score * ranking_config.relevance_weight +
                doc.diversity_score * ranking_config.diversity_weight +
                doc.freshness_score * ranking_config.freshness_weight +
                doc.authority_score * ranking_config.authority_weight
            )
        
        # Final sort
        ranked_docs.sort(key=lambda x: x.final_score, reverse=True)
        
        # Assign ranks
        for i, doc in enumerate(ranked_docs):
            doc.rank = i + 1
        
        # Log ranking
        self._log_ranking(query, len(ranked_docs))
        
        return ranked_docs
    
    def _calculate_relevance_score(
        self,
        query: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> float:
        """Calculate relevance score based on query-content match."""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Exact phrase match
        if query_lower in content_lower:
            base_score = 1.0
        else:
            # Term overlap
            query_terms = set(query_lower.split())
            content_terms = set(content_lower.split())
            
            if not query_terms:
                return 0.0
            
            overlap = len(query_terms & content_terms)
            base_score = overlap / len(query_terms)
        
        # Boost for title matches
        title = metadata.get('title', '').lower()
        if query_lower in title:
            base_score = min(1.0, base_score + 0.2)
        
        # Boost for metadata keywords
        keywords = metadata.get('keywords', [])
        if keywords:
            keyword_matches = sum(
                1 for kw in keywords
                if query_lower in kw.lower()
            )
            if keyword_matches > 0:
                base_score = min(1.0, base_score + 0.1 * keyword_matches)
        
        return base_score
    
    def _calculate_freshness_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate freshness score based on document age."""
        created_date = metadata.get('created_date')
        updated_date = metadata.get('updated_date')
        
        # Use updated date if available, otherwise created date
        doc_date = updated_date or created_date
        
        if not doc_date:
            return 0.5  # Neutral score if no date
        
        if isinstance(doc_date, str):
            try:
                doc_date = datetime.fromisoformat(doc_date.replace('Z', '+00:00'))
            except:
                return 0.5
        
        now = datetime.now()
        age_days = (now - doc_date).days
        
        # Exponential decay
        decay_constant = self.config.freshness_decay_days
        freshness_score = math.exp(-age_days / decay_constant)
        
        return freshness_score
    
    def _calculate_authority_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate authority score based on document metadata."""
        score = 0.0
        
        # Author authority
        author = metadata.get('author', '')
        if author:
            # In production, this would check against an authority database
            score += 0.3
        
        # Source authority
        source = metadata.get('source', '')
        high_authority_sources = {
            'arxiv', 'nature', 'science', 'ieee', 'acm',
            'github', 'stackoverflow', 'medium'
        }
        
        if source:
            if any(trusted in source.lower() for trusted in high_authority_sources):
                score += 0.4
            else:
                score += 0.2
        
        # Citation count (if available)
        citations = metadata.get('citations', 0)
        if citations > 0:
            # Logarithmic scaling for citations
            score += min(0.3, math.log10(citations + 1) / 10)
        
        # View count / popularity
        views = metadata.get('views', 0)
        if views > 0:
            score += min(0.2, math.log10(views + 1) / 10)
        
        return min(1.0, score)
    
    def _apply_diversification(
        self,
        documents: List[RankedDocument],
        config: RankingConfig
    ) -> List[RankedDocument]:
        """
        Apply result diversification to avoid similar documents clustering.
        
        Args:
            documents: Pre-ranked documents
            config: Ranking configuration
        
        Returns:
            Diversified list of RankedDocument objects
        """
        if len(documents) <= 1:
            return documents
        
        diversified = []
        seen_clusters = set()
        similar_count = defaultdict(int)
        
        for doc in documents:
            # Calculate similarity to already selected documents
            max_similarity = 0.0
            
            for selected in diversified:
                similarity = self._calculate_document_similarity(doc, selected)
                max_similarity = max(max_similarity, similarity)
            
            # Check if we've hit the limit for similar documents
            if max_similarity > 0.7:  # High similarity threshold
                cluster_id = self._get_document_cluster(doc)
                
                if similar_count[cluster_id] >= config.max_similar_results:
                    # Skip this document, too similar to existing ones
                    doc.diversity_score = 0.0
                    continue
                
                similar_count[cluster_id] += 1
                doc.diversity_score = 0.3  # Moderate diversity score
            else:
                doc.diversity_score = 1.0  # High diversity score
            
            diversified.append(doc)
        
        return diversified
    
    def _calculate_document_similarity(
        self,
        doc1: RankedDocument,
        doc2: RankedDocument
    ) -> float:
        """Calculate similarity between two documents."""
        # Simple content overlap similarity
        content1 = set(doc1.content.lower().split())
        content2 = set(doc2.content.lower().split())
        
        if not content1 or not content2:
            return 0.0
        
        intersection = content1 & content2
        union = content1 | content2
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        # Also check metadata similarity
        source1 = doc1.metadata.get('source', '')
        source2 = doc2.metadata.get('source', '')
        
        if source1 and source2 and source1 == source2:
            jaccard_similarity = min(1.0, jaccard_similarity + 0.3)
        
        return jaccard_similarity
    
    def _get_document_cluster(self, doc: RankedDocument) -> str:
        """Get cluster identifier for a document."""
        # Use source as cluster identifier
        source = doc.metadata.get('source', 'unknown')
        
        # Alternatively, could use author or topic
        if not source or source == 'unknown':
            author = doc.metadata.get('author', 'unknown')
            return f"author:{author}"
        
        return f"source:{source}"
    
    def re_rank(
        self,
        query: str,
        documents: List[RankedDocument],
        feedback: Optional[Dict[str, float]] = None
    ) -> List[RankedDocument]:
        """
        Re-rank documents based on user feedback.
        
        Args:
            query: Original search query
            documents: Currently ranked documents
            feedback: Dictionary mapping document_id to relevance feedback (0-1)
        
        Returns:
            Re-ranked list of RankedDocument objects
        """
        if not feedback:
            return documents
        
        # Adjust weights based on feedback
        for doc in documents:
            doc_id = doc.document_id
            
            if doc_id in feedback:
                feedback_score = feedback[doc_id]
                
                # Boost documents with positive feedback
                if feedback_score > 0.7:
                    doc.relevance_score = min(1.0, doc.relevance_score * 1.2)
                # Penalize documents with negative feedback
                elif feedback_score < 0.3:
                    doc.relevance_score = max(0.0, doc.relevance_score * 0.8)
        
        # Recalculate final scores
        for doc in documents:
            doc.final_score = (
                doc.relevance_score * self.config.relevance_weight +
                doc.diversity_score * self.config.diversity_weight +
                doc.freshness_score * self.config.freshness_weight +
                doc.authority_score * self.config.authority_weight
            )
        
        # Re-sort
        documents.sort(key=lambda x: x.final_score, reverse=True)
        
        # Re-assign ranks
        for i, doc in enumerate(documents):
            doc.rank = i + 1
        
        return documents
    
    def _log_ranking(self, query: str, result_count: int):
        """Log ranking for analytics."""
        self.ranking_history.append({
            'query': query,
            'result_count': result_count,
            'timestamp': datetime.now(),
            'config': {
                'relevance_weight': self.config.relevance_weight,
                'diversity_weight': self.config.diversity_weight,
                'enable_diversification': self.config.enable_diversification
            }
        })
    
    def get_ranking_statistics(self) -> Dict[str, Any]:
        """Get ranking statistics."""
        if not self.ranking_history:
            return {}
        
        total_rankings = len(self.ranking_history)
        avg_results = sum(
            h['result_count'] for h in self.ranking_history
        ) / total_rankings
        
        return {
            'total_rankings': total_rankings,
            'average_results_per_ranking': avg_results,
            'config_used': {
                'relevance_weight': self.config.relevance_weight,
                'diversity_weight': self.config.diversity_weight,
                'freshness_weight': self.config.freshness_weight,
                'authority_weight': self.config.authority_weight
            }
        }

# Global document ranking service instance
document_ranking_service = DocumentRankingService()

def test_document_ranking():
    """Test the document ranking service."""
    service = DocumentRankingService()
    
    # Mock documents
    documents = [
        {
            'document_id': 'doc1',
            'content': 'Machine learning algorithms for data analysis',
            'metadata': {
                'title': 'ML Algorithms',
                'author': 'Dr. Smith',
                'source': 'arxiv',
                'created_date': '2024-01-15',
                'citations': 50
            }
        },
        {
            'document_id': 'doc2',
            'content': 'Deep learning neural networks in Python',
            'metadata': {
                'title': 'Deep Learning',
                'author': 'Dr. Johnson',
                'source': 'github',
                'created_date': '2023-06-20',
                'views': 1000
            }
        },
        {
            'document_id': 'doc3',
            'content': 'Data science with Python and SQL',
            'metadata': {
                'title': 'Data Science',
                'author': 'Prof. Williams',
                'source': 'medium',
                'created_date': '2024-03-10'
            }
        }
    ]
    
    # Rank documents
    ranked = service.rank_documents(
        query="machine learning python",
        documents=documents
    )
    
    print("Ranked Documents:")
    for doc in ranked:
        print(f"{doc.rank}. {doc.document_id}: {doc.final_score:.3f}")
        print(f"   Relevance: {doc.relevance_score:.3f}, Diversity: {doc.diversity_score:.3f}")
        print(f"   Freshness: {doc.freshness_score:.3f}, Authority: {doc.authority_score:.3f}")
    
    # Get statistics
    stats = service.get_ranking_statistics()
    print(f"\nRanking Statistics: {stats}")

if __name__ == "__main__":
    test_document_ranking()
