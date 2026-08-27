"""
Semantic Search Service for RAG KB.
Advanced semantic search using embedding-based similarity matching.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SemanticSearchResult:
    """Represents a semantic search result."""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    rank: int

@dataclass
class SearchConfig:
    """Configuration for semantic search."""
    top_k: int = 10
    min_similarity: float = 0.5
    rerank: bool = True
    include_metadata: bool = True

class SemanticSearchService:
    """Service for semantic search using embeddings."""
    
    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self.document_embeddings: Dict[str, np.ndarray] = {}
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.search_history: List[Dict[str, Any]] = []
    
    def add_document(
        self,
        document_id: str,
        content: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a document to the semantic search index.
        
        Args:
            document_id: Unique document identifier
            content: Document text content
            embedding: Pre-computed embedding (optional)
            metadata: Additional document metadata
        """
        if embedding is None:
            # In production, this would use an embedding model
            # For now, create a placeholder
            embedding = self._generate_embedding(content)
        
        self.document_embeddings[document_id] = embedding
        self.documents[document_id] = {
            'content': content,
            'metadata': metadata or {},
            'added_at': datetime.now()
        }
        
        logger.debug(f"Added document {document_id} to semantic index")
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text (placeholder).
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        # In production, this would use a model like sentence-transformers
        # For now, create a random embedding for demonstration
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(768)  # Typical embedding dimension
    
    def search(
        self,
        query: str,
        config: Optional[SearchConfig] = None
    ) -> List[SemanticSearchResult]:
        """
        Perform semantic search.
        
        Args:
            query: Search query
            config: Optional search configuration
        
        Returns:
            List of SemanticSearchResult objects
        """
        search_config = config or self.config
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Calculate similarities
        results = []
        
        for doc_id, doc_embedding in self.document_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            if similarity >= search_config.min_similarity:
                doc_data = self.documents[doc_id]
                
                result = SemanticSearchResult(
                    document_id=doc_id,
                    content=doc_data['content'],
                    metadata=doc_data['metadata'] if search_config.include_metadata else {},
                    similarity_score=similarity,
                    rank=0
                )
                
                results.append(result)
        
        # Sort by similarity
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Limit to top_k
        results = results[:search_config.top_k]
        
        # Assign ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        # Log search
        self._log_search(query, len(results))
        
        return results
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def search_with_filter(
        self,
        query: str,
        metadata_filter: Dict[str, Any],
        config: Optional[SearchConfig] = None
    ) -> List[SemanticSearchResult]:
        """
        Perform semantic search with metadata filtering.
        
        Args:
            query: Search query
            metadata_filter: Filter criteria for metadata
            config: Optional search configuration
        
        Returns:
            List of SemanticSearchResult objects
        """
        # First perform semantic search
        results = self.search(query, config)
        
        # Filter by metadata
        filtered_results = []
        
        for result in results:
            if self._matches_filter(result.metadata, metadata_filter):
                filtered_results.append(result)
        
        # Re-rank after filtering
        for i, result in enumerate(filtered_results):
            result.rank = i + 1
        
        return filtered_results
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_criteria.items():
            if key not in metadata:
                return False
            
            if metadata[key] != value:
                return False
        
        return True
    
    def batch_search(
        self,
        queries: List[str],
        config: Optional[SearchConfig] = None
    ) -> List[List[SemanticSearchResult]]:
        """
        Perform batch semantic search.
        
        Args:
            queries: List of search queries
            config: Optional search configuration
        
        Returns:
            List of search result lists
        """
        return [self.search(query, config) for query in queries]
    
    def find_similar_documents(
        self,
        document_id: str,
        top_k: int = 5
    ) -> List[SemanticSearchResult]:
        """
        Find documents similar to a given document.
        
        Args:
            document_id: ID of the reference document
            top_k: Number of similar documents to return
        
        Returns:
            List of similar SemanticSearchResult objects
        """
        if document_id not in self.document_embeddings:
            logger.warning(f"Document {document_id} not found")
            return []
        
        # Get reference embedding
        ref_embedding = self.document_embeddings[document_id]
        
        # Calculate similarities with all other documents
        results = []
        
        for doc_id, doc_embedding in self.document_embeddings.items():
            if doc_id == document_id:
                continue
            
            similarity = self._cosine_similarity(ref_embedding, doc_embedding)
            
            doc_data = self.documents[doc_id]
            
            result = SemanticSearchResult(
                document_id=doc_id,
                content=doc_data['content'],
                metadata=doc_data['metadata'],
                similarity_score=similarity,
                rank=0
            )
            
            results.append(result)
        
        # Sort and limit
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        results = results[:top_k]
        
        # Assign ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def _log_search(self, query: str, result_count: int):
        """Log search for analytics."""
        self.search_history.append({
            'query': query,
            'result_count': result_count,
            'timestamp': datetime.now(),
            'config': {
                'top_k': self.config.top_k,
                'min_similarity': self.config.min_similarity
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
            'indexed_documents': len(self.documents),
            'top_queries': top_queries
        }
    
    def remove_document(self, document_id: str) -> bool:
        """
        Remove a document from the index.
        
        Args:
            document_id: ID of document to remove
        
        Returns:
            True if removed, False if not found
        """
        if document_id in self.document_embeddings:
            del self.document_embeddings[document_id]
            del self.documents[document_id]
            logger.info(f"Removed document {document_id} from semantic index")
            return True
        
        return False
    
    def clear_index(self):
        """Clear all documents from the index."""
        self.document_embeddings.clear()
        self.documents.clear()
        logger.info("Cleared semantic search index")

# Global semantic search service instance
semantic_search_service = SemanticSearchService()

def test_semantic_search():
    """Test the semantic search service."""
    service = SemanticSearchService()
    
    # Add some documents
    service.add_document(
        document_id="doc1",
        content="Machine learning is a subset of artificial intelligence",
        metadata={"category": "AI", "author": "John"}
    )
    
    service.add_document(
        document_id="doc2",
        content="Deep learning uses neural networks for complex tasks",
        metadata={"category": "AI", "author": "Jane"}
    )
    
    service.add_document(
        document_id="doc3",
        content="Python is a popular programming language for data science",
        metadata={"category": "Programming", "author": "Bob"}
    )
    
    # Perform search
    results = service.search("artificial intelligence")
    
    print(f"Search results for 'artificial intelligence':")
    for result in results:
        print(f"{result.rank}. {result.document_id}: {result.similarity_score:.3f}")
        print(f"   {result.content[:50]}...")
    
    # Find similar documents
    similar = service.find_similar_documents("doc1")
    print(f"\nDocuments similar to doc1:")
    for result in similar:
        print(f"{result.rank}. {result.document_id}: {result.similarity_score:.3f}")
    
    # Get statistics
    stats = service.get_search_statistics()
    print(f"\nSearch statistics: {stats}")

if __name__ == "__main__":
    test_semantic_search()
