"""
Query Expansion Service for RAG KB.
Improves search recall by expanding queries with synonyms and related terms.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)

@dataclass
class ExpandedQuery:
    """Represents an expanded query with original and additional terms."""
    original_query: str
    expanded_terms: List[str]
    synonyms: Dict[str, List[str]]
    related_terms: List[str]
    final_query: str

@dataclass
class ExpansionConfig:
    """Configuration for query expansion."""
    enable_synonyms: bool = True
    enable_related_terms: bool = True
    max_expansions: int = 5
    min_confidence: float = 0.6
    domain_specific: bool = True

class QueryExpansionService:
    """Service for intelligent query expansion."""
    
    def __init__(self, config: Optional[ExpansionConfig] = None):
        self.config = config or ExpansionConfig()
        self.expansion_history: List[Dict[str, Any]] = []
        
        # Domain-specific synonym mappings
        self.domain_synonyms = {
            # Technology/Software
            'ai': ['artificial intelligence', 'machine learning', 'ml', 'deep learning'],
            'ml': ['machine learning', 'artificial intelligence', 'ai'],
            'python': ['python programming', 'python language', 'py'],
            'javascript': ['js', 'ecmascript', 'node.js'],
            'database': ['db', 'data store', 'data storage'],
            'api': ['application programming interface', 'web service', 'endpoint'],
            'cloud': ['cloud computing', 'cloud services', 'saas'],
            'devops': ['development operations', 'ci/cd', 'deployment'],
            
            # Business/General
            'strategy': ['strategic planning', 'business strategy', 'tactics'],
            'management': ['administration', 'leadership', 'oversight'],
            'analysis': ['analytics', 'data analysis', 'examination'],
            'report': ['document', 'paper', 'study', 'brief'],
            'process': ['workflow', 'procedure', 'methodology'],
            'system': ['platform', 'framework', 'architecture'],
            'solution': ['answer', 'resolution', 'approach'],
            'implementation': ['deployment', 'execution', 'rollout'],
            
            # Data/Analytics
            'analytics': ['data analysis', 'metrics', 'insights', 'statistics'],
            'visualization': ['charts', 'graphs', 'data viz', 'dashboard'],
            'model': ['algorithm', 'predictive model', 'ml model'],
            'training': ['learning', 'education', 'development'],
            'performance': ['efficiency', 'speed', 'optimization'],
        }
        
        # Related term mappings (semantic associations)
        self.related_terms_map = {
            'ai': ['neural networks', 'nlp', 'computer vision', 'robotics'],
            'machine learning': ['supervised learning', 'unsupervised learning', 'reinforcement learning'],
            'python': ['django', 'flask', 'pandas', 'numpy'],
            'database': ['sql', 'nosql', 'postgresql', 'mongodb'],
            'cloud': ['aws', 'azure', 'gcp', 'serverless'],
            'security': ['encryption', 'authentication', 'authorization', 'cybersecurity'],
            'data': ['big data', 'data science', 'data engineering', 'etl'],
            'web': ['frontend', 'backend', 'fullstack', 'http'],
        }
    
    def expand_query(
        self,
        query: str,
        config: Optional[ExpansionConfig] = None
    ) -> ExpandedQuery:
        """
        Expand a query with synonyms and related terms.
        
        Args:
            query: Original search query
            config: Optional expansion configuration
        
        Returns:
            ExpandedQuery with expansion details
        """
        expansion_config = config or self.config
        
        # Extract terms from query
        terms = self._extract_terms(query)
        
        # Find synonyms for each term
        synonyms = {}
        expanded_terms = []
        
        if expansion_config.enable_synonyms:
            for term in terms:
                term_synonyms = self._find_synonyms(term, expansion_config.domain_specific)
                if term_synonyms:
                    synonyms[term] = term_synonyms
                    expanded_terms.extend(term_synonyms)
        
        # Find related terms
        related = []
        if expansion_config.enable_related_terms:
            for term in terms:
                term_related = self._find_related_terms(term)
                if term_related:
                    related.extend(term_related)
        
        # Limit expansions
        if expansion_config.max_expansions:
            expanded_terms = expanded_terms[:expansion_config.max_expansions]
            related = related[:expansion_config.max_expansions]
        
        # Build final expanded query
        final_terms = terms + expanded_terms + related
        final_query = ' '.join(final_terms)
        
        expanded_query = ExpandedQuery(
            original_query=query,
            expanded_terms=expanded_terms,
            synonyms=synonyms,
            related_terms=related,
            final_query=final_query
        )
        
        # Log expansion
        self._log_expansion(query, expanded_query)
        
        return expanded_query
    
    def _extract_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from query."""
        # Remove special characters and split
        cleaned = re.sub(r'[^\w\s]', ' ', query)
        terms = cleaned.split()
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must'
        }
        
        meaningful_terms = [
            term for term in terms
            if term.lower() not in stop_words and len(term) > 2
        ]
        
        return meaningful_terms
    
    def _find_synonyms(self, term: str, domain_specific: bool) -> List[str]:
        """Find synonyms for a term."""
        term_lower = term.lower()
        
        if domain_specific:
            # Check domain-specific synonyms
            if term_lower in self.domain_synonyms:
                return self.domain_synonyms[term_lower]
        
        # Check for partial matches
        partial_matches = []
        for key, synonyms in self.domain_synonyms.items():
            if term_lower in key or key in term_lower:
                partial_matches.extend(synonyms)
        
        return partial_matches[:3]  # Limit to top 3
    
    def _find_related_terms(self, term: str) -> List[str]:
        """Find semantically related terms."""
        term_lower = term.lower()
        
        if term_lower in self.related_terms_map:
            return self.related_terms_map[term_lower]
        
        # Check for partial matches
        partial_matches = []
        for key, related in self.related_terms_map.items():
            if term_lower in key or key in term_lower:
                partial_matches.extend(related)
        
        return partial_matches[:3]  # Limit to top 3
    
    def batch_expand(
        self,
        queries: List[str],
        config: Optional[ExpansionConfig] = None
    ) -> List[ExpandedQuery]:
        """
        Expand multiple queries in batch.
        
        Args:
            queries: List of queries to expand
            config: Optional expansion configuration
        
        Returns:
            List of ExpandedQuery objects
        """
        return [self.expand_query(query, config) for query in queries]
    
    def _log_expansion(self, original: str, expanded: ExpandedQuery):
        """Log query expansion for analytics."""
        self.expansion_history.append({
            'original_query': original,
            'expanded_query': expanded.final_query,
            'expansion_count': len(expanded.expanded_terms) + len(expanded.related_terms),
            'synonyms_used': len(expanded.synonyms),
            'timestamp': datetime.now()
        })
    
    def get_expansion_statistics(self) -> Dict[str, Any]:
        """Get query expansion statistics."""
        if not self.expansion_history:
            return {}
        
        total_expansions = len(self.expansion_history)
        avg_expansion_count = sum(
            h['expansion_count'] for h in self.expansion_history
        ) / total_expansions
        
        # Most expanded terms
        all_synonyms = {}
        for history in self.expansion_history:
            # This would need to be tracked differently in production
            pass
        
        return {
            'total_expansions': total_expansions,
            'average_expansion_count': avg_expansion_count,
            'config_used': {
                'enable_synonyms': self.config.enable_synonyms,
                'enable_related_terms': self.config.enable_related_terms,
                'max_expansions': self.config.max_expansions
            }
        }
    
    def add_domain_synonym(self, term: str, synonyms: List[str]):
        """Add custom domain-specific synonyms."""
        term_lower = term.lower()
        if term_lower not in self.domain_synonyms:
            self.domain_synonyms[term_lower] = []
        
        self.domain_synonyms[term_lower].extend(synonyms)
        logger.info(f"Added synonyms for '{term}': {synonyms}")
    
    def add_related_terms(self, term: str, related: List[str]):
        """Add custom related terms."""
        term_lower = term.lower()
        if term_lower not in self.related_terms_map:
            self.related_terms_map[term_lower] = []
        
        self.related_terms_map[term_lower].extend(related)
        logger.info(f"Added related terms for '{term}': {related}")

# Global query expansion service instance
query_expansion_service = QueryExpansionService()

def test_query_expansion():
    """Test the query expansion service."""
    service = QueryExpansionService()
    
    # Test single query expansion
    query = "machine learning python"
    expanded = service.expand_query(query)
    
    print(f"Original Query: {expanded.original_query}")
    print(f"Expanded Query: {expanded.final_query}")
    print(f"Synonyms: {expanded.synonyms}")
    print(f"Related Terms: {expanded.related_terms}")
    print()
    
    # Test batch expansion
    queries = ["ai database", "cloud security", "python web"]
    batch_results = service.batch_expand(queries)
    
    print("Batch Expansion Results:")
    for result in batch_results:
        print(f"  '{result.original_query}' -> '{result.final_query}'")
    
    # Get statistics
    stats = service.get_expansion_statistics()
    print(f"\nExpansion Statistics: {stats}")

if __name__ == "__main__":
    test_query_expansion()
