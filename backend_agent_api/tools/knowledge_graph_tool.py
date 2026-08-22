"""
Knowledge graph tool for RAG knowledge base.
Provides knowledge graph construction and analysis capabilities.
"""

from typing import Optional, Dict, Any, List, Set, Tuple
from pydantic import BaseModel, Field
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class Entity(BaseModel):
    """Represents an entity in the knowledge graph."""
    id: str
    text: str
    entity_type: str
    frequency: int = 1
    document_ids: List[str] = Field(default_factory=list)

class Relation(BaseModel):
    """Represents a relation between entities in the knowledge graph."""
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    document_ids: List[str] = Field(default_factory=list)

class KnowledgeGraphRequest(BaseModel):
    """Request model for knowledge graph operations."""
    document_ids: Optional[List[str]] = Field(default=None, description="Document IDs to include in graph")
    entity_types: Optional[List[str]] = Field(default=None, description="Entity types to extract")
    min_frequency: int = Field(default=2, description="Minimum entity frequency to include")
    max_nodes: int = Field(default=100, description="Maximum number of nodes in graph")

class KnowledgeGraphResponse(BaseModel):
    """Response model for knowledge graph operations."""
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

class KnowledgeGraphTool:
    """Tool for constructing and analyzing knowledge graphs from documents."""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.entity_patterns = {
            'person': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'organization': r'\b[A-Z][a-zA-Z&]+(?:\s+[A-Z][a-zA-Z&]+)*\b',
            'location': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            'date': r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b',
            'number': r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'https?://[^\s<>"{}|\\^`\[\]]+'
        }
    
    async def build_graph(
        self,
        documents: List[Dict[str, Any]],
        config: KnowledgeGraphRequest
    ) -> KnowledgeGraphResponse:
        """
        Build a knowledge graph from document contents.
        
        Args:
            documents: List of documents with id and content
            config: Configuration for graph construction
        
        Returns:
            KnowledgeGraphResponse with entities and relations
        """
        try:
            self.entities.clear()
            self.relations.clear()
            
            # Extract entities from each document
            for doc in documents:
                doc_id = doc.get('id', '')
                content = doc.get('content', '')
                
                if not content:
                    continue
                
                await self._extract_entities(content, doc_id, config)
            
            # Filter entities by frequency
            filtered_entities = {
                eid: entity for eid, entity in self.entities.items()
                if entity.frequency >= config.min_frequency
            }
            
            # Limit number of nodes
            if len(filtered_entities) > config.max_nodes:
                # Sort by frequency and keep top nodes
                sorted_entities = sorted(
                    filtered_entities.items(),
                    key=lambda x: x[1].frequency,
                    reverse=True
                )
                filtered_entities = dict(sorted_entities[:config.max_nodes])
            
            # Build relations between entities
            await self._build_relations(documents, filtered_entities, config)
            
            # Convert to response format
            entities_list = [
                {
                    'id': entity.id,
                    'text': entity.text,
                    'type': entity.entity_type,
                    'frequency': entity.frequency,
                    'document_count': len(entity.document_ids)
                }
                for eid, entity in filtered_entities.items()
            ]
            
            relations_list = [
                {
                    'source': rel.source_id,
                    'target': rel.target_id,
                    'type': rel.relation_type,
                    'weight': rel.weight,
                    'document_count': len(rel.document_ids)
                }
                for rel in self.relations
                if rel.source_id in filtered_entities and rel.target_id in filtered_entities
            ]
            
            # Calculate statistics
            statistics = {
                'total_entities': len(filtered_entities),
                'total_relations': len(relations_list),
                'entity_types': self._get_entity_type_counts(filtered_entities),
                'avg_frequency': sum(e.frequency for e in filtered_entities.values()) / len(filtered_entities) if filtered_entities else 0,
                'most_connected': self._get_most_connected_nodes(filtered_entities, relations_list)
            }
            
            return KnowledgeGraphResponse(
                entities=entities_list,
                relations=relations_list,
                statistics=statistics,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return KnowledgeGraphResponse(
                entities=[],
                relations=[],
                statistics={},
                success=False,
                error_message=str(e)
            )
    
    async def _extract_entities(
        self,
        content: str,
        doc_id: str,
        config: KnowledgeGraphRequest
    ):
        """Extract entities from document content."""
        entity_types_to_extract = config.entity_types or list(self.entity_patterns.keys())
        
        for entity_type in entity_types_to_extract:
            if entity_type not in self.entity_patterns:
                continue
            
            pattern = self.entity_patterns[entity_type]
            matches = re.finditer(pattern, content)
            
            for match in matches:
                entity_text = match.group()
                entity_id = f"{entity_type}:{entity_text.lower()}"
                
                if entity_id in self.entities:
                    self.entities[entity_id].frequency += 1
                    if doc_id not in self.entities[entity_id].document_ids:
                        self.entities[entity_id].document_ids.append(doc_id)
                else:
                    self.entities[entity_id] = Entity(
                        id=entity_id,
                        text=entity_text,
                        entity_type=entity_type,
                        frequency=1,
                        document_ids=[doc_id]
                    )
    
    async def _build_relations(
        self,
        documents: List[Dict[str, Any]],
        entities: Dict[str, Entity],
        config: KnowledgeGraphRequest
    ):
        """Build relations between entities based on co-occurrence."""
        entity_id_set = set(entities.keys())
        
        for doc in documents:
            doc_id = doc.get('id', '')
            content = doc.get('content', '')
            
            if not content:
                continue
            
            # Find entities in this document
            doc_entities = []
            for entity_id, entity in entities.items():
                if doc_id in entity.document_ids:
                    doc_entities.append(entity_id)
            
            # Create relations between co-occurring entities
            for i, eid1 in enumerate(doc_entities):
                for eid2 in doc_entities[i+1:]:
                    # Check if relation already exists
                    existing_relation = None
                    for rel in self.relations:
                        if (rel.source_id == eid1 and rel.target_id == eid2) or \
                           (rel.source_id == eid2 and rel.target_id == eid1):
                            existing_relation = rel
                            break
                    
                    if existing_relation:
                        existing_relation.weight += 1
                        if doc_id not in existing_relation.document_ids:
                            existing_relation.document_ids.append(doc_id)
                    else:
                        # Determine relation type based on entity types
                        entity1_type = entities[eid1].entity_type
                        entity2_type = entities[eid2].entity_type
                        relation_type = f"{entity1_type}_{entity2_type}"
                        
                        self.relations.append(Relation(
                            source_id=eid1,
                            target_id=eid2,
                            relation_type=relation_type,
                            weight=1.0,
                            document_ids=[doc_id]
                        ))
    
    def _get_entity_type_counts(self, entities: Dict[str, Entity]) -> Dict[str, int]:
        """Get counts of entities by type."""
        type_counts = defaultdict(int)
        for entity in entities.values():
            type_counts[entity.entity_type] += 1
        return dict(type_counts)
    
    def _get_most_connected_nodes(
        self,
        entities: Dict[str, Entity],
        relations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get the most connected nodes in the graph."""
        connection_counts = defaultdict(int)
        
        for rel in relations:
            connection_counts[rel['source']] += 1
            connection_counts[rel['target']] += 1
        
        # Sort by connection count
        sorted_connections = sorted(
            connection_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top 5
        return [
            {
                'entity_id': eid,
                'text': entities[eid].text if eid in entities else eid,
                'connections': count
            }
            for eid, count in sorted_connections[:5]
        ]
    
    async def find_shortest_path(
        self,
        source_id: str,
        target_id: str
    ) -> Optional[List[str]]:
        """Find shortest path between two entities using BFS."""
        if source_id not in self.entities or target_id not in self.entities:
            return None
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for rel in self.relations:
            adjacency[rel.source_id].append(rel.target_id)
            adjacency[rel.target_id].append(rel.source_id)
        
        # BFS
        from collections import deque
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        
        while queue:
            current, path = queue.popleft()
            
            if current == target_id:
                return path
            
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    async def get_entity_neighbors(
        self,
        entity_id: str,
        max_depth: int = 1
    ) -> Dict[str, List[str]]:
        """Get neighbors of an entity up to a certain depth."""
        if entity_id not in self.entities:
            return {}
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for rel in self.relations:
            adjacency[rel.source_id].append(rel.target_id)
            adjacency[rel.target_id].append(rel.source_id)
        
        result = {entity_id: adjacency[entity_id]}
        
        if max_depth > 1:
            current_level = set(adjacency[entity_id])
            for depth in range(2, max_depth + 1):
                next_level = set()
                for node in current_level:
                    next_level.update(adjacency[node])
                result[f'depth_{depth}'] = list(next_level)
                current_level = next_level
        
        return result

# Tool metadata for registration
TOOL_METADATA = {
    "name": "knowledge_graph",
    "description": "Construct and analyze knowledge graphs from documents",
    "parameters": {
        "document_ids": {
            "type": "array",
            "description": "Document IDs to include in graph",
            "items": {"type": "string"}
        },
        "entity_types": {
            "type": "array",
            "description": "Entity types to extract (person, organization, location, etc.)",
            "items": {"type": "string"}
        },
        "min_frequency": {
            "type": "integer",
            "description": "Minimum entity frequency to include",
            "default": 2
        },
        "max_nodes": {
            "type": "integer",
            "description": "Maximum number of nodes in graph",
            "default": 100
        }
    }
}

async def test_knowledge_graph():
    """Test the knowledge graph tool."""
    tool = KnowledgeGraphTool()
    
    # Mock documents
    documents = [
        {
            'id': 'doc1',
            'content': 'John Smith works at Google in Mountain View. He met with Jane Doe from Microsoft last week.'
        },
        {
            'id': 'doc2',
            'content': 'Jane Doe and John Smith discussed the partnership between Google and Microsoft in Seattle.'
        },
        {
            'id': 'doc3',
            'content': 'The Google Microsoft partnership was announced in 2024. Contact john@google.com for details.'
        }
    ]
    
    config = KnowledgeGraphRequest(
        min_frequency=1,
        max_nodes=50
    )
    
    response = await tool.build_graph(documents, config)
    
    print(f"Success: {response.success}")
    print(f"\nEntities ({len(response.entities)}):")
    for entity in response.entities[:10]:
        print(f"  - {entity['text']} ({entity['type']}) - freq: {entity['frequency']}")
    
    print(f"\nRelations ({len(response.relations)}):")
    for rel in response.relations[:10]:
        print(f"  - {rel['source']} -> {rel['target']} ({rel['type']}) - weight: {rel['weight']}")
    
    print(f"\nStatistics:")
    for key, value in response.statistics.items():
        print(f"  - {key}: {value}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_knowledge_graph())
