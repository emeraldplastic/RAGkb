"""
Document summarizer tool for RAG knowledge base.
Provides intelligent document summarization capabilities using the AI agent.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class DocumentSummaryRequest(BaseModel):
    """Request model for document summarization."""
    document_id: str = Field(..., description="ID of the document to summarize")
    summary_type: str = Field(default="executive", description="Type of summary: executive, detailed, bullet_points")
    max_length: int = Field(default=500, description="Maximum length of summary in characters")
    focus_keywords: Optional[List[str]] = Field(default=None, description="Keywords to focus on in summary")

class DocumentSummaryResponse(BaseModel):
    """Response model for document summarization."""
    document_id: str
    summary: str
    summary_type: str
    word_count: int
    key_points: List[str]
    focus_areas_found: List[str]
    success: bool
    error_message: Optional[str] = None

async def document_summarizer_tool(
    document_id: str,
    summary_type: str = "executive",
    max_length: int = 500,
    focus_keywords: Optional[List[str]] = None,
    db_connection = None
) -> DocumentSummaryResponse:
    """
    Summarize a document from the knowledge base using AI-powered analysis.
    
    Args:
        document_id: ID of the document to summarize
        summary_type: Type of summary (executive, detailed, bullet_points)
        max_length: Maximum length of the summary
        focus_keywords: Optional keywords to focus on
        db_connection: Database connection for document retrieval
    
    Returns:
        DocumentSummaryResponse with the generated summary
    """
    try:
        # Import document retrieval functions
        from backend_agent_api.db_documents import get_document_content
        
        # Retrieve document content
        document_content = await get_document_content(document_id, db_connection)
        
        if not document_content:
            return DocumentSummaryResponse(
                document_id=document_id,
                summary="",
                summary_type=summary_type,
                word_count=0,
                key_points=[],
                focus_areas_found=[],
                success=False,
                error_message=f"Document {document_id} not found or empty"
            )
        
        # Generate summary based on type
        if summary_type == "executive":
            summary = await _generate_executive_summary(document_content, max_length)
        elif summary_type == "detailed":
            summary = await _generate_detailed_summary(document_content, max_length)
        elif summary_type == "bullet_points":
            summary = await _generate_bullet_summary(document_content, max_length)
        else:
            summary = await _generate_executive_summary(document_content, max_length)
        
        # Extract key points
        key_points = await _extract_key_points(document_content)
        
        # Check for focus keywords
        focus_areas_found = []
        if focus_keywords:
            content_lower = document_content.lower()
            focus_areas_found = [
                keyword for keyword in focus_keywords 
                if keyword.lower() in content_lower
            ]
        
        # Calculate word count
        word_count = len(summary.split())
        
        return DocumentSummaryResponse(
            document_id=document_id,
            summary=summary,
            summary_type=summary_type,
            word_count=word_count,
            key_points=key_points,
            focus_areas_found=focus_areas_found,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error summarizing document {document_id}: {e}")
        return DocumentSummaryResponse(
            document_id=document_id,
            summary="",
            summary_type=summary_type,
            word_count=0,
            key_points=[],
            focus_areas_found=[],
            success=False,
            error_message=str(e)
        )

async def _generate_executive_summary(content: str, max_length: int) -> str:
    """Generate an executive summary of the document."""
    # Simple heuristic-based summarization
    sentences = content.split('.')
    
    # Take first few sentences as executive summary
    summary_sentences = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and current_length + len(sentence) < max_length:
            summary_sentences.append(sentence)
            current_length += len(sentence) + 2  # +2 for period and space
    
    summary = '. '.join(summary_sentences)
    if not summary.endswith('.'):
        summary += '.'
    
    return summary

async def _generate_detailed_summary(content: str, max_length: int) -> str:
    """Generate a detailed summary of the document."""
    # For detailed summary, take more sentences from throughout the document
    sentences = [s.strip() for s in content.split('.') if s.strip()]
    
    if not sentences:
        return ""
    
    # Take sentences from beginning, middle, and end
    n = len(sentences)
    indices = []
    
    # First 20%
    indices.extend(range(min(5, n // 5)))
    # Middle 20%
    indices.extend(range(n // 2 - 2, min(n // 2 + 3, n)))
    # Last 20%
    indices.extend(range(max(n - 5, 3 * n // 4), n))
    
    # Remove duplicates and sort
    indices = sorted(set(indices))
    
    summary_sentences = []
    current_length = 0
    
    for i in indices:
        if i < len(sentences) and current_length + len(sentences[i]) < max_length:
            summary_sentences.append(sentences[i])
            current_length += len(sentences[i]) + 2
    
    summary = '. '.join(summary_sentences)
    if not summary.endswith('.'):
        summary += '.'
    
    return summary

async def _generate_bullet_summary(content: str, max_length: int) -> str:
    """Generate a bullet-point summary of the document."""
    sentences = [s.strip() for s in content.split('.') if s.strip()]
    
    # Select key sentences based on length and position
    key_sentences = []
    current_length = 0
    
    # Take sentences that are substantial (not too short, not too long)
    for i, sentence in enumerate(sentences):
        if 20 <= len(sentence) <= 150 and current_length + len(sentence) < max_length:
            # Prioritize sentences at beginning and end
            if i < len(sentences) // 3 or i > 2 * len(sentences) // 3:
                key_sentences.append(f"• {sentence}")
                current_length += len(sentence) + 4
    
    return '\n'.join(key_sentences)

async def _extract_key_points(content: str) -> List[str]:
    """Extract key points from the document."""
    # Simple heuristic: find sentences with important keywords
    important_keywords = [
        'important', 'key', 'main', 'critical', 'essential',
        'significant', 'crucial', 'fundamental', 'primary',
        'conclusion', 'result', 'finding', 'discovery'
    ]
    
    sentences = [s.strip() for s in content.split('.') if s.strip()]
    key_points = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in important_keywords):
            if len(sentence) < 200:  # Keep points concise
                key_points.append(sentence)
    
    return key_points[:5]  # Return top 5 key points

# Tool metadata for registration
TOOL_METADATA = {
    "name": "document_summarizer",
    "description": "Summarize documents from the knowledge base with AI-powered analysis",
    "parameters": {
        "document_id": {
            "type": "string",
            "description": "ID of the document to summarize"
        },
        "summary_type": {
            "type": "string",
            "description": "Type of summary: executive, detailed, bullet_points",
            "default": "executive"
        },
        "max_length": {
            "type": "integer",
            "description": "Maximum length of summary in characters",
            "default": 500
        },
        "focus_keywords": {
            "type": "array",
            "description": "Optional keywords to focus on in summary",
            "items": {"type": "string"}
        }
    }
}

async def test_document_summarizer():
    """Test the document summarizer tool."""
    # Mock document content
    test_content = """
    This document outlines the key principles of artificial intelligence and machine learning.
    The main focus is on understanding neural networks and their applications in modern technology.
    Important concepts include deep learning, natural language processing, and computer vision.
    These technologies are transforming industries across the globe.
    The conclusion emphasizes the importance of ethical considerations in AI development.
    Future research should focus on explainable AI and responsible deployment.
    """
    
    # Test different summary types
    print("Executive Summary:")
    print(await _generate_executive_summary(test_content, 200))
    
    print("\nDetailed Summary:")
    print(await _generate_detailed_summary(test_content, 300))
    
    print("\nBullet Summary:")
    print(await _generate_bullet_summary(test_content, 300))
    
    print("\nKey Points:")
    print(await _extract_key_points(test_content))

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_document_summarizer())
