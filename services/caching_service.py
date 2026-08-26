"""
Caching Service for RAG KB.
Implements intelligent caching for improved performance and reduced API costs.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import hashlib
import json
import logging
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int
    last_accessed: datetime
    size_bytes: int
    tags: List[str]

@dataclass
class CacheConfig:
    """Configuration for caching service."""
    default_ttl_seconds: int = 3600  # 1 hour default
    max_size_mb: int = 100  # Max cache size in MB
    max_entries: int = 10000  # Max number of entries
    enable_compression: bool = False
    cleanup_interval_seconds: int = 300  # 5 minutes
    hit_rate_threshold: float = 0.7  # Minimum hit rate to keep cache

class CachingService:
    """Service for intelligent caching of RAG operations."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_bytes': 0
        }
    
    def generate_key(
        self,
        operation: str,
        params: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Generate a cache key from operation and parameters.
        
        Args:
            operation: Name of the operation being cached
            params: Parameters for the operation
            tags: Optional tags for the cache entry
        
        Returns:
            Cache key string
        """
        # Create a deterministic key from params
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        
        key = f"{operation}:{param_hash}"
        
        return key
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self.cache:
            self.stats['misses'] += 1
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if entry.expires_at and datetime.now() > entry.expires_at:
            self._remove_entry(key)
            self.stats['misses'] += 1
            return None
        
        # Update access statistics
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        
        self.stats['hits'] += 1
        logger.debug(f"Cache hit for key: {key}")
        
        return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Store a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            tags: Optional tags for the entry
        
        Returns:
            True if cached successfully, False otherwise
        """
        # Check cache size limits
        if len(self.cache) >= self.config.max_entries:
            self._evict_lru()
        
        # Calculate expiration
        ttl = ttl_seconds or self.config.default_ttl_seconds
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None
        
        # Calculate size
        try:
            size_bytes = len(json.dumps(value).encode())
        except:
            size_bytes = len(str(value).encode())
        
        # Check size limit
        if self.stats['size_bytes'] + size_bytes > self.config.max_size_mb * 1024 * 1024:
            self._evict_by_size()
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            expires_at=expires_at,
            access_count=0,
            last_accessed=datetime.now(),
            size_bytes=size_bytes,
            tags=tags or []
        )
        
        self.cache[key] = entry
        self.stats['size_bytes'] += size_bytes
        
        logger.debug(f"Cached value for key: {key} (size: {size_bytes} bytes)")
        
        return True
    
    def delete(self, key: str) -> bool:
        """
        Delete a specific cache entry.
        
        Args:
            key: Cache key to delete
        
        Returns:
            True if deleted, False if not found
        """
        if key in self.cache:
            self._remove_entry(key)
            logger.debug(f"Deleted cache entry: {key}")
            return True
        
        return False
    
    def delete_by_tag(self, tag: str) -> int:
        """
        Delete all cache entries with a specific tag.
        
        Args:
            tag: Tag to filter by
        
        Returns:
            Number of entries deleted
        """
        keys_to_delete = [
            key for key, entry in self.cache.items()
            if tag in entry.tags
        ]
        
        for key in keys_to_delete:
            self._remove_entry(key)
        
        logger.debug(f"Deleted {len(keys_to_delete)} entries with tag: {tag}")
        
        return len(keys_to_delete)
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        count = len(self.cache)
        self.cache.clear()
        self.stats['size_bytes'] = 0
        
        logger.info(f"Cleared {count} cache entries")
        
        return count
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.expires_at and now > entry.expires_at
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'entries': len(self.cache),
            'size_bytes': self.stats['size_bytes'],
            'size_mb': self.stats['size_bytes'] / (1024 * 1024),
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'evictions': self.stats['evictions'],
            'config': {
                'max_entries': self.config.max_entries,
                'max_size_mb': self.config.max_size_mb,
                'default_ttl_seconds': self.config.default_ttl_seconds
            }
        }
    
    def _remove_entry(self, key: str):
        """Remove an entry and update size statistics."""
        if key in self.cache:
            entry = self.cache[key]
            self.stats['size_bytes'] -= entry.size_bytes
            del self.cache[key]
    
    def _evict_lru(self):
        """Evict least recently used entries."""
        if not self.cache:
            return
        
        # Sort by last accessed time
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Evict 10% of entries
        evict_count = max(1, len(sorted_entries) // 10)
        
        for key, _ in sorted_entries[:evict_count]:
            self._remove_entry(key)
            self.stats['evictions'] += 1
        
        logger.debug(f"Evicted {evict_count} LRU entries")
    
    def _evict_by_size(self):
        """Evict entries to free up space."""
        if not self.cache:
            return
        
        # Sort by size (largest first)
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].size_bytes,
            reverse=True
        )
        
        target_size = self.config.max_size_mb * 1024 * 1024 * 0.8  # Target 80% of max
        
        freed_bytes = 0
        for key, entry in sorted_entries:
            if self.stats['size_bytes'] - freed_bytes <= target_size:
                break
            
            self._remove_entry(key)
            freed_bytes += entry.size_bytes
            self.stats['evictions'] += 1
        
        logger.debug(f"Evicted entries to free {freed_bytes} bytes")
    
    def cache_decorator(
        self,
        operation: str,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Decorator for caching function results.
        
        Args:
            operation: Name of the operation
            ttl_seconds: Time to live in seconds
            tags: Optional tags for cache entries
        
        Returns:
            Decorator function
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                params = {'args': args, 'kwargs': kwargs}
                key = self.generate_key(operation, params, tags)
                
                # Try to get from cache
                cached_result = self.get(key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                self.set(key, result, ttl_seconds, tags)
                
                return result
            
            return wrapper
        return decorator

# Global caching service instance
caching_service = CachingService()

def test_caching_service():
    """Test the caching service."""
    service = CachingService()
    
    # Test basic get/set
    key = service.generate_key("test_op", {"param": "value"})
    service.set(key, {"result": "data"}, ttl_seconds=60)
    
    result = service.get(key)
    print(f"Get result: {result}")
    
    # Test cache miss
    miss_result = service.get("nonexistent_key")
    print(f"Miss result: {miss_result}")
    
    # Test statistics
    stats = service.get_stats()
    print(f"Cache stats: {stats}")
    
    # Test cleanup
    service.set(key, {"result": "data"}, ttl_seconds=1)
    import time
    time.sleep(2)
    cleaned = service.cleanup_expired()
    print(f"Cleaned {cleaned} expired entries")

if __name__ == "__main__":
    test_caching_service()
