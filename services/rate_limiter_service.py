"""
Rate Limiter Service for RAG KB.
Implements intelligent rate limiting to prevent abuse and ensure fair usage.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_allowance: int = 10
    enable_ip_tracking: bool = True
    enable_user_tracking: bool = True
    enable_api_key_tracking: bool = True

@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining_requests: int
    reset_time: datetime
    limit_type: str
    retry_after: Optional[int] = None

class RateLimiterService:
    """Service for rate limiting API requests."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        
        # Track requests by identifier (IP, user ID, API key)
        self.request_history: Dict[str, List[datetime]] = defaultdict(list)
        
        # Track burst usage
        self.burst_tokens: Dict[str, int] = defaultdict(lambda: self.config.burst_allowance)
        self.burst_last_refill: Dict[str, datetime] = defaultdict(datetime.now)
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'allowed_requests': 0,
            'blocked_requests': 0,
            'by_identifier': defaultdict(int)
        }
    
    def check_rate_limit(
        self,
        identifier: str,
        identifier_type: str = "ip"
    ) -> RateLimitResult:
        """
        Check if a request is allowed based on rate limits.
        
        Args:
            identifier: Unique identifier (IP, user ID, API key)
            identifier_type: Type of identifier (ip, user, api_key)
        
        Returns:
            RateLimitResult with check outcome
        """
        self.stats['total_requests'] += 1
        self.stats['by_identifier'][identifier] += 1
        
        now = datetime.now()
        key = f"{identifier_type}:{identifier}"
        
        # Clean old request history
        self._cleanup_old_requests(key, now)
        
        # Check burst limit first
        if not self._check_burst_limit(key, now):
            self.stats['blocked_requests'] += 1
            return RateLimitResult(
                allowed=False,
                remaining_requests=0,
                reset_time=now + timedelta(seconds=1),
                limit_type="burst",
                retry_after=1
            )
        
        # Check minute limit
        minute_count = self._count_requests_in_period(key, now, timedelta(minutes=1))
        if minute_count >= self.config.requests_per_minute:
            self.stats['blocked_requests'] += 1
            reset_time = self._get_next_reset_time(key, now, timedelta(minutes=1))
            return RateLimitResult(
                allowed=False,
                remaining_requests=time_remaining(reset_time, now),
                reset_time=reset_time,
                limit_type="minute",
                retry_after=int((reset_time - now).total_seconds())
            )
        
        # Check hour limit
        hour_count = self._count_requests_in_period(key, now, timedelta(hours=1))
        if hour_count >= self.config.requests_per_hour:
            self.stats['blocked_requests'] += 1
            reset_time = self._get_next_reset_time(key, now, timedelta(hours=1))
            return RateLimitResult(
                allowed=False,
                remaining_requests=time_remaining(reset_time, now),
                reset_time=reset_time,
                limit_type="hour",
                retry_after=int((reset_time - now).total_seconds())
            )
        
        # Check day limit
        day_count = self._count_requests_in_period(key, now, timedelta(days=1))
        if day_count >= self.config.requests_per_day:
            self.stats['blocked_requests'] += 1
            reset_time = self._get_next_reset_time(key, now, timedelta(days=1))
            return RateLimitResult(
                allowed=False,
                remaining_requests=time_remaining(reset_time, now),
                reset_time=reset_time,
                limit_type="day",
                retry_after=int((reset_time - now).total_seconds())
            )
        
        # Request allowed - record it
        self.request_history[key].append(now)
        self._consume_burst_token(key, now)
        self.stats['allowed_requests'] += 1
        
        # Calculate remaining requests
        remaining = min(
            self.config.requests_per_minute - minute_count - 1,
            self.config.requests_per_hour - hour_count - 1,
            self.config.requests_per_day - day_count - 1
        )
        
        return RateLimitResult(
            allowed=True,
            remaining_requests=max(0, remaining),
            reset_time=now + timedelta(minutes=1),
            limit_type="combined"
        )
    
    def _cleanup_old_requests(self, key: str, now: datetime):
        """Remove requests older than 1 day."""
        if key not in self.request_history:
            return
        
        cutoff = now - timedelta(days=1)
        self.request_history[key] = [
            req_time for req_time in self.request_history[key]
            if req_time > cutoff
        ]
    
    def _count_requests_in_period(
        self,
        key: str,
        now: datetime,
        period: timedelta
    ) -> int:
        """Count requests within a time period."""
        if key not in self.request_history:
            return 0
        
        cutoff = now - period
        return sum(
            1 for req_time in self.request_history[key]
            if req_time > cutoff
        )
    
    def _get_next_reset_time(
        self,
        key: str,
        now: datetime,
        period: timedelta
    ) -> datetime:
        """Calculate the next reset time for a period."""
        if key not in self.request_history or not self.request_history[key]:
            return now + period
        
        # Find the oldest request in the period
        cutoff = now - period
        oldest_in_period = min(
            req_time for req_time in self.request_history[key]
            if req_time > cutoff
        )
        
        return oldest_in_period + period
    
    def _check_burst_limit(self, key: str, now: datetime) -> bool:
        """Check if burst tokens are available."""
        # Refill burst tokens
        last_refill = self.burst_last_refill[key]
        time_since_refill = (now - last_refill).total_seconds()
        
        # Refill 1 token per second up to max
        tokens_to_add = min(
            int(time_since_refill),
            self.config.burst_allowance
        )
        
        if tokens_to_add > 0:
            self.burst_tokens[key] = min(
                self.config.burst_allowance,
                self.burst_tokens[key] + tokens_to_add
            )
            self.burst_last_refill[key] = now
        
        return self.burst_tokens[key] > 0
    
    def _consume_burst_token(self, key: str, now: datetime):
        """Consume one burst token."""
        if self.burst_tokens[key] > 0:
            self.burst_tokens[key] -= 1
            self.burst_last_refill[key] = now
    
    def reset_identifier(self, identifier: str, identifier_type: str = "ip"):
        """
        Reset rate limit for a specific identifier.
        
        Args:
            identifier: Identifier to reset
            identifier_type: Type of identifier
        """
        key = f"{identifier_type}:{identifier}"
        
        if key in self.request_history:
            del self.request_history[key]
        
        if key in self.burst_tokens:
            del self.burst_tokens[key]
        
        if key in self.burst_last_refill:
            del self.burst_last_refill[key]
        
        logger.info(f"Reset rate limit for {identifier_type}:{identifier}")
    
    def get_identifier_stats(
        self,
        identifier: str,
        identifier_type: str = "ip"
    ) -> Dict[str, Any]:
        """
        Get statistics for a specific identifier.
        
        Args:
            identifier: Identifier to check
            identifier_type: Type of identifier
        
        Returns:
            Dictionary with identifier statistics
        """
        key = f"{identifier_type}:{identifier}"
        now = datetime.now()
        
        minute_count = self._count_requests_in_period(key, now, timedelta(minutes=1))
        hour_count = self._count_requests_in_period(key, now, timedelta(hours=1))
        day_count = self._count_requests_in_period(key, now, timedelta(days=1))
        
        return {
            'identifier': identifier,
            'identifier_type': identifier_type,
            'requests_last_minute': minute_count,
            'requests_last_hour': hour_count,
            'requests_last_day': day_count,
            'burst_tokens_remaining': self.burst_tokens.get(key, self.config.burst_allowance),
            'limits': {
                'per_minute': self.config.requests_per_minute,
                'per_hour': self.config.requests_per_hour,
                'per_day': self.config.requests_per_day,
                'burst': self.config.burst_allowance
            }
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """
        Get global rate limiting statistics.
        
        Returns:
            Dictionary with global statistics
        """
        total_requests = self.stats['total_requests']
        allowed_requests = self.stats['allowed_requests']
        blocked_requests = self.stats['blocked_requests']
        
        return {
            'total_requests': total_requests,
            'allowed_requests': allowed_requests,
            'blocked_requests': blocked_requests,
            'block_rate': blocked_requests / total_requests if total_requests > 0 else 0,
            'active_identifiers': len(self.request_history),
            'top_identifiers': dict(
                sorted(
                    self.stats['by_identifier'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            ),
            'config': {
                'requests_per_minute': self.config.requests_per_minute,
                'requests_per_hour': self.config.requests_per_hour,
                'requests_per_day': self.config.requests_per_day,
                'burst_allowance': self.config.burst_allowance
            }
        }
    
    def update_config(self, new_config: RateLimitConfig):
        """
        Update rate limiting configuration.
        
        Args:
            new_config: New configuration to apply
        """
        self.config = new_config
        logger.info("Updated rate limiting configuration")

def time_remaining(reset_time: datetime, now: datetime) -> int:
    """Calculate remaining requests based on reset time."""
    return max(0, int((reset_time - now).total_seconds()))

# Global rate limiter service instance
rate_limiter_service = RateLimiterService()

def test_rate_limiter():
    """Test the rate limiter service."""
    service = RateLimiterService()
    
    # Test normal requests
    for i in range(5):
        result = service.check_rate_limit("test_ip", "ip")
        print(f"Request {i+1}: Allowed={result.allowed}, Remaining={result.remaining_requests}")
    
    # Get identifier stats
    stats = service.get_identifier_stats("test_ip", "ip")
    print(f"\nIdentifier stats: {stats}")
    
    # Get global stats
    global_stats = service.get_global_stats()
    print(f"\nGlobal stats: {global_stats}")

if __name__ == "__main__":
    test_rate_limiter()
