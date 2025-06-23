"""
Cache manager for translation results.
"""

import json
import logging
from typing import Any, Dict, Optional

import aioredis
from aioredis import Redis

from .config import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class CacheManager:
    """Redis-based cache manager for translation results."""
    
    def __init__(self):
        self.redis: Optional[Redis] = None
        self.enabled = settings.redis_enabled
    
    async def initialize(self):
        """Initialize Redis connection if enabled."""
        if not self.enabled:
            logger.info("Cache disabled")
            return
        
        try:
            self.redis = await aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True
            )
            
            # Test connection
            await self.redis.ping()
            logger.info("Cache initialized with Redis")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self.enabled = False
            self.redis = None
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            logger.info("Cache connection closed")
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value by key."""
        if not self.enabled or not self.redis:
            return None
        
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Dict[str, Any], ttl: int = 3600):
        """Set cached value with TTL."""
        if not self.enabled or not self.redis:
            return
        
        try:
            serialized_value = json.dumps(value, ensure_ascii=False)
            await self.redis.setex(key, ttl, serialized_value)
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def delete(self, key: str):
        """Delete cached value."""
        if not self.enabled or not self.redis:
            return
        
        try:
            await self.redis.delete(key)
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
    
    async def health_check(self) -> str:
        """Check cache health status."""
        if not self.enabled:
            return "disabled"
        
        if not self.redis:
            return "disconnected"
        
        try:
            await self.redis.ping()
            return "healthy"
            
        except Exception:
            return "unhealthy"


# In-memory cache fallback for when Redis is not available
class MemoryCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value."""
        import time
        
        if key in self.cache:
            entry = self.cache[key]
            if entry["expires"] > time.time():
                return entry["value"]
            else:
                del self.cache[key]
        
        return None
    
    async def set(self, key: str, value: Dict[str, Any], ttl: int = 3600):
        """Set cached value with TTL."""
        import time
        
        # Simple LRU eviction
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = {
            "value": value,
            "expires": time.time() + ttl
        }
    
    async def health_check(self) -> str:
        """Check cache health."""
        return "memory"