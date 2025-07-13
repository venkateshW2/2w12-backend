# NEW FILE: core/database_manager.py
import redis
import json
import hashlib
import time
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class SoundToolsDatabase:
    """
    Redis-based caching and data management for 2W12 Sound Tools
    Handles analysis caching, research data, and performance tracking
    Falls back to no-cache mode if Redis unavailable
    """
    
    def __init__(self):
        self.redis_available = False
        self.redis_client = None
        
        # Try multiple Redis configurations
        redis_configs = [
            {'host': 'localhost', 'port': 6379},  # Native local Redis
            {'host': '127.0.0.1', 'port': 6379},  # Alternative local
            {'host': 'redis', 'port': 6379},      # Docker service name
        ]
        
        for config in redis_configs:
            try:
                self.redis_client = redis.Redis(
                    host=config['host'],
                    port=config['port'],
                    db=0,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2
                )
                self._test_connection()
                break
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed for {config['host']}:{config['port']}: {e}")
                continue
        
        if not self.redis_available:
            logger.warning("⚠️ Redis not available - running in no-cache mode")
            logger.info("💡 To enable caching, install and start Redis: sudo systemctl start redis")
        
    def _test_connection(self):
        """Test Redis connection and log status"""
        try:
            self.redis_client.ping()
            self.redis_available = True
            logger.info("✅ Redis connection successful")
            
            # Initialize stats if not exist
            if not self.redis_client.exists("stats:cache_hits"):
                self.redis_client.set("stats:cache_hits", 0)
                self.redis_client.set("stats:cache_misses", 0)
                self.redis_client.set("stats:cache_stores", 0)
                logger.info("🆕 Initialized cache statistics")
                
        except Exception as e:
            self.redis_available = False
            raise e  # Let the outer try-catch handle it
    
    def create_file_fingerprint(self, file_content: bytes, filename: str) -> str:
        """
        Create unique fingerprint for file caching
        Uses content sample + metadata for reliable identification
        """
        # Use first 8KB + filename + size for fingerprint
        content_sample = file_content[:8192]
        file_info = f"{filename}_{len(file_content)}"
        fingerprint_data = content_sample + file_info.encode()
        
        return hashlib.md5(fingerprint_data).hexdigest()
    
    def cache_analysis_result(self, fingerprint: str, analysis_data: Dict, ttl: int = 604800) -> bool:
        """
        Cache analysis results for fast retrieval
        
        Args:
            fingerprint: Unique file identifier
            analysis_data: Complete analysis results
            ttl: Time to live in seconds (default: 7 days)
        
        Returns:
            bool: True if cached successfully
        """
        if not self.redis_available:
            logger.debug(f"⚠️ Redis not available - skipping cache store for {fingerprint[:8]}")
            return False
            
        try:
            cache_key = f"analysis:{fingerprint}"
            
            # Add caching metadata
            cache_data = {
                **analysis_data,
                "cached_at": time.time(),
                "fingerprint": fingerprint,
                "cache_version": "v1.0"
            }
            
            # Store with expiration
            success = self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(cache_data, default=str)
            )
            
            if success:
                self.redis_client.incr("stats:cache_stores")
                logger.info(f"✅ Cached analysis for {fingerprint[:8]}... (TTL: {ttl}s)")
                return True
            else:
                logger.error(f"❌ Failed to cache analysis for {fingerprint[:8]}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Cache store failed for {fingerprint[:8]}: {e}")
            return False
    
    def get_cached_analysis(self, fingerprint: str) -> Optional[Dict]:
        """
        Retrieve cached analysis results
        
        Args:
            fingerprint: Unique file identifier
            
        Returns:
            Dict or None: Cached analysis data if found
        """
        if not self.redis_available:
            logger.debug(f"⚠️ Redis not available - cache miss for {fingerprint[:8]}")
            return None
            
        try:
            cache_key = f"analysis:{fingerprint}"
            data = self.redis_client.get(cache_key)
            
            if data:
                self.redis_client.incr("stats:cache_hits")
                # Update TTL on access (extend by 1 day)
                self.redis_client.expire(cache_key, 86400)
                logger.info(f"✅ Cache HIT for {fingerprint[:8]}...")
                return json.loads(data)
            else:
                self.redis_client.incr("stats:cache_misses")
                logger.info(f"❌ Cache MISS for {fingerprint[:8]}...")
                return None
                
        except Exception as e:
            logger.error(f"❌ Cache retrieval failed for {fingerprint[:8]}: {e}")
            if self.redis_available:
                self.redis_client.incr("stats:cache_misses")
            return None
    
    def store_research_data(self, fingerprint: str, ml_data: Dict, validation_data: Dict = None) -> bool:
        """
        Store data for ML research and model improvement
        (Used in Week 2 for MusicBrainz validation)
        
        Args:
            fingerprint: File identifier
            ml_data: ML analysis results
            validation_data: Ground truth data for comparison
        """
        if not self.redis_available:
            logger.debug(f"⚠️ Redis not available - skipping research data store")
            return False
            
        try:
            research_key = f"research:{fingerprint}"
            
            research_record = {
                "fingerprint": fingerprint,
                "ml_analysis": ml_data,
                "validation_data": validation_data,
                "research_timestamp": time.time(),
                "research_version": "v1.0"
            }
            
            # Store research data with longer TTL (30 days)
            success = self.redis_client.setex(
                research_key,
                2592000,  # 30 days
                json.dumps(research_record, default=str)
            )
            
            if success:
                # Add to research queue for batch processing
                self.redis_client.lpush("research_queue", fingerprint)
                logger.info(f"🔬 Stored research data for {fingerprint[:8]}...")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"❌ Research data storage failed: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics"""
        if not self.redis_available:
            return {
                "performance": {
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "cache_stores": 0,
                    "hit_rate_percent": 0.0,
                    "total_requests": 0
                },
                "storage": {
                    "memory_used_mb": 0.0,
                    "cached_analyses": 0,
                    "research_records": 0
                },
                "status": "redis_unavailable",
                "message": "Redis not available - caching disabled"
            }
            
        try:
            hits = int(self.redis_client.get("stats:cache_hits") or 0)
            misses = int(self.redis_client.get("stats:cache_misses") or 0)
            stores = int(self.redis_client.get("stats:cache_stores") or 0)
            
            total_requests = hits + misses
            hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
            
            # Memory usage
            memory_info = self.redis_client.info("memory")
            memory_used_mb = memory_info.get("used_memory", 0) / 1024 / 1024
            
            # Key counts
            analysis_keys = len(self.redis_client.keys("analysis:*"))
            research_keys = len(self.redis_client.keys("research:*"))
            
            return {
                "performance": {
                    "cache_hits": hits,
                    "cache_misses": misses,
                    "cache_stores": stores,
                    "hit_rate_percent": round(hit_rate, 2),
                    "total_requests": total_requests
                },
                "storage": {
                    "memory_used_mb": round(memory_used_mb, 2),
                    "cached_analyses": analysis_keys,
                    "research_records": research_keys
                },
                "status": "healthy" if hit_rate > 0 or total_requests == 0 else "needs_optimization"
            }
            
        except Exception as e:
            logger.error(f"❌ Stats retrieval failed: {e}")
            return {"error": str(e), "status": "error"}
    
    def cleanup_expired_data(self) -> Dict[str, int]:
        """Manual cleanup of expired data (for maintenance)"""
        if not self.redis_available:
            return {
                "deleted_analysis": 0,
                "deleted_research": 0,
                "status": "redis_unavailable",
                "message": "Redis not available - no cleanup needed"
            }
            
        try:
            # This is automatically handled by Redis TTL, but we can force cleanup
            deleted_analysis = 0
            deleted_research = 0
            
            # Find and delete expired keys (Redis handles this automatically, but useful for stats)
            for key in self.redis_client.scan_iter(match="analysis:*"):
                ttl = self.redis_client.ttl(key)
                if ttl == -1:  # No expiration set (shouldn't happen)
                    self.redis_client.expire(key, 604800)  # Set 7-day expiration
                    
            logger.info("🧹 Cleanup completed")
            return {
                "deleted_analysis": deleted_analysis,
                "deleted_research": deleted_research,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return {"error": str(e), "status": "failed"}