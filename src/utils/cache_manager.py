"""
缓存管理系统
提供多级缓存、缓存策略和缓存失效机制
"""

import time
import json
import hashlib
import threading
from typing import Any, Dict, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """缓存策略"""
    LRU = "lru"  # 最近最少使用
    LFU = "lfu"  # 最少使用频率
    FIFO = "fifo"  # 先进先出
    TTL = "ttl"  # 基于时间

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    
    @property
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """更新访问时间"""
        self.last_accessed = time.time()
        self.access_count += 1

class LRUCache:
    """LRU缓存实现"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired:
                    del self.cache[key]
                    return None
                
                # 移动到末尾（最近使用）
                self.cache.move_to_end(key)
                entry.touch()
                return entry.value
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """设置缓存值"""
        with self.lock:
            if key in self.cache:
                # 更新现有条目
                entry = self.cache[key]
                entry.value = value
                entry.created_at = time.time()
                entry.ttl = ttl
                self.cache.move_to_end(key)
            else:
                # 新增条目
                if len(self.cache) >= self.max_size:
                    # 删除最久未使用的条目
                    self.cache.popitem(last=False)
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    access_count=0,
                    ttl=ttl
                )
                self.cache[key] = entry
    
    def delete(self, key: str) -> bool:
        """删除缓存条目"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        with self.lock:
            return len(self.cache)
    
    def cleanup_expired(self):
        """清理过期条目"""
        with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                del self.cache[key]
            return len(expired_keys)

class MultiLevelCache:
    """多级缓存"""
    
    def __init__(self):
        self.l1_cache = LRUCache(max_size=100)  # 内存缓存
        self.l2_cache = LRUCache(max_size=1000)  # 扩展缓存
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            # 先查L1缓存
            value = self.l1_cache.get(key)
            if value is not None:
                return value
            
            # 再查L2缓存
            value = self.l2_cache.get(key)
            if value is not None:
                # 提升到L1缓存
                self.l1_cache.put(key, value)
                return value
            
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """设置缓存值"""
        with self.lock:
            # 同时存储到两级缓存
            self.l1_cache.put(key, value, ttl)
            self.l2_cache.put(key, value, ttl)
    
    def delete(self, key: str):
        """删除缓存条目"""
        with self.lock:
            self.l1_cache.delete(key)
            self.l2_cache.delete(key)
    
    def clear(self):
        """清空所有缓存"""
        with self.lock:
            self.l1_cache.clear()
            self.l2_cache.clear()

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, strategy: CacheStrategy = CacheStrategy.LRU):
        self.strategy = strategy
        self.caches: Dict[str, LRUCache] = {}
        self.global_cache = MultiLevelCache()
        self.lock = threading.RLock()
        
        # 启动清理线程
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def get_cache(self, namespace: str) -> LRUCache:
        """获取命名空间缓存"""
        with self.lock:
            if namespace not in self.caches:
                self.caches[namespace] = LRUCache()
            return self.caches[namespace]
    
    def cache_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _cleanup_loop(self):
        """清理循环"""
        while True:
            try:
                time.sleep(300)  # 每5分钟清理一次
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"缓存清理错误: {e}")
    
    def _cleanup_expired(self):
        """清理过期缓存"""
        with self.lock:
            total_cleaned = 0
            
            # 清理全局缓存
            total_cleaned += self.global_cache.l1_cache.cleanup_expired()
            total_cleaned += self.global_cache.l2_cache.cleanup_expired()
            
            # 清理命名空间缓存
            for cache in self.caches.values():
                total_cleaned += cache.cleanup_expired()
            
            if total_cleaned > 0:
                logger.info(f"清理了 {total_cleaned} 个过期缓存条目")

def cached(
    ttl: Optional[float] = None,
    namespace: str = "default",
    key_func: Optional[Callable] = None
):
    """缓存装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager.cache_key(func.__name__, *args, **kwargs)
            
            # 尝试从缓存获取
            cache = cache_manager.get_cache(namespace)
            cached_result = cache.get(cache_key)
            
            if cached_result is not None:
                logger.debug(f"缓存命中: {func.__name__}")
                return cached_result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl)
            logger.debug(f"缓存存储: {func.__name__}")
            
            return result
        
        return wrapper
    return decorator

def async_cached(
    ttl: Optional[float] = None,
    namespace: str = "default",
    key_func: Optional[Callable] = None
):
    """异步缓存装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager.cache_key(func.__name__, *args, **kwargs)
            
            # 尝试从缓存获取
            cache = cache_manager.get_cache(namespace)
            cached_result = cache.get(cache_key)
            
            if cached_result is not None:
                logger.debug(f"缓存命中: {func.__name__}")
                return cached_result
            
            # 执行函数并缓存结果
            result = await func(*args, **kwargs)
            cache.put(cache_key, result, ttl)
            logger.debug(f"缓存存储: {func.__name__}")
            
            return result
        
        return wrapper
    return decorator

class CacheStats:
    """缓存统计"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.puts = 0
        self.deletes = 0
        self.lock = threading.Lock()
    
    def record_hit(self):
        with self.lock:
            self.hits += 1
    
    def record_miss(self):
        with self.lock:
            self.misses += 1
    
    def record_put(self):
        with self.lock:
            self.puts += 1
    
    def record_delete(self):
        with self.lock:
            self.deletes += 1
    
    @property
    def hit_rate(self) -> float:
        with self.lock:
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "puts": self.puts,
                "deletes": self.deletes,
                "hit_rate": self.hit_rate,
                "total_requests": self.hits + self.misses
            }

# 全局缓存管理器实例
cache_manager = CacheManager()
cache_stats = CacheStats()