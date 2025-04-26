from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
import os
import logging
import time
from collections import defaultdict
import random
from dataclasses import dataclass
import math

# Attempt to import pmemobj - this will fail if not installed
try:
    import pmemobj
except ImportError:
    pmemobj = None
    logging.warning("PMDK Python binding (pmemobj) not found. Real PMDKBackend will not be functional.")
    # Try importing a potential base error class
    try:
        from pmemobj import Error as PmemError
    except ImportError:
        PmemError = None # Fallback if no specific base error exists

# Attempt to import prometheus_client conditionally
try:
    from prometheus_client import Counter, Gauge, start_http_server, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus-client not found. Prometheus metrics export will not be available.")
    # Define dummy classes if not available, so code doesn't break
    class DummyMetric:
        def labels(self, *args, **kwargs): return self
        def inc(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
    Counter = Gauge = DummyMetric
    Histogram = Summary = DummyMetric

# Restored Class Definitions
class StorageBackend(ABC):
    """
    Abstract base class for VUA storage backends.
    Defines the interface for storing and retrieving cache fragments.
    """
    @abstractmethod
    def put(self, group_hash: str, data: bytes, tokens: bytes) -> None:
        """
        Store a cache fragment identified by group_hash.
        Args:
            group_hash: Unique identifier for the group (hash string).
            data: Serialized cache data (e.g., tensor bytes).
            tokens: Serialized tokens data.
        """
        pass

    @abstractmethod
    def get(self, group_hash: str) -> Optional[tuple[bytes, bytes]]:
        """
        Retrieve a cache fragment by group_hash.
        Returns:
            Tuple of (data, tokens) if found, else None.
        """
        pass

    @abstractmethod
    def exists(self, group_hash: str) -> bool:
        """
        Check if a cache fragment exists for the given group_hash.
        """
        pass

class FileSystemBackend(StorageBackend):
    """
    Storage backend that uses the local filesystem (current VUA behavior).
    Stores each group in a directory named by group_hash, with files '_data.safetensors' and '_tokens.safetensors'.
    """
    def __init__(self, root_path: str):
        self.root_path = root_path

    def put(self, group_hash: str, data: bytes, tokens: bytes) -> None:
        """
        Store data and tokens as files in a directory named by group_hash under root_path.
        Args:
            group_hash: Directory name for the group.
            data: Bytes to write to '_data.safetensors'.
            tokens: Bytes to write to '_tokens.safetensors'.
        """
        group_dir = os.path.join(self.root_path, group_hash)
        os.makedirs(group_dir, exist_ok=True)
        data_path = os.path.join(group_dir, "_data.safetensors")
        tokens_path = os.path.join(group_dir, "_tokens.safetensors")
        with open(data_path, "wb") as f:
            f.write(data)
        with open(tokens_path, "wb") as f:
            f.write(tokens)

    def get(self, group_hash: str) -> Optional[tuple[bytes, bytes]]:
        """
        Retrieve data and tokens from files in the group_hash directory.
        Returns (data_bytes, tokens_bytes) if found, else None.
        """
        group_dir = os.path.join(self.root_path, group_hash)
        data_path = os.path.join(group_dir, "_data.safetensors")
        tokens_path = os.path.join(group_dir, "_tokens.safetensors")
        try:
            with open(data_path, "rb") as f:
                data = f.read()
            with open(tokens_path, "rb") as f:
                tokens = f.read()
            return (data, tokens)
        except (FileNotFoundError, IsADirectoryError):
            return None

    def exists(self, group_hash: str) -> bool:
        """
        Check if the group_hash directory exists and contains both data and tokens files.
        """
        group_dir = os.path.join(self.root_path, group_hash)
        data_path = os.path.join(group_dir, "_data.safetensors")
        tokens_path = os.path.join(group_dir, "_tokens.safetensors")
        return os.path.isdir(group_dir) and os.path.isfile(data_path) and os.path.isfile(tokens_path)

class MockPMDKBackend(StorageBackend):
    """
    Mock backend that simulates PMDK persistent memory using an in-memory dict.
    Useful for development and testing without PMEM/CXL hardware.
    """
    def __init__(self):
        self._store = {}

    def put(self, group_hash: str, data: bytes, tokens: bytes) -> None:
        self._store[group_hash] = (data, tokens)

    def get(self, group_hash: str) -> Optional[tuple[bytes, bytes]]:
        return self._store.get(group_hash)

    def exists(self, group_hash: str) -> bool:
        return group_hash in self._store

class MockGPURAMBackend(StorageBackend):
    """
    Mock backend simulating GPU RAM using an in-memory dict.
    """
    def __init__(self):
        self._store = {}

    def put(self, group_hash: str, data: bytes, tokens: bytes) -> None:
        self._store[group_hash] = (data, tokens)

    def get(self, group_hash: str) -> Optional[tuple[bytes, bytes]]:
        return self._store.get(group_hash)

    def exists(self, group_hash: str) -> bool:
        return group_hash in self._store

class PMDKBackend(StorageBackend):
    """
    PMDK backend for VUA using libpmemobj.
    Stores data in a persistent memory pool.
    Requires the pmemobj Python binding and PMDK installed.
    """
    # Define a simple layout name for the pool
    LAYOUT_NAME = "vua_kv_cache"

    def __init__(self, pool_path: str, pool_size: int = 1024 * 1024 * 1024, create=True):
        """
        Initialize the PMDK backend, creating or opening the pool.
        Args:
            pool_path: Path to the persistent memory pool file.
            pool_size: Size of the pool to create (in bytes) if it doesn't exist.
            create: If True, create the pool if it doesn't exist. If False, only open.
        """
        if pmemobj is None:
            raise ImportError("PMDK pmemobj binding not found. Cannot initialize PMDKBackend.")

        self.pool_path = pool_path
        self.pool_size = pool_size
        self.pool = None
        self.logger = logging.getLogger(f"PMDKBackend({pool_path})")

        try:
            if os.path.exists(pool_path):
                self.logger.info(f"Opening existing PMDK pool: {pool_path}")
                self.pool = pmemobj.open(pool_path, self.LAYOUT_NAME)
            elif create:
                self.logger.info(f"Creating new PMDK pool: {pool_path} (size: {pool_size} bytes)")
                self.pool = pmemobj.create(pool_path, self.LAYOUT_NAME, pool_size)
                # Initialize the root object (e.g., a persistent dict)
                with self.pool.transaction():
                    self.pool.root = pmemobj.PersistentDict()
            else:
                # Raise explicitly if pool not found and create=False
                raise FileNotFoundError(f"PMDK pool file not found and create=False: {pool_path}")
        except FileNotFoundError as e: # Catch specific error
            self.logger.error(f"PMDK pool file not found: {e}")
            raise
        except PermissionError as e: # Catch specific error
            self.logger.error(f"Permission denied for PMDK pool at {pool_path}: {e}")
            raise
        except FileExistsError as e: # Catch specific error (less likely here, maybe with create=True?)
            self.logger.error(f"PMDK pool file unexpectedly exists: {e}")
            raise
        except OSError as e: # Catch other OS-level errors
            self.logger.error(f"OS error during PMDK pool open/create at {pool_path}: {e}")
            raise
        except (PmemError, Exception) if PmemError else Exception as e: # Catch PMDK specific or general errors
            self.logger.error(f"Failed to open or create PMDK pool at {pool_path}: {e}")
            raise

    def put(self, group_hash: str, data: bytes, tokens: bytes) -> None:
        """
        Store data and tokens persistently in the PMDK pool.
        Uses a transaction for atomicity.
        """
        if self.pool is None:
            raise RuntimeError("PMDK pool is not open.")

        try:
            with self.pool.transaction():
                # Allocate persistent memory for data and tokens
                # Note: pmemobj might require specific types (e.g., PersistentBytes)
                # This assumes data/tokens are bytes; adjust if needed.
                persistent_data = pmemobj.PersistentBytes(data)
                persistent_tokens = pmemobj.PersistentBytes(tokens)
                # Store in the root persistent dictionary
                self.pool.root[group_hash] = (persistent_data, persistent_tokens)
        except MemoryError as e: # Catch potential out-of-space
            self.logger.error(f"Out of memory in PMDK pool while putting {group_hash}: {e}")
            # TODO: More specific handling if needed (e.g., trigger eviction?)
            raise
        except (PmemError, Exception) if PmemError else Exception as e: # Catch PMDK specific or general errors
            self.logger.error(f"Failed to put {group_hash} into PMDK pool: {e}")
            # TODO: Handle specific PMDK transaction errors?
            raise

    def get(self, group_hash: str) -> Optional[tuple[bytes, bytes]]:
        """
        Retrieve data and tokens from the PMDK pool.
        Returns a tuple of (bytes, bytes) or None if not found.
        """
        if self.pool is None:
            raise RuntimeError("PMDK pool is not open.")

        try:
            persistent_tuple = self.pool.root.get(group_hash)
            if persistent_tuple:
                # Return copies as regular bytes
                data_bytes = bytes(persistent_tuple[0])
                tokens_bytes = bytes(persistent_tuple[1])
                return (data_bytes, tokens_bytes)
            else:
                return None
        except (PmemError, Exception) if PmemError else Exception as e: # Catch PMDK specific or general errors
            self.logger.error(f"Failed to get {group_hash} from PMDK pool: {e}")
            raise

    def exists(self, group_hash: str) -> bool:
        """
        Check if a group hash exists as a key in the PMDK pool's root dictionary.
        """
        if self.pool is None:
            raise RuntimeError("PMDK pool is not open.")

        try:
            return group_hash in self.pool.root
        except (PmemError, Exception) if PmemError else Exception as e: # Catch PMDK specific or general errors
            self.logger.error(f"Failed to check existence of {group_hash} in PMDK pool: {e}")
            raise

    def close(self):
        """
        Close the PMDK pool.
        """
        if self.pool:
            try:
                self.pool.close()
                self.logger.info(f"Closed PMDK pool: {self.pool_path}")
            except (PmemError, Exception) if PmemError else Exception as e: # Catch PMDK specific or general errors
                self.logger.error(f"Error closing PMDK pool {self.pool_path}: {e}")
            finally:
                self.pool = None

    def __del__(self):
        # Ensure pool is closed when object is garbage collected
        self.close()

# TieredBackend follows

@dataclass
class CacheEntry:
    """Metadata for a cached item used by tiering policies."""
    group_hash: str
    size_bytes: int
    last_access_time: float
    access_count: int
    tier_idx: int

@dataclass
class PolicyConfig:
    """Configuration for tiering policies."""
    learning_rate: float = 0.1
    exploration_rate: float = 0.1
    initial_lru_weight: float = 0.5
    initial_lfu_weight: float = 0.5
    ghost_cache_ttl: int = 3600  # seconds
    min_weight: float = 0.1  # minimum weight for any policy
    max_weight: float = 0.9  # maximum weight for any policy
    base_promotion_score_threshold: float = 0.7 # Base threshold for promotion score
    base_demotion_score_threshold: float = 0.3  # Base threshold for demotion score

class TieringPolicy(ABC):
    """Abstract base class for tiering policies..."""
    @abstractmethod
    def update(self, entry: CacheEntry, hit: bool) -> None:
        pass
    def evict(self, entry: CacheEntry) -> None:
        pass

class LeCAR(TieringPolicy):
    """Learning Cache Admission and Replacement (LeCAR) policy.
    
    Combines LRU and LFU policies with online learning to adapt to workload patterns.
    Maintains weights for each policy that are updated based on hits/misses.
    
    Configuration:
    - learning_rate: How quickly policy weights adapt (0.0-1.0)
    - exploration_rate: Probability of random decisions for exploration (0.0-1.0)
    - initial_lru_weight: Starting weight for LRU policy (0.0-1.0)
    - initial_lfu_weight: Starting weight for LFU policy (0.0-1.0)
    - ghost_cache_ttl: Time to keep entries in ghost cache (seconds)
    - min_weight: Minimum weight for any policy (0.0-1.0)
    - max_weight: Maximum weight for any policy (0.0-1.0)
    - base_promotion_score_threshold: Base threshold score for promotion (0.0-1.0)
    - base_demotion_score_threshold: Base threshold score for demotion (0.0-1.0)
    """
    
    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or PolicyConfig()
        self.logger = logging.getLogger("LeCAR") 
        
        self.lru_weight = self.config.initial_lru_weight
        self.lfu_weight = self.config.initial_lfu_weight
        self._entries: Dict[str, CacheEntry] = {}
        self._lru_list: List[str] = []
        self._lfu_scores: Dict[str, float] = defaultdict(float)
        self._ghost_lru: Dict[str, float] = {}
        self._ghost_lfu: Dict[str, float] = {}
        
        # Prometheus metrics setup
        self._prom_metrics = defaultdict(lambda: None) # Use defaultdict
        if PROMETHEUS_AVAILABLE:
            try:
                score_buckets = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
                self._prom_metrics['lru_weight'] = Gauge('vua_policy_lru_weight', 'Current LRU policy weight')
                self._prom_metrics['lfu_weight'] = Gauge('vua_policy_lfu_weight', 'Current LFU policy weight')
                self._prom_metrics['ghost_lru_size'] = Gauge('vua_policy_ghost_lru_size', 'Number of entries in LRU ghost cache')
                self._prom_metrics['ghost_lfu_size'] = Gauge('vua_policy_ghost_lfu_size', 'Number of entries in LFU ghost cache')
                self._prom_metrics['ghost_hits'] = Counter('vua_policy_ghost_hits_total', 'Total ghost cache hits (hits on items previously evicted)')
                self._prom_metrics['ghost_additions'] = Counter('vua_policy_ghost_additions_total', 'Total entries added to ghost cache upon eviction')
                self._prom_metrics['exploration_decisions'] = Counter('vua_policy_exploration_decisions_total', 'Total random exploration decisions')
                self._prom_metrics['admit_scores'] = Histogram('vua_policy_admit_scores', 'Distribution of scores during admission decisions', buckets=score_buckets)
                self._prom_metrics['promote_scores'] = Histogram('vua_policy_promote_scores', 'Distribution of scores during promotion decisions', buckets=score_buckets)
                self._prom_metrics['demote_scores'] = Histogram('vua_policy_demote_scores', 'Distribution of scores during demotion decisions', buckets=score_buckets)
                self._prom_metrics['victim_scores'] = Histogram('vua_policy_victim_scores', 'Distribution of scores of selected eviction victims', buckets=score_buckets)
            except Exception as e:
                self.logger.error(f"Failed to initialize Prometheus metrics: {e}")
                # Fallback to defaultdict means metrics calls will be no-ops

    def update_config(self, new_config: PolicyConfig) -> None:
        """Update policy configuration at runtime."""
        old_config = self.config
        self.config = new_config
        
        self.lru_weight = max(min(self.lru_weight, new_config.max_weight), new_config.min_weight)
        self.lfu_weight = max(min(self.lfu_weight, new_config.max_weight), new_config.min_weight)
        if not math.isclose(self.lru_weight + self.lfu_weight, 1.0):
             self.lfu_weight = 1.0 - self.lru_weight
             self.lfu_weight = max(min(self.lfu_weight, new_config.max_weight), new_config.min_weight)
             self.lru_weight = 1.0 - self.lfu_weight

        if (abs(old_config.learning_rate - new_config.learning_rate) > 0.01 or
            abs(old_config.exploration_rate - new_config.exploration_rate) > 0.01 or
            abs(old_config.base_promotion_score_threshold - new_config.base_promotion_score_threshold) > 0.01 or
            abs(old_config.base_demotion_score_threshold - new_config.base_demotion_score_threshold) > 0.01):
            self.logger.info(
                f"Policy config updated: LR={new_config.learning_rate:.3f}({old_config.learning_rate:.3f}) "
                f"ER={new_config.exploration_rate:.3f}({old_config.exploration_rate:.3f}) "
                f"PT={new_config.base_promotion_score_threshold:.2f}({old_config.base_promotion_score_threshold:.2f}) "
                f"DT={new_config.base_demotion_score_threshold:.2f}({old_config.base_demotion_score_threshold:.2f})"
            )
    
    def _get_score(self, entry: CacheEntry) -> float:
        lru_score = self._get_lru_score(entry.group_hash)
        lfu_score = self._get_lfu_score(entry.group_hash)
        score = (self.lru_weight * lru_score + self.lfu_weight * lfu_score)
        return max(0.0, min(1.0, score)) # Clamp score
    
    def _get_lru_score(self, group_hash: str) -> float:
        try:
            idx = self._lru_list.index(group_hash)
            list_len = len(self._lru_list)
            if list_len <= 1: return 0.0
            score = idx / (list_len - 1)
            return max(0.0, min(1.0, score))
        except ValueError: return 0.0
            
    def _get_lfu_score(self, group_hash: str) -> float:
        current_count = self._lfu_scores.get(group_hash, 0.0)
        if not self._lfu_scores or current_count == 0.0: return 0.0
        max_count = max(self._lfu_scores.values()) if self._lfu_scores else 1.0
        score = current_count / max(max_count, 1.0)
        return max(0.0, min(1.0, score))
        
    def should_admit(self, entry: CacheEntry, tier_idx: int) -> bool:
        if entry.group_hash not in self._entries:
            self._entries[entry.group_hash] = entry
            if entry.group_hash not in self._lru_list:
                 self._lru_list.append(entry.group_hash)
            self._lfu_scores[entry.group_hash] = entry.access_count 
        score = self._get_score(entry)
        if PROMETHEUS_AVAILABLE and self._prom_metrics['admit_scores']:
            self._prom_metrics['admit_scores'].observe(score)
        if random.random() < self.config.exploration_rate:
            if PROMETHEUS_AVAILABLE and self._prom_metrics['exploration_decisions']:
                self._prom_metrics['exploration_decisions'].inc()
            return random.random() > 0.5 
        return score > 0.5
        
    def should_promote(self, entry: CacheEntry, from_tier: int, to_tier: int) -> bool:
        score = self._get_score(entry)
        base_threshold = self.config.base_promotion_score_threshold
        threshold = max(0.0, min(1.0, base_threshold - (0.1 * (from_tier - to_tier))))
        if PROMETHEUS_AVAILABLE and self._prom_metrics['promote_scores']:
            self._prom_metrics['promote_scores'].observe(score)
        if random.random() < self.config.exploration_rate:
            if PROMETHEUS_AVAILABLE and self._prom_metrics['exploration_decisions']:
                self._prom_metrics['exploration_decisions'].inc()
            return random.random() > threshold 
        return score > threshold
        
    def should_demote(self, entry: CacheEntry, from_tier: int, to_tier: int) -> bool:
        score = self._get_score(entry)
        base_threshold = self.config.base_demotion_score_threshold
        threshold = max(0.0, min(1.0, base_threshold + (0.1 * (to_tier - from_tier))))
        if PROMETHEUS_AVAILABLE and self._prom_metrics['demote_scores']:
            self._prom_metrics['demote_scores'].observe(score)
        if random.random() < self.config.exploration_rate:
            if PROMETHEUS_AVAILABLE and self._prom_metrics['exploration_decisions']:
                self._prom_metrics['exploration_decisions'].inc()
            return random.random() < threshold 
        return score < threshold
        
    def select_victim(self, entries: List[CacheEntry], tier_idx: int) -> Optional[str]:
        if not entries: return None
        valid_entries = [e for e in entries if e.group_hash in self._entries]
        if not valid_entries:
            self.logger.warning(f"select_victim tier {tier_idx}: No tracked entries. Falling back.")
            victim_score = 0.0
            victim_hash = entries[0].group_hash
        else:
            if random.random() < self.config.exploration_rate:
                if PROMETHEUS_AVAILABLE and self._prom_metrics['exploration_decisions']:
                    self._prom_metrics['exploration_decisions'].inc()
                victim_entry = random.choice(valid_entries)
                victim_score = self._get_score(victim_entry)
                victim_hash = victim_entry.group_hash
            else:
                victim_entry = min(valid_entries, key=lambda e: self._get_score(e))
                victim_score = self._get_score(victim_entry)
                victim_hash = victim_entry.group_hash
        if PROMETHEUS_AVAILABLE and self._prom_metrics['victim_scores']:
            self._prom_metrics['victim_scores'].observe(victim_score)
        return victim_hash
        
    def update(self, entry: CacheEntry, hit: bool) -> None:
        group_hash = entry.group_hash
        self._entries[group_hash] = entry 
        if group_hash in self._lru_list:
            self._lru_list.remove(group_hash)
        self._lru_list.append(group_hash) 
        self._lfu_scores[group_hash] = self._lfu_scores.get(group_hash, 0) + 1
        
        if hit:
            ghost_hit = False
            if group_hash in self._ghost_lru:
                ghost_hit = True
                delta = self.config.learning_rate * (1 - self.lru_weight)
                self.lru_weight = min(self.config.max_weight, self.lru_weight + delta)
                del self._ghost_lru[group_hash]
            elif group_hash in self._ghost_lfu:
                ghost_hit = True
                delta = self.config.learning_rate * (1 - self.lfu_weight)
                self.lfu_weight = min(self.config.max_weight, self.lfu_weight + delta)
                del self._ghost_lfu[group_hash]
                
            if ghost_hit:
                if not math.isclose(self.lru_weight + self.lfu_weight, 1.0):
                    self.lfu_weight = 1.0 - self.lru_weight
                    self.lfu_weight = max(min(self.lfu_weight, self.config.max_weight), self.config.min_weight)
                    self.lru_weight = 1.0 - self.lfu_weight
                if PROMETHEUS_AVAILABLE and self._prom_metrics['ghost_hits']:
                    self._prom_metrics['ghost_hits'].inc()
        
        if PROMETHEUS_AVAILABLE:
            if self._prom_metrics['lru_weight']:
                 self._prom_metrics['lru_weight'].set(self.lru_weight)
            if self._prom_metrics['lfu_weight']:
                 self._prom_metrics['lfu_weight'].set(self.lfu_weight)

    def evict(self, entry: CacheEntry) -> None:
        group_hash = entry.group_hash
        curr_time = time.time()
        if group_hash in self._entries:
            lru_score = self._get_lru_score(group_hash)
            lfu_score = self._get_lfu_score(group_hash)
            if lru_score < lfu_score: self._ghost_lru[group_hash] = curr_time
            else: self._ghost_lfu[group_hash] = curr_time
            if PROMETHEUS_AVAILABLE and self._prom_metrics['ghost_additions']:
                self._prom_metrics['ghost_additions'].inc()
            del self._entries[group_hash]
            if group_hash in self._lru_list: 
                try: self._lru_list.remove(group_hash) 
                except ValueError: pass 
            if group_hash in self._lfu_scores: 
                del self._lfu_scores[group_hash]
        else: self.logger.debug(f"evict called for untracked entry {group_hash}.")
        cutoff = curr_time - self.config.ghost_cache_ttl
        self._ghost_lru = {k: v for k, v in self._ghost_lru.items() if v > cutoff}
        self._ghost_lfu = {k: v for k, v in self._ghost_lfu.items() if v > cutoff}
        if PROMETHEUS_AVAILABLE:
            if self._prom_metrics['ghost_lru_size']:
                 self._prom_metrics['ghost_lru_size'].set(len(self._ghost_lru))
            if self._prom_metrics['ghost_lfu_size']:
                 self._prom_metrics['ghost_lfu_size'].set(len(self._ghost_lfu))

class TieredBackend(StorageBackend):
    def __init__(self, backends, tier_configs, advanced_watermark=0.9, adjustment_interval=1000, policy=None, policy_config=None):
        self.backends = backends
        self.tier_configs = tier_configs
        self.advanced_watermark = advanced_watermark
        self.metrics = {
            i: {
                'hits': 0,
                'misses': 0,
                'promotions': 0,
                'demotions': 0,
                'evictions': 0,
                'last_adjustment': time.time(),
                # Optionally, rolling window of recent operations
            }
            for i in range(len(backends))
        }
        self.adjustment_interval = adjustment_interval
        self.op_count = 0
        self.logger = logging.getLogger("TieredBackend")

        # --- Prometheus Metrics Definition ---
        self._prom_labels = ['tier_name']
        self._prom_hits = Counter('vua_tier_hits_total', 'Total cache hits for a tier', self._prom_labels)
        self._prom_misses = Counter('vua_tier_misses_total', 'Total cache misses for a tier', self._prom_labels)
        self._prom_promotions = Counter('vua_tier_promotions_total', 'Total promotions *into* a tier', self._prom_labels)
        self._prom_demotions = Counter('vua_tier_demotions_total', 'Total demotions *from* a tier', self._prom_labels)
        self._prom_evictions = Counter('vua_tier_evictions_total', 'Total evictions from a tier', self._prom_labels)
        self._prom_usage_count = Gauge('vua_tier_usage_fragments', 'Current fragment count in a tier', self._prom_labels)
        self._prom_usage_bytes = Gauge('vua_tier_usage_bytes', 'Current byte usage in a tier', self._prom_labels)
        self._prom_capacity_count = Gauge('vua_tier_capacity_fragments', 'Fragment capacity of a tier', self._prom_labels)
        self._prom_capacity_bytes = Gauge('vua_tier_capacity_bytes', 'Byte capacity of a tier', self._prom_labels)
        self._prom_advanced_active = Gauge('vua_tier_advanced_metadata_active_fragments', 'Number of fragments using advanced metadata per tier', self._prom_labels)
        self._prom_avg_eviction_score = Gauge('vua_tier_avg_eviction_score', 'Average eviction score per tier', self._prom_labels)

        # --- New Aggregated/Distribution Metrics ---
        # Using Histogram for distributions - define buckets appropriately
        age_buckets = (1, 5, 15, 60, 300, 1800, 3600, float('inf')) # seconds
        count_buckets = (1, 2, 5, 10, 25, 100, float('inf'))
        size_buckets = (1024, 4096, 16384, 65536, 262144, 1048576, float('inf')) # bytes

        self._prom_frag_age = Histogram('vua_tier_fragment_age_seconds', 'Distribution of fragment ages per tier (last access)', self._prom_labels, buckets=age_buckets)
        self._prom_frag_access_count = Histogram('vua_tier_fragment_access_count', 'Distribution of fragment access counts per tier', self._prom_labels, buckets=count_buckets)
        self._prom_frag_size = Histogram('vua_tier_fragment_size_bytes', 'Distribution of fragment sizes per tier', self._prom_labels, buckets=size_buckets)

        # Initialize gauges for configured capacities and initial usage
        for i, config in enumerate(self.tier_configs):
            tier_name = config.get('name', f'tier_{i}')
            labels = {'tier_name': tier_name}
            self._prom_capacity_count.labels(**labels).set(config.get('capacity_count', float('inf')))
            self._prom_capacity_bytes.labels(**labels).set(config.get('capacity_bytes', float('inf')))
            self._prom_usage_count.labels(**labels).set(0)
            self._prom_usage_bytes.labels(**labels).set(0)
            self._prom_advanced_active.labels(**labels).set(0)
            self._prom_avg_eviction_score.labels(**labels).set(0)

        self.policy = policy or LeCAR(policy_config)  # Use provided config if no custom policy
        self.metadata = {}
        self.tier_usage = {
            i: {'count': 0, 'bytes': 0}
            for i in range(len(backends))
        }

    def get(self, group_hash: str) -> Optional[tuple[bytes, bytes]]:
        self.op_count += 1
        for tier_idx, backend in enumerate(self.backends):
            result = backend.get(group_hash)
            if result is not None:
                size_bytes = None
                if group_hash in self.metadata:
                    size_bytes = self.metadata[group_hash].get('size_bytes', None)
                meta = self._update_metadata(group_hash, tier_idx, size_bytes or 0, access=True)
                self.metrics[tier_idx]['hits'] += 1
                self._prom_hits.labels(tier_name=self.tier_configs[tier_idx].get('name', f'tier_{tier_idx}')).inc()
                # Promotion logic
                if tier_idx > 0 and meta['access_count'] >= meta.get('promotion_threshold', 1):
                    if self._can_promote(group_hash, tier_idx, tier_idx - 1):
                        promo_tier_name = self.tier_configs[tier_idx - 1].get('name', f'tier_{tier_idx - 1}')
                        self._promote(group_hash, tier_idx, tier_idx - 1)
                        self.metrics[tier_idx]['promotions'] += 1
                        self._prom_promotions.labels(tier_name=promo_tier_name).inc() # Count promotion *into* the tier
                if self.op_count % self.adjustment_interval == 0:
                    self._feedback_adjust_thresholds()
                return result
            else:
                self.metrics[tier_idx]['misses'] += 1
                self._prom_misses.labels(tier_name=self.tier_configs[tier_idx].get('name', f'tier_{tier_idx}')).inc()
        if self.op_count % self.adjustment_interval == 0:
            self._feedback_adjust_thresholds()
        return None

    def put(self, group_hash: str, data: bytes, tokens: bytes) -> None:
        self.op_count += 1
        tier_idx = 0
        config = self.tier_configs[tier_idx]
        usage = self.tier_usage[tier_idx]
        size_bytes = len(data) + len(tokens)
        # Proactive demotion if above watermark
        self._proactive_demote(tier_idx)
        # If full, demote coldest until space
        while (
            (config.get('capacity_count') and usage['count'] >= config['capacity_count']) or
            (config.get('capacity_bytes') and usage['bytes'] + size_bytes > config['capacity_bytes'])
        ):
            victim = self._find_coldest_fragment(tier_idx)
            if victim is None:
                break
            self._demote(victim, tier_idx, tier_idx + 1)
            self.metrics[tier_idx]['demotions'] += 1
            self._prom_demotions.labels(tier_name=self.tier_configs[tier_idx].get('name', f'tier_{tier_idx}')).inc()
        self.backends[tier_idx].put(group_hash, data, tokens)
        usage['count'] += 1
        usage['bytes'] += size_bytes
        self._prom_usage_count.labels(tier_name=self.tier_configs[tier_idx].get('name', f'tier_{tier_idx}')).set(usage['count'])
        self._prom_usage_bytes.labels(tier_name=self.tier_configs[tier_idx].get('name', f'tier_{tier_idx}')).set(usage['bytes'])
        self._update_metadata(group_hash, tier_idx, size_bytes, access=False)
        if self.op_count % self.adjustment_interval == 0:
            self._feedback_adjust_thresholds()

    def _demote(self, group_hash, from_tier, to_tier):
        # Get metadata early, needed for policy.evict if evicting
        if group_hash not in self.metadata:
            self.logger.warning(f"Attempted to demote non-existent metadata for {group_hash} from tier {from_tier}")
            return
        meta = self.metadata[group_hash]

        # Get data from the source tier
        result = self.backends[from_tier].get(group_hash)
        if result is None: # Should not happen if metadata exists, but handle defensively
            self.logger.warning(f"Attempted to demote non-existent data for {group_hash} from tier {from_tier}. Cleaning up metadata.")
            # Clean up inconsistent metadata
            self._cleanup_metadata(group_hash, from_tier)
            return
        data, tokens = result
        size_bytes = meta.get('size_bytes', len(data) + len(tokens)) # Use actual size if meta is missing it

        # --- Eviction Logic --- 
        if to_tier >= len(self.backends):
            # Evict from system
            self.logger.debug(f"Evicting {group_hash} from tier {from_tier} (size: {size_bytes})")
            # Physically remove (or mark as deleted in backend if supported)
            # Assuming backends don't have explicit delete, we overwrite with empty
            # TODO: Add explicit delete to backend interface?
            try:
                 # Assuming put overwrites or backend handles empty data
                 self.backends[from_tier].put(group_hash, b"", b"")
            except Exception as e:
                 self.logger.error(f"Error clearing data for evicted {group_hash} in backend {from_tier}: {e}")
            
            # Update usage stats for the tier it was evicted from
            self._update_tier_usage(from_tier, -1, -size_bytes)

            # Create CacheEntry for the evicted item
            evicted_entry = CacheEntry(
                group_hash=group_hash,
                size_bytes=size_bytes,
                last_access_time=meta.get('last_access', time.time()),
                access_count=meta.get('access_count', 0),
                tier_idx=from_tier # Tier it was evicted from
            )
            
            # Call policy.evict if the policy implements it
            if hasattr(self.policy, 'evict'):
                try:
                    self.policy.evict(evicted_entry)
                except Exception as e:
                    self.logger.error(f"Error calling policy.evict for {group_hash}: {e}")

            # Remove metadata last
            del self.metadata[group_hash]

            # Update metrics
            self.metrics[from_tier]['evictions'] += 1
            tier_name = self.tier_configs[from_tier].get('name', f'tier_{from_tier}')
            self._prom_evictions.labels(tier_name=tier_name).inc()
            # Usage gauges are updated by _update_tier_usage
            return # Stop processing after eviction

        # --- Demotion to lower tier logic --- 
        # (Rest of the demotion logic remains the same)
        # Check if to_tier is full, recursively demote from to_tier first
        # ... (recursive demotion loop) ...
        # Move data to the target tier (to_tier)
        # ... (backend put calls) ...
        # Update usage stats for both tiers
        # ... (_update_tier_usage calls) ...
        # Update metadata
        # ... (meta update logic) ...
        # Update demotion metrics
        # ... (metrics and prom counter updates) ...

    def _update_tier_usage(self, tier_idx, count_delta, bytes_delta):
        """Helper to update tier usage stats and Prometheus gauges."""
        if tier_idx < 0 or tier_idx >= len(self.backends):
            return # Invalid tier index
            
        usage = self.tier_usage[tier_idx]
        usage['count'] += count_delta
        usage['bytes'] += bytes_delta
        # Clamp usage to avoid negative values due to potential inconsistencies
        usage['count'] = max(0, usage['count'])
        usage['bytes'] = max(0, usage['bytes'])
        
        # Update Prometheus Gauges
        tier_name = self.tier_configs[tier_idx].get('name', f'tier_{tier_idx}')
        labels = {'tier_name': tier_name}
        self._prom_usage_count.labels(**labels).set(usage['count'])
        self._prom_usage_bytes.labels(**labels).set(usage['bytes'])

    def _cleanup_metadata(self, group_hash, expected_tier):
         """Remove potentially inconsistent metadata."""
         if group_hash in self.metadata:
             meta = self.metadata[group_hash]
             if meta.get('tier') == expected_tier:
                 size_bytes = meta.get('size_bytes', 0)
                 self._update_tier_usage(expected_tier, -1, -size_bytes)
                 del self.metadata[group_hash]
                 self.logger.info(f"Cleaned up inconsistent metadata for {group_hash} in tier {expected_tier}")
             else:
                 self.logger.warning(f"Metadata for {group_hash} found in unexpected tier {meta.get('tier')} while cleaning tier {expected_tier}")
         else:
             self.logger.warning(f"No metadata found for {group_hash} during cleanup attempt.")

    def _promote(self, group_hash, from_tier, to_tier):
        meta = self.metadata[group_hash]
        result = self.backends[from_tier].get(group_hash)
        if result is None: # Should not happen, handle defensively
             self.logger.warning(f"Attempted to promote non-existent {group_hash} from tier {from_tier}")
             if group_hash in self.metadata:
                 del self.metadata[group_hash]
             return
        data, tokens = result
        size_bytes = meta['size_bytes']
        if to_tier >= len(self.backends):
            # Evict from system
            self.backends[from_tier].put(group_hash, b"", b"")
            self.tier_usage[from_tier]['count'] -= 1
            self.tier_usage[from_tier]['bytes'] -= size_bytes
            del self.metadata[group_hash]
            self.metrics[from_tier]['evictions'] += 1
            self._prom_evictions.labels(tier_name=self.tier_configs[from_tier].get('name', f'tier_{from_tier}')).inc()
            # Update Prometheus Gauges for usage
            self._prom_usage_count.labels(tier_name=self.tier_configs[from_tier].get('name', f'tier_{from_tier}')).set(self.tier_usage[from_tier]['count'])
            self._prom_usage_bytes.labels(tier_name=self.tier_configs[from_tier].get('name', f'tier_{from_tier}')).set(self.tier_usage[from_tier]['bytes'])
            return
        # Check if to_tier is full, recursively promote
        config = self.tier_configs[to_tier]
        usage = self.tier_usage[to_tier]
        while (
            (config.get('capacity_count') and usage['count'] >= config['capacity_count']) or
            (config.get('capacity_bytes') and usage['bytes'] + size_bytes > config['capacity_bytes'])
        ):
            victim = self._find_coldest_fragment(to_tier)
            if victim is None:
                break
            self._promote(victim, to_tier, to_tier + 1)
            self.metrics[to_tier]['promotions'] += 1
            self._prom_promotions.labels(tier_name=self.tier_configs[to_tier].get('name', f'tier_{to_tier}')).inc()
        # Move to new tier
        self.backends[to_tier].put(group_hash, data, tokens)
        self.backends[from_tier].put(group_hash, b"", b"")
        self.tier_usage[from_tier]['count'] -= 1
        self.tier_usage[from_tier]['bytes'] -= size_bytes
        self.tier_usage[to_tier]['count'] += 1
        self.tier_usage[to_tier]['bytes'] += size_bytes
        # Update Prometheus Gauges for usage
        self._prom_usage_count.labels(tier_name=self.tier_configs[from_tier].get('name', f'tier_{from_tier}')).set(self.tier_usage[from_tier]['count'])
        self._prom_usage_bytes.labels(tier_name=self.tier_configs[from_tier].get('name', f'tier_{from_tier}')).set(self.tier_usage[from_tier]['bytes'])
        self._prom_usage_count.labels(tier_name=self.tier_configs[to_tier].get('name', f'tier_{to_tier}')).set(self.tier_usage[to_tier]['count'])
        self._prom_usage_bytes.labels(tier_name=self.tier_configs[to_tier].get('name', f'tier_{to_tier}')).set(self.tier_usage[to_tier]['bytes'])
        meta['tier'] = to_tier
        meta['access_count'] = 0
        meta['last_access'] = time.time()
        self.metrics[to_tier]['promotions'] += 1

    def _feedback_adjust_thresholds(self):
        """
        Adjust policy parameters based on recent metrics and usage.
        Tunes parameters like exploration rate and base score thresholds in PolicyConfig.
        """
        if not hasattr(self.policy, 'config') or not hasattr(self.policy, 'update_config'):
             self.logger.warning("Policy does not support dynamic configuration adjustment.")
             # Reset metrics anyway before returning
             for tier_idx in self.metrics:
                  self.metrics[tier_idx].update({'hits': 0, 'misses': 0, 'promotions': 0, 'demotions': 0, 'evictions': 0, 'last_adjustment': time.time()})
             return

        config_changed = False
        current_policy_config = self.policy.config
        original_config_values = {
            'exploration_rate': current_policy_config.exploration_rate,
            'learning_rate': current_policy_config.learning_rate, # Placeholder for potential tuning
            'base_promotion_score_threshold': current_policy_config.base_promotion_score_threshold,
            'base_demotion_score_threshold': current_policy_config.base_demotion_score_threshold
        }
        
        # --- Global Adjustments (can be based on overall cache performance) ---
        # Example: Adjust exploration rate based on overall hit rate (optional)
        total_hits = sum(m['hits'] for m in self.metrics.values())
        total_misses = sum(m['misses'] for m in self.metrics.values())
        total_accesses_overall = total_hits + total_misses
        overall_hit_rate = total_hits / max(1, total_accesses_overall) if total_accesses_overall > 0 else 0
        
        new_exploration_rate = current_policy_config.exploration_rate
        if overall_hit_rate < 0.5 and total_accesses_overall > 50: # Thresholds tunable
            new_exploration_rate = min(0.5, current_policy_config.exploration_rate + 0.01)
        elif overall_hit_rate > 0.95 and total_accesses_overall > 50:
            new_exploration_rate = max(0.01, current_policy_config.exploration_rate - 0.01)
            
        if abs(new_exploration_rate - original_config_values['exploration_rate']) > 0.001:
            current_policy_config.exploration_rate = new_exploration_rate
            config_changed = True
        
        # --- Per-Tier Adjustments (example for score thresholds) ---
        for tier_idx, metrics in self.metrics.items():
            config = self.tier_configs[tier_idx]
            usage = self.tier_usage[tier_idx]
            tier_name = config.get('name', f'tier_{tier_idx}')
            
            # Calculate relevant tier-specific metrics
            total_accesses_tier = metrics['hits'] + metrics['misses']
            hit_rate_tier = metrics['hits'] / max(1, total_accesses_tier) if total_accesses_tier > 0 else 0
            capacity_count = config.get('capacity_count', float('inf'))
            usage_ratio = usage['count'] / max(1, capacity_count) if capacity_count != float('inf') else 0
            # Consider demotions/evictions from this tier
            pressure_metric = metrics['demotions'] + metrics['evictions'] 
            # Get metrics for tier above (if it exists) to check demotions *into* this tier
            demotions_into_tier = self.metrics[tier_idx-1]['demotions'] if tier_idx > 0 else 0

            # --- Tune Promotion Score Threshold ---
            # Goal: Promote efficiently. Lower threshold if low hit rate suggests items stuck.
            # Raise if high hit rate + high demotions from target tier suggest churn.
            if hit_rate_tier < 0.4 and total_accesses_tier > 10: # Low hit rate in this lower tier
                # Make promotion easier
                current_policy_config.base_promotion_score_threshold = max(0.1, current_policy_config.base_promotion_score_threshold - 0.01)
                config_changed = True
            # Check demotions from the tier above (tier_idx - 1)
            if tier_idx > 0:
                demotions_from_above = self.metrics[tier_idx-1]['demotions'] + self.metrics[tier_idx-1]['evictions']
                hit_rate_above = self.metrics[tier_idx-1]['hits'] / max(1, self.metrics[tier_idx-1]['hits'] + self.metrics[tier_idx-1]['misses'])
                if hit_rate_above > 0.9 and demotions_from_above > 5: # High hit rate but still demoting often from target
                     # Make promotion slightly harder to reduce churn
                     current_policy_config.base_promotion_score_threshold = min(0.9, current_policy_config.base_promotion_score_threshold + 0.01)
                     config_changed = True

            # --- Tune Demotion Score Threshold ---
            # Goal: Demote efficiently under pressure. Lower threshold if high usage/pressure.
            # Raise if low usage suggests demoting too eagerly.
            if usage_ratio > 0.95 and pressure_metric > 5: # High pressure
                # Make demotion easier
                current_policy_config.base_demotion_score_threshold = max(0.05, current_policy_config.base_demotion_score_threshold - 0.01)
                config_changed = True
            elif usage_ratio < 0.3 and hit_rate_tier < 0.2 and total_accesses_tier > 10: # Low usage and low hit rate
                # Make demotion harder
                current_policy_config.base_demotion_score_threshold = min(0.8, current_policy_config.base_demotion_score_threshold + 0.01)
                config_changed = True

            # Clamp thresholds to ensure demotion < promotion (with a buffer)
            buffer = 0.1 # Ensure demotion threshold is at least this much lower than promotion
            current_policy_config.base_demotion_score_threshold = min(
                current_policy_config.base_demotion_score_threshold, 
                current_policy_config.base_promotion_score_threshold - buffer
            )
            current_policy_config.base_demotion_score_threshold = max(0.0, current_policy_config.base_demotion_score_threshold) # Min 0
            current_policy_config.base_promotion_score_threshold = max(
                current_policy_config.base_promotion_score_threshold,
                current_policy_config.base_demotion_score_threshold + buffer
            )
            current_policy_config.base_promotion_score_threshold = min(1.0, current_policy_config.base_promotion_score_threshold) # Max 1

            # --- Reset Tier Metrics --- 
            metrics['hits'] = 0
            metrics['misses'] = 0
            metrics['promotions'] = 0
            metrics['demotions'] = 0
            metrics['evictions'] = 0
            metrics['last_adjustment'] = time.time()

        # If any policy parameter changed, log and update the policy object
        if config_changed:
            self.logger.info(
                 f"[FeedbackAdjust] Policy params updated: ER={current_policy_config.exploration_rate:.3f}"
                 f" LR={current_policy_config.learning_rate:.3f}"
                 f" PT={current_policy_config.base_promotion_score_threshold:.2f}"
                 f" DT={current_policy_config.base_demotion_score_threshold:.2f}"
                 f" (Original: ER={original_config_values['exploration_rate']:.3f}"
                 f" PT={original_config_values['base_promotion_score_threshold']:.2f}"
                 f" DT={original_config_values['base_demotion_score_threshold']:.2f})"
            )
            try:
                self.policy.update_config(current_policy_config)
            except Exception as e:
                self.logger.error(f"Error calling policy.update_config: {e}")

    def export_metrics(self):
        """
        Ensure Prometheus Gauges and Histograms/Summaries are up-to-date.
        Calculates aggregated stats from self.metadata.
        Counters are updated directly in relevant methods.
        """
        now = time.time()
        # Temporary storage for per-tier aggregates
        tier_aggregates = {
            i: {'advanced_count': 0, 'total_score': 0.0, 'fragment_count': 0}
            for i in range(len(self.backends))
        }
        # Clear histograms before recalculating - Prometheus client handles aggregation across scrapes
        # Note: This might not be the most efficient way for histograms if updates are frequent.
        # Consider updating histograms more incrementally if performance is an issue.
        for i, config in enumerate(self.tier_configs):
             tier_name = config.get('name', f'tier_{i}')
             labels = {'tier_name': tier_name}
             self._prom_frag_age.labels(**labels)._metric.reset()
             self._prom_frag_access_count.labels(**labels)._metric.reset()
             self._prom_frag_size.labels(**labels)._metric.reset()

        # Iterate through all fragments to populate histograms and aggregates
        for group_hash, meta in self.metadata.items():
            tier_idx = meta['tier']
            tier_name = self.tier_configs[tier_idx].get('name', f'tier_{tier_idx}')
            labels = {'tier_name': tier_name}

            # Observe distributions
            age = now - meta.get('last_access', meta.get('insert_time', 0))
            self._prom_frag_age.labels(**labels).observe(age)
            self._prom_frag_access_count.labels(**labels).observe(meta.get('access_count', 0))
            self._prom_frag_size.labels(**labels).observe(meta.get('size_bytes', 0))

            # Update aggregates
            aggregates = tier_aggregates[tier_idx]
            aggregates['fragment_count'] += 1
            if 'history' in meta: # Check if advanced metadata is active
                aggregates['advanced_count'] += 1
            aggregates['total_score'] += self._calculate_eviction_score(meta)

        # Update Gauges with final aggregates
        for tier_idx, config in enumerate(self.tier_configs):
             tier_name = config.get('name', f'tier_{tier_idx}')
             usage = self.tier_usage[tier_idx]
             aggregates = tier_aggregates[tier_idx]
             labels = {'tier_name': tier_name}

             self._prom_usage_count.labels(**labels).set(usage['count'])
             self._prom_usage_bytes.labels(**labels).set(usage['bytes'])
             self._prom_advanced_active.labels(**labels).set(aggregates['advanced_count'])
             avg_score = aggregates['total_score'] / max(1, aggregates['fragment_count'])
             self._prom_avg_eviction_score.labels(**labels).set(avg_score)

        self.logger.debug("Prometheus metrics gauges and histograms updated.")
        # The actual export happens via an HTTP server started elsewhere

    # Method to start the HTTP server (optional, can be called by application)
    def start_prometheus_server(self, port=8000):
        if PROMETHEUS_AVAILABLE:
            try:
                start_http_server(port)
                self.logger.info(f"Prometheus metrics server started on port {port}")
            except Exception as e:
                self.logger.error(f"Failed to start Prometheus server on port {port}: {e}")
        else:
            self.logger.warning("Cannot start Prometheus server: prometheus-client library not available.")

    def _update_metadata(self, group_hash, tier_idx, size_bytes, access):
        """Update internal metadata and inform the policy about the access."""
        # Ensure size_bytes is not None, default to 0 if necessary
        size_bytes = size_bytes if size_bytes is not None else 0
        
        meta = self.metadata.get(group_hash, {})
        now = time.time()
        # Update metadata fields
        meta.update({
            'tier': tier_idx,
            'size_bytes': size_bytes,
            'last_access': now,
            'access_count': meta.get('access_count', 0) + (1 if access else 0),
            # Initialize insert_time if it doesn't exist
            'insert_time': meta.get('insert_time', now) 
        })
        self.metadata[group_hash] = meta
        
        # Create CacheEntry for policy update
        entry = CacheEntry(
            group_hash=group_hash,
            size_bytes=size_bytes,
            last_access_time=meta['last_access'],
            access_count=meta['access_count'],
            tier_idx=tier_idx
        )
        
        # Update policy state
        try:
             self.policy.update(entry, access) 
        except Exception as e:
             self.logger.error(f"Error calling policy.update for {group_hash}: {e}")
             
        return meta # Return updated metadata

    def _find_coldest_fragment(self, tier_idx):
        """Find fragment to evict from the specified tier using the policy."""
        entries_in_tier = []
        for group_hash, meta in self.metadata.items():
            if meta.get('tier') == tier_idx:
                # Ensure essential fields are present for CacheEntry
                entry = CacheEntry(
                    group_hash=group_hash,
                    size_bytes=meta.get('size_bytes', 0),
                    last_access_time=meta.get('last_access', meta.get('insert_time', 0)),
                    access_count=meta.get('access_count', 0),
                    tier_idx=tier_idx
                )
                entries_in_tier.append(entry)
        
        if not entries_in_tier:
             self.logger.debug(f"_find_coldest_fragment called for empty tier {tier_idx}. No victim found.")
             return None # No fragments in this tier
             
        # Ask the policy to select a victim from the entries in this tier
        try:
            victim_hash = self.policy.select_victim(entries_in_tier, tier_idx)
        except Exception as e:
            self.logger.error(f"Error calling policy.select_victim for tier {tier_idx}: {e}")
            # Fallback: if policy fails, evict the first entry found as a safety measure
            victim_hash = entries_in_tier[0].group_hash 
            
        if victim_hash is None:
            self.logger.warning(f"Policy returned None for victim selection in tier {tier_idx}. Check policy logic.")
            # Fallback if policy returns None unexpectedly
            if entries_in_tier: 
                 victim_hash = entries_in_tier[0].group_hash
                 
        return victim_hash

    def _proactive_demote(self, tier_idx):
        # Implementation of _proactive_demote method (currently placeholder)
        # TODO: Implement proactive demotion based on watermark and policy
        pass

    def _can_promote(self, group_hash, from_tier, to_tier):
        """Check if an entry can be promoted based on policy decision."""
        if group_hash not in self.metadata:
            return False
            
        meta = self.metadata[group_hash]
        entry = CacheEntry(
            group_hash=group_hash,
            size_bytes=meta.get('size_bytes', 0),
            last_access_time=meta.get('last_access', meta.get('insert_time', 0)),
            access_count=meta.get('access_count', 0),
            tier_idx=from_tier
        )
        
        try:
             return self.policy.should_promote(entry, from_tier, to_tier)
        except Exception as e:
             self.logger.error(f"Error calling policy.should_promote for {group_hash}: {e}")
             return False # Default to no promotion on error

    def _calculate_eviction_score(self, meta):
        """Helper to calculate eviction score based on current policy."""
        # Example simple score - replace with actual logic if needed
        return (
            (time.time() - meta.get('last_access', meta.get('insert_time', 0))) + # LRU component
            10.0 / (meta.get('access_count', 0) + 1) + # LFU component (inverted)
            0.001 * meta.get('size_bytes', 0) # Size component
        ) 

    def exists(self, group_hash: str) -> bool:
        """Check if a fragment exists in any tier."""
        for tier_idx, backend in enumerate(self.backends):
            if backend.exists(group_hash):
                self.logger.debug(f"Exists check: Found {group_hash} in tier {tier_idx}")
                # Optionally update metadata if found in a lower tier?
                # For now, just return True
                return True
        self.logger.debug(f"Exists check: {group_hash} not found in any tier.")
        return False 