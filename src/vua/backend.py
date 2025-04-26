from abc import ABC, abstractmethod
from typing import Any, Optional
import os
import logging
import time

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

class TieredBackend(StorageBackend):
    def __init__(self, backends, tier_configs, advanced_watermark=0.9, adjustment_interval=1000):
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
        self._prom_promotions = Counter('vua_tier_promotions_total', 'Total promotions from a tier', self._prom_labels)
        self._prom_demotions = Counter('vua_tier_demotions_total', 'Total demotions from a tier', self._prom_labels)
        self._prom_evictions = Counter('vua_tier_evictions_total', 'Total evictions from a tier', self._prom_labels)
        self._prom_usage_count = Gauge('vua_tier_usage_fragments', 'Current fragment count in a tier', self._prom_labels)
        self._prom_usage_bytes = Gauge('vua_tier_usage_bytes', 'Current byte usage in a tier', self._prom_labels)
        self._prom_capacity_count = Gauge('vua_tier_capacity_fragments', 'Fragment capacity of a tier', self._prom_labels)
        self._prom_capacity_bytes = Gauge('vua_tier_capacity_bytes', 'Byte capacity of a tier', self._prom_labels)
        self._prom_promo_threshold = Gauge('vua_tier_promotion_threshold', 'Current promotion threshold for a tier', self._prom_labels)
        self._prom_demotion_threshold = Gauge('vua_tier_demotion_threshold', 'Current demotion threshold for a tier', self._prom_labels)

        # --- New Aggregated/Distribution Metrics ---
        # Using Histogram for distributions - define buckets appropriately
        age_buckets = (1, 5, 15, 60, 300, 1800, 3600, float('inf')) # seconds
        count_buckets = (1, 2, 5, 10, 25, 100, float('inf'))
        size_buckets = (1024, 4096, 16384, 65536, 262144, 1048576, float('inf')) # bytes

        self._prom_frag_age = Histogram('vua_tier_fragment_age_seconds', 'Distribution of fragment ages per tier (last access)', self._prom_labels, buckets=age_buckets)
        self._prom_frag_access_count = Histogram('vua_tier_fragment_access_count', 'Distribution of fragment access counts per tier', self._prom_labels, buckets=count_buckets)
        self._prom_frag_size = Histogram('vua_tier_fragment_size_bytes', 'Distribution of fragment sizes per tier', self._prom_labels, buckets=size_buckets)
        self._prom_advanced_active = Gauge('vua_tier_advanced_metadata_active_fragments', 'Number of fragments using advanced metadata per tier', self._prom_labels)
        self._prom_avg_eviction_score = Gauge('vua_tier_avg_eviction_score', 'Average eviction score per tier', self._prom_labels)

        # Initialize gauges for configured capacities and initial thresholds
        for i, config in enumerate(self.tier_configs):
            tier_name = config.get('name', f'tier_{i}')
            labels = {'tier_name': tier_name}
            self._prom_capacity_count.labels(**labels).set(config.get('capacity_count', float('inf')))
            self._prom_capacity_bytes.labels(**labels).set(config.get('capacity_bytes', float('inf')))
            self._prom_promo_threshold.labels(**labels).set(config.get('promotion_threshold', 1))
            self._prom_demotion_threshold.labels(**labels).set(config.get('demotion_threshold', 1))
            self._prom_usage_count.labels(**labels).set(0)
            self._prom_usage_bytes.labels(**labels).set(0)
            self._prom_advanced_active.labels(**labels).set(0)
            self._prom_avg_eviction_score.labels(**labels).set(0)

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
        meta = self.metadata[group_hash]
        result = self.backends[from_tier].get(group_hash)
        if result is None: # Should not happen if metadata exists, but handle defensively
            self.logger.warning(f"Attempted to demote non-existent {group_hash} from tier {from_tier}")
            # Clean up potentially inconsistent metadata if needed
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
        # Check if to_tier is full, recursively demote
        config = self.tier_configs[to_tier]
        usage = self.tier_usage[to_tier]
        while (
            (config.get('capacity_count') and usage['count'] >= config['capacity_count']) or
            (config.get('capacity_bytes') and usage['bytes'] + size_bytes > config['capacity_bytes'])
        ):
            victim = self._find_coldest_fragment(to_tier)
            if victim is None:
                break
            self._demote(victim, to_tier, to_tier + 1)
            self.metrics[to_tier]['demotions'] += 1
            self._prom_demotions.labels(tier_name=self.tier_configs[to_tier].get('name', f'tier_{to_tier}')).inc()
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
        self.metrics[to_tier]['demotions'] += 1

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
        Adjust promotion/demotion thresholds and watermarks based on recent metrics.
        Log adjustments and export metrics for Prometheus/Grafana.
        """
        for tier_idx, metrics in self.metrics.items():
            hit_rate = metrics['hits'] / max(1, (metrics['hits'] + metrics['misses']))
            old_promo = self.tier_configs[tier_idx]['promotion_threshold']
            # Example heuristic:
            if hit_rate < 0.7:
                # Lower promotion threshold (promote more aggressively)
                self.tier_configs[tier_idx]['promotion_threshold'] = max(
                    1, self.tier_configs[tier_idx]['promotion_threshold'] - 1
                )
            elif hit_rate > 0.95:
                # Raise promotion threshold (be more selective)
                self.tier_configs[tier_idx]['promotion_threshold'] += 1
            # Similar logic for demotion_threshold and watermark can be added here
            new_promo = self.tier_configs[tier_idx]['promotion_threshold']
            new_demotion = self.tier_configs[tier_idx].get('demotion_threshold', 1) # Assuming we add this
            if new_promo != old_promo:
                self.logger.info(f"[Tier {tier_idx}] Adjusted promotion_threshold: {old_promo} -> {new_promo} (hit_rate={hit_rate:.2f})")
            # Update Prometheus Gauges for thresholds
            tier_name = self.tier_configs[tier_idx].get('name', f'tier_{tier_idx}')
            self._prom_promo_threshold.labels(tier_name=tier_name).set(new_promo)
            self._prom_demotion_threshold.labels(tier_name=tier_name).set(new_demotion)
            # Reset metrics or use rolling window
            metrics['hits'] = 0
            metrics['misses'] = 0
            metrics['promotions'] = 0
            metrics['demotions'] = 0
            metrics['evictions'] = 0
        # Export metrics for Prometheus/Grafana
        self.export_metrics() # This now mainly updates gauges

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
            self._prom_promo_threshold.labels(**labels).set(config.get('promotion_threshold', 1))
            self._prom_demotion_threshold.labels(**labels).set(config.get('demotion_threshold', 1))
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
        # Implementation of _update_metadata method
        pass

    def _find_coldest_fragment(self, tier_idx):
        # Implementation of _find_coldest_fragment method
        pass

    def _proactive_demote(self, tier_idx):
        # Implementation of _proactive_demote method
        pass

    def _can_promote(self, group_hash, from_tier, to_tier):
        # Implementation of _can_promote method
        pass

    def _promote(self, group_hash, from_tier, to_tier):
        # Implementation of _promote method
        pass

    def _demote(self, group_hash, from_tier, to_tier):
        # Implementation of _demote method
        pass

    def _feedback_adjust_thresholds(self):
        # Implementation of _feedback_adjust_thresholds method
        pass

    def export_metrics(self):
        # Implementation of export_metrics method
        pass

    def start_prometheus_server(self, port):
        # Implementation of start_prometheus_server method
        pass

    def _calculate_eviction_score(self, meta):
        """Helper to calculate eviction score based on current policy."""
        # Example simple score - replace with actual logic if needed
        return (
            (time.time() - meta.get('last_access', meta.get('insert_time', 0))) + # LRU component
            10.0 / (meta.get('access_count', 0) + 1) + # LFU component (inverted)
            0.001 * meta.get('size_bytes', 0) # Size component
        ) 