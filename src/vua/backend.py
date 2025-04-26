from abc import ABC, abstractmethod
from typing import Any, Optional
import os
import logging
import time

# Restored Class Definitions
class StorageBackend(ABC):
    """
    Abstract base class for VUA storage backends.
    Defines the interface for storing and retrieving cache fragments.
    """
    @abstractmethod
    def put(self, group_hash: str, data: Any, tokens: Any) -> None:
        """
        Store a cache fragment identified by group_hash.
        Args:
            group_hash: Unique identifier for the group (hash string).
            data: Serialized cache data (e.g., tensor bytes).
            tokens: Serialized tokens data.
        """
        pass

    @abstractmethod
    def get(self, group_hash: str) -> Optional[tuple]:
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

    def put(self, group_hash: str, data: Any, tokens: Any) -> None:
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

    def get(self, group_hash: str) -> Optional[tuple]:
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

    def put(self, group_hash: str, data: Any, tokens: Any) -> None:
        self._store[group_hash] = (data, tokens)

    def get(self, group_hash: str) -> Optional[tuple]:
        return self._store.get(group_hash)

    def exists(self, group_hash: str) -> bool:
        return group_hash in self._store

class PMDKBackend(StorageBackend):
    """
    Simulated PMDK backend for VUA. Uses a dedicated directory to mimic a PMDK pool.
    Each group is stored as files in root_path/pmdk_pool/group_hash/.
    This is a placeholder for future real PMDK integration.
    """
    def __init__(self, root_path: str):
        self.pool_path = os.path.join(root_path, "pmdk_pool")
        os.makedirs(self.pool_path, exist_ok=True)

    def put(self, group_hash: str, data: Any, tokens: Any) -> None:
        """
        Simulate storing data and tokens in a PMDK pool by writing to files in a dedicated directory.
        TODO: Replace with real PMDK object store logic when available.
        """
        group_dir = os.path.join(self.pool_path, group_hash)
        os.makedirs(group_dir, exist_ok=True)
        data_path = os.path.join(group_dir, "_data.safetensors")
        tokens_path = os.path.join(group_dir, "_tokens.safetensors")
        with open(data_path, "wb") as f:
            f.write(data)
        with open(tokens_path, "wb") as f:
            f.write(tokens)

    def get(self, group_hash: str) -> Optional[tuple]:
        """
        Simulate retrieving data and tokens from a PMDK pool by reading from files.
        TODO: Replace with real PMDK object store logic when available.
        """
        group_dir = os.path.join(self.pool_path, group_hash)
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
        Simulate checking for a group in a PMDK pool by checking for files in the directory.
        TODO: Replace with real PMDK object store logic when available.
        """
        group_dir = os.path.join(self.pool_path, group_hash)
        data_path = os.path.join(group_dir, "_data.safetensors")
        tokens_path = os.path.join(group_dir, "_tokens.safetensors")
        return os.path.isdir(group_dir) and os.path.isfile(data_path) and os.path.isfile(tokens_path)

# TieredBackend follows

class TieredBackend(StorageBackend):
    def __init__(self, backends, tier_configs, advanced_watermark=0.9):
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
        self.adjustment_interval = 1000  # Number of ops between adjustments
        self.op_count = 0
        self.logger = logging.getLogger("TieredBackend")

    def get(self, group_hash: str) -> Optional[tuple]:
        self.op_count += 1
        for tier_idx, backend in enumerate(self.backends):
            result = backend.get(group_hash)
            if result is not None:
                size_bytes = None
                if group_hash in self.metadata:
                    size_bytes = self.metadata[group_hash].get('size_bytes', None)
                meta = self._update_metadata(group_hash, tier_idx, size_bytes or 0, access=True)
                self.metrics[tier_idx]['hits'] += 1
                # Promotion logic
                if tier_idx > 0 and meta['access_count'] >= meta.get('promotion_threshold', 1):
                    if self._can_promote(group_hash, tier_idx, tier_idx - 1):
                        self._promote(group_hash, tier_idx, tier_idx - 1)
                        self.metrics[tier_idx]['promotions'] += 1
                if self.op_count % self.adjustment_interval == 0:
                    self._feedback_adjust_thresholds()
                return result
            else:
                self.metrics[tier_idx]['misses'] += 1
        if self.op_count % self.adjustment_interval == 0:
            self._feedback_adjust_thresholds()
        return None

    def put(self, group_hash: str, data: Any, tokens: Any) -> None:
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
        self.backends[tier_idx].put(group_hash, data, tokens)
        usage['count'] += 1
        usage['bytes'] += size_bytes
        self._update_metadata(group_hash, tier_idx, size_bytes, access=False)
        if self.op_count % self.adjustment_interval == 0:
            self._feedback_adjust_thresholds()

    def _demote(self, group_hash, from_tier, to_tier):
        meta = self.metadata[group_hash]
        data, tokens = self.backends[from_tier].get(group_hash)
        size_bytes = meta['size_bytes']
        if to_tier >= len(self.backends):
            # Evict from system
            self.backends[from_tier].put(group_hash, b"", b"")
            self.tier_usage[from_tier]['count'] -= 1
            self.tier_usage[from_tier]['bytes'] -= size_bytes
            del self.metadata[group_hash]
            self.metrics[from_tier]['evictions'] += 1
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
        # Move to new tier
        self.backends[to_tier].put(group_hash, data, tokens)
        self.backends[from_tier].put(group_hash, b"", b"")
        self.tier_usage[from_tier]['count'] -= 1
        self.tier_usage[from_tier]['bytes'] -= size_bytes
        self.tier_usage[to_tier]['count'] += 1
        self.tier_usage[to_tier]['bytes'] += size_bytes
        meta['tier'] = to_tier
        meta['access_count'] = 0
        meta['last_access'] = time.time()
        self.metrics[to_tier]['demotions'] += 1

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
            if new_promo != old_promo:
                self.logger.info(f"[Tier {tier_idx}] Adjusted promotion_threshold: {old_promo} -> {new_promo} (hit_rate={hit_rate:.2f})")
            # Reset metrics or use rolling window
            metrics['hits'] = 0
            metrics['misses'] = 0
            metrics['promotions'] = 0
            metrics['demotions'] = 0
            metrics['evictions'] = 0
        # Export metrics for Prometheus/Grafana
        self.export_metrics()

    def export_metrics(self):
        """
        Export current metrics for Prometheus/Grafana integration.
        This is a stub; in production, use a library like prometheus_client to expose metrics.
        """
        # Example: print metrics (replace with Prometheus push or HTTP endpoint)
        for tier_idx, metrics in self.metrics.items():
            self.logger.info(f"[Tier {tier_idx}] Metrics: {metrics}")
        # TODO: Integrate with prometheus_client or another metrics exporter 