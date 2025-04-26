from abc import ABC, abstractmethod
from typing import Any, Optional
import os
import time

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

class TieredBackend(StorageBackend):
    """
    Storage backend that manages multiple tiers (e.g., GPU, DRAM, PMEM, Storage).
    Data is placed in the highest (fastest) tier on put, and promoted/demoted between tiers on get/evict.
    Backends are provided in priority order (fastest to slowest).
    Tracks both fragment count and bytes per tier, and uses a metadata table for access tracking and dynamic thresholds.
    """
    def __init__(self, backends, tier_configs, advanced_watermark=0.9):
        """
        Args:
            backends: List of StorageBackend instances, ordered from fastest to slowest.
            tier_configs: List of dicts with keys 'name', 'capacity_count', 'capacity_bytes', 'watermark', etc.
            advanced_watermark: Fraction of tier usage at which to activate advanced metadata/logic (e.g., 0.9 for 90%).
        """
        self.backends = backends
        self.tier_configs = tier_configs
        self.advanced_watermark = advanced_watermark
        # Metadata table: group_hash -> dict with tier, access_count, last_access, size_bytes, insert_time, ...
        self.metadata = {}
        # Per-tier usage tracking: {tier_idx: {'count': int, 'bytes': int}}
        self.tier_usage = {i: {'count': 0, 'bytes': 0} for i in range(len(backends))}
        # TODO: Add dynamic threshold state, e.g., recent hit/miss rates

    def _should_activate_advanced(self, tier_idx):
        """
        Return True if the tier is above the advanced watermark (by count or bytes).
        """
        config = self.tier_configs[tier_idx]
        usage = self.tier_usage[tier_idx]
        count_full = config.get('capacity_count') and usage['count'] >= config['capacity_count'] * self.advanced_watermark
        bytes_full = config.get('capacity_bytes') and usage['bytes'] >= config['capacity_bytes'] * self.advanced_watermark
        return bool(count_full or bytes_full)

    def _update_metadata(self, group_hash, tier_idx, size_bytes, access=False):
        now = time.time()
        meta = self.metadata.setdefault(group_hash, {})
        # Always-present fields
        if 'insert_time' not in meta:
            meta['insert_time'] = now
        meta['tier'] = tier_idx
        meta['size_bytes'] = size_bytes
        if access:
            meta['access_count'] = meta.get('access_count', 0) + 1
            meta['last_access'] = now
        else:
            meta.setdefault('access_count', 0)
            meta.setdefault('last_access', now)
        # Advanced fields (activate if tier is above watermark)
        if self._should_activate_advanced(tier_idx):
            # History of recent accesses
            if access:
                meta.setdefault('history', []).append(now)
                # Optionally limit history length
                if len(meta['history']) > 100:
                    meta['history'] = meta['history'][-100:]
            else:
                meta.setdefault('history', [])
            # Per-tier thresholds (can be overridden per-fragment)
            meta.setdefault('promotion_threshold', self.tier_configs[tier_idx].get('promotion_threshold', 1))
            meta.setdefault('demotion_threshold', self.tier_configs[tier_idx].get('demotion_threshold', 1))
            # Latency estimates (stub)
            meta.setdefault('latency_estimates', {})
            # Eviction score (stub, to be computed by policy)
            meta.setdefault('eviction_score', 0.0)
            # Snoop filter (stub, info about adjacent tiers)
            meta.setdefault('snoop_filter', self._snoop_filter(tier_idx))
        return meta

    def _snoop_filter(self, tier_idx):
        """
        Return info about adjacent tiers (capacity, usage, readiness).
        """
        info = {}
        if tier_idx > 0:
            info['above'] = {
                'usage': self.tier_usage[tier_idx - 1],
                'config': self.tier_configs[tier_idx - 1],
            }
        if tier_idx < len(self.backends) - 1:
            info['below'] = {
                'usage': self.tier_usage[tier_idx + 1],
                'config': self.tier_configs[tier_idx + 1],
            }
        return info

    def _find_coldest_fragment(self, tier_idx):
        """
        Find the coldest (least recently/frequently used) fragment in the given tier.
        Uses a composite eviction score (LRU/LFU hybrid).
        """
        candidates = [
            (gh, meta) for gh, meta in self.metadata.items() if meta['tier'] == tier_idx
        ]
        if not candidates:
            return None
        # Score: lower is colder (evict first)
        def score(meta):
            # LRU: older last_access, LFU: lower access_count, larger size
            return (
                meta.get('eviction_score', 0.0) +
                (time.time() - meta['last_access']) +
                10.0 / (meta['access_count'] + 1) +
                0.001 * meta['size_bytes']
            )
        coldest = min(candidates, key=lambda x: score(x[1]))
        return coldest[0]

    def _can_promote(self, group_hash, from_tier, to_tier):
        """
        Check if promotion is possible (capacity, snoop filter, etc.).
        """
        config = self.tier_configs[to_tier]
        usage = self.tier_usage[to_tier]
        if config.get('capacity_count') and usage['count'] >= config['capacity_count']:
            return False
        if config.get('capacity_bytes') and usage['bytes'] >= config['capacity_bytes']:
            return False
        # Optionally, check snoop filter for readiness
        return True

    def _promote(self, group_hash, from_tier, to_tier):
        """
        Move fragment from from_tier to to_tier, evicting from to_tier if needed.
        """
        meta = self.metadata[group_hash]
        data, tokens = self.backends[from_tier].get(group_hash)
        # Check if to_tier is full, evict if needed
        config = self.tier_configs[to_tier]
        usage = self.tier_usage[to_tier]
        size_bytes = meta['size_bytes']
        while (
            (config.get('capacity_count') and usage['count'] >= config['capacity_count']) or
            (config.get('capacity_bytes') and usage['bytes'] + size_bytes > config['capacity_bytes'])
        ):
            victim = self._find_coldest_fragment(to_tier)
            if victim is None:
                break
            self._demote(victim, to_tier, to_tier + 1)
        # Move to new tier
        self.backends[to_tier].put(group_hash, data, tokens)
        self.backends[from_tier].put(group_hash, b"", b"")  # Optionally remove from old tier
        self.tier_usage[from_tier]['count'] -= 1
        self.tier_usage[from_tier]['bytes'] -= size_bytes
        self.tier_usage[to_tier]['count'] += 1
        self.tier_usage[to_tier]['bytes'] += size_bytes
        meta['tier'] = to_tier
        meta['access_count'] = 0
        meta['last_access'] = time.time()

    def _demote(self, group_hash, from_tier, to_tier):
        """
        Move fragment from from_tier to to_tier. If to_tier is the last, evict.
        """
        meta = self.metadata[group_hash]
        data, tokens = self.backends[from_tier].get(group_hash)
        size_bytes = meta['size_bytes']
        if to_tier >= len(self.backends):
            # Evict from system
            self.backends[from_tier].put(group_hash, b"", b"")
            self.tier_usage[from_tier]['count'] -= 1
            self.tier_usage[from_tier]['bytes'] -= size_bytes
            del self.metadata[group_hash]
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

    def _proactive_demote(self, tier_idx):
        """
        Proactively demote coldest fragments if tier is above watermark.
        """
        config = self.tier_configs[tier_idx]
        usage = self.tier_usage[tier_idx]
        while (
            (config.get('capacity_count') and usage['count'] > config['capacity_count'] * config.get('watermark', 0.9)) or
            (config.get('capacity_bytes') and usage['bytes'] > config['capacity_bytes'] * config.get('watermark', 0.9))
        ):
            victim = self._find_coldest_fragment(tier_idx)
            if victim is None:
                break
            self._demote(victim, tier_idx, tier_idx + 1)

    def put(self, group_hash: str, data: Any, tokens: Any) -> None:
        """
        Store data and tokens in the highest (fastest) tier, update metadata and usage.
        Evict/demote if tier is full or above watermark.
        """
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
        self.backends[tier_idx].put(group_hash, data, tokens)
        usage['count'] += 1
        usage['bytes'] += size_bytes
        self._update_metadata(group_hash, tier_idx, size_bytes, access=False)

    def get(self, group_hash: str) -> Optional[tuple]:
        """
        Retrieve data and tokens, searching from fastest to slowest tier.
        Update access tracking and promote if needed.
        """
        for tier_idx, backend in enumerate(self.backends):
            result = backend.get(group_hash)
            if result is not None:
                size_bytes = None
                if group_hash in self.metadata:
                    size_bytes = self.metadata[group_hash].get('size_bytes', None)
                meta = self._update_metadata(group_hash, tier_idx, size_bytes or 0, access=True)
                # Promotion logic
                if tier_idx > 0 and meta['access_count'] >= meta.get('promotion_threshold', 1):
                    if self._can_promote(group_hash, tier_idx, tier_idx - 1):
                        self._promote(group_hash, tier_idx, tier_idx - 1)
                return result
        return None

    def exists(self, group_hash: str) -> bool:
        """
        Check if data exists in any tier.
        """
        return any(backend.exists(group_hash) for backend in self.backends) 