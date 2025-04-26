#!/usr/bin/env python3

import torch
import os
import time
import sys
import argparse
import logging
import tempfile
import shutil
from typing import Any, Optional

from vua.core import VUA, VUAConfig
from vua.backend import TieredBackend, MockPMDKBackend, FileSystemBackend

def generate_rand_kvcache(n_layers, seq_len, batch_size, num_heads, head_size):
    # Simplified version for example
    layers = []
    for i in range(0, n_layers):
        size = (batch_size, num_heads, seq_len, head_size)
        t = torch.randn(size, dtype=torch.float16)
        layers.append([t, t.clone()]) # Simulating K and V
    return layers

# Assuming MockPMDKBackend simulates DRAM or PMEM, and we need another mock for GPU RAM
class MockGPURAMBackend(StorageBackend):
    """
    Mock backend simulating GPU RAM using an in-memory dict.
    """
    def __init__(self):
        self._store = {}

    def put(self, group_hash: str, data: Any, tokens: Any) -> None:
        self._store[group_hash] = (data, tokens)

    def get(self, group_hash: str) -> Optional[tuple]:
        return self._store.get(group_hash)

    def exists(self, group_hash: str) -> bool:
        return group_hash in self._store

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, 'storage_tier')
        os.makedirs(storage_path)

        # Configure Tiers (GPU RAM + DRAM + Filesystem Storage)
        tier_configs = [
            {'name': 'gpu', 'capacity_count': 1, 'capacity_bytes': 10000, 'promotion_threshold': 2, 'watermark': 0.9},
            {'name': 'dram', 'capacity_count': 2, 'capacity_bytes': 20000, 'promotion_threshold': 2, 'watermark': 0.9},
            {'name': 'storage', 'capacity_count': 10, 'capacity_bytes': 50000, 'promotion_threshold': 2, 'watermark': 0.9},
        ]
        backends = [
            MockGPURAMBackend(), # Tier 0: Mock GPU RAM
            MockPMDKBackend(), # Tier 1: Mock DRAM
            FileSystemBackend(storage_path) # Tier 2: Storage
        ]

        # Create TieredBackend
        tiered_backend = TieredBackend(backends, tier_configs)

        # Instantiate VUA with TieredBackend
        # Note: root_path for VUA is less relevant when using a custom backend,
        # but required by the constructor. We pass temp_dir.
        cache = VUA(VUAConfig, temp_dir, backend=tiered_backend)

        logging.info("--- Tiered Backend Example (3 Tiers) ---")

        # Generate sample data
        tokens1 = VUAConfig.trim_to_split_factor(torch.randint(0, 1000, (VUAConfig.split_factor,))) # Group 0
        tokens2 = VUAConfig.trim_to_split_factor(torch.randint(1000, 2000, (VUAConfig.split_factor,))) # Group 1
        tokens3 = VUAConfig.trim_to_split_factor(torch.randint(2000, 3000, (VUAConfig.split_factor,))) # Group 2
        tokens4 = VUAConfig.trim_to_split_factor(torch.randint(3000, 4000, (VUAConfig.split_factor,))) # Group 3

        kvcache1 = generate_rand_kvcache(1, VUAConfig.split_factor, 1, 1, 1)
        kvcache2 = generate_rand_kvcache(1, VUAConfig.split_factor, 1, 1, 1)
        kvcache3 = generate_rand_kvcache(1, VUAConfig.split_factor, 1, 1, 1)
        kvcache4 = generate_rand_kvcache(1, VUAConfig.split_factor, 1, 1, 1)

        # Put fragment 1 - goes to Tier 0 (GPU)
        logging.info("Putting fragment 1...")
        cache.put(tokens1, kvcache1)
        logging.info(f"Metadata: {tiered_backend.metadata}")

        # Put fragment 2 - demotes frag1 to Tier 1 (DRAM)
        logging.info("Putting fragment 2...")
        cache.put(tokens2, kvcache2)
        logging.info(f"Metadata: {tiered_backend.metadata}")

        # Put fragment 3 - demotes frag1 to Tier 2 (Storage), demotes frag2 to Tier 1 (DRAM)
        logging.info("Putting fragment 3...")
        cache.put(tokens3, kvcache3)
        logging.info(f"Metadata: {tiered_backend.metadata}")

        # Put fragment 4 - demotes frag2 to Tier 2 (Storage), demotes frag3 to Tier 1 (DRAM)
        logging.info("Putting fragment 4...")
        cache.put(tokens4, kvcache4)
        logging.info(f"Metadata after 4 puts:")
        for gh, meta in tiered_backend.metadata.items():
            logging.info(f"  {gh}: {meta}")
        # Expected: frag1, frag2 in tier 2; frag3 in tier 1; frag4 in tier 0

        # Access fragment 1 (in tier 2) multiple times to trigger promotion
        logging.info("Getting fragment 1 multiple times (expecting promotion)...")
        promo_threshold = tier_configs[0]['promotion_threshold'] # Promotion to GPU
        for i in range(promo_threshold + 1):
            res1 = cache.get_closest(tokens1, device="cpu")
            time.sleep(0.01)

        logging.info("Metadata after repeated gets:")
        for gh, meta in tiered_backend.metadata.items():
            logging.info(f"  {gh}: {meta}")
        # Expected: frag1 promoted to tier 0, others demoted accordingly

        logging.info("--- Example Complete ---")

if __name__ == "__main__":
    main() 