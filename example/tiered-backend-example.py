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
from vua.backend import (
    TieredBackend,
    MockPMDKBackend,
    FileSystemBackend,
    MockGPURAMBackend,
    PMDKBackend,
    StorageBackend
)

def generate_rand_kvcache(n_layers, seq_len, batch_size, num_heads, head_size):
    # Simplified version for example
    layers = []
    for i in range(0, n_layers):
        size = (batch_size, num_heads, seq_len, head_size)
        t = torch.randn(size, dtype=torch.float16)
        layers.append([t, t.clone()]) # Simulating K and V
    return layers

def main():
    parser = argparse.ArgumentParser(description="VUA Tiered Backend Example")
    parser.add_argument("--tier0-backend", default="mock_gpu", choices=["mock_gpu", "mock_dram", "fs", "pmdk"], help="Backend for Tier 0 (fastest)")
    parser.add_argument("--tier1-backend", default="mock_dram", choices=["mock_gpu", "mock_dram", "fs", "pmdk"], help="Backend for Tier 1")
    parser.add_argument("--tier2-backend", default="fs", choices=["mock_gpu", "mock_dram", "fs", "pmdk"], help="Backend for Tier 2 (slowest)")
    parser.add_argument("--fs-path", default=None, help="Path for FileSystemBackend (used if 'fs' is selected for any tier)")
    parser.add_argument("--pmdk-path", default=None, help="Path for PMDK pool file (used if 'pmdk' is selected for any tier)")
    parser.add_argument("--pmdk-size", type=int, default=1024*1024*100, help="Size (bytes) for PMDK pool creation") # 100MB default
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set logging level")
    parser.add_argument("--prometheus-port", type=int, default=None, help="Port to start Prometheus metrics server on (e.g., 8000)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s.%(msecs)03d [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    with tempfile.TemporaryDirectory() as temp_dir:
        default_fs_path = args.fs_path or os.path.join(temp_dir, 'storage_tier')
        default_pmdk_path = args.pmdk_path or os.path.join(temp_dir, 'pmdk.pool')
        os.makedirs(os.path.dirname(default_fs_path), exist_ok=True)
        os.makedirs(os.path.dirname(default_pmdk_path), exist_ok=True)

        def create_backend(backend_type, tier_idx):
            if backend_type == "mock_gpu":
                logging.info(f"Using MockGPURAMBackend for Tier {tier_idx}")
                return MockGPURAMBackend()
            elif backend_type == "mock_dram":
                logging.info(f"Using MockPMDKBackend (as DRAM) for Tier {tier_idx}")
                return MockPMDKBackend()
            elif backend_type == "fs":
                path = default_fs_path + f"_t{tier_idx}" # Ensure unique paths if multiple FS tiers
                logging.info(f"Using FileSystemBackend for Tier {tier_idx} at {path}")
                return FileSystemBackend(path)
            elif backend_type == "pmdk":
                path = default_pmdk_path + f"_t{tier_idx}" # Ensure unique paths if multiple PMDK tiers
                logging.info(f"Using PMDKBackend for Tier {tier_idx} at {path} (size: {args.pmdk_size})")
                try:
                    # Attempt to create/open the pool
                    return PMDKBackend(path, pool_size=args.pmdk_size, create=True)
                except ImportError as e:
                    logging.error(f"Cannot use PMDK backend: {e}. Falling back to Mock DRAM.")
                    return MockPMDKBackend()
                except Exception as e:
                    logging.error(f"Error initializing PMDK backend at {path}: {e}. Falling back to Mock DRAM.")
                    return MockPMDKBackend()
            else:
                raise ValueError(f"Unknown backend type: {backend_type}")

        # Configure Tiers based on args
        tier_configs = [
            {'name': 'tier0', 'capacity_count': 1, 'capacity_bytes': 10000, 'promotion_threshold': 2, 'watermark': 0.9},
            {'name': 'tier1', 'capacity_count': 2, 'capacity_bytes': 20000, 'promotion_threshold': 2, 'watermark': 0.9},
            {'name': 'tier2', 'capacity_count': 10, 'capacity_bytes': 50000, 'promotion_threshold': 2, 'watermark': 0.9},
        ]
        backends = [
            create_backend(args.tier0_backend, 0),
            create_backend(args.tier1_backend, 1),
            create_backend(args.tier2_backend, 2),
        ]

        # Create TieredBackend
        tiered_backend = TieredBackend(backends, tier_configs)

        # Instantiate VUA with TieredBackend
        cache = VUA(VUAConfig, temp_dir, backend=tiered_backend)

        # Start Prometheus server if requested
        if args.prometheus_port:
            if isinstance(tiered_backend, TieredBackend):
                tiered_backend.start_prometheus_server(args.prometheus_port)
            else:
                logging.warning("Prometheus export only available when using TieredBackend.")

        logging.info(f"--- Tiered Backend Example ({args.tier0_backend}/{args.tier1_backend}/{args.tier2_backend}) ---")

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

        # Keep running if Prometheus server is active
        if args.prometheus_port:
            logging.info(f"Prometheus server running on port {args.prometheus_port}. Press Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logging.info("Exiting.")

if __name__ == "__main__":
    main() 