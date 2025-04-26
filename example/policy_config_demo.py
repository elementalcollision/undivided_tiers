#!/usr/bin/env python3
"""
Example demonstrating VUA's configurable tiering policy features.
Shows how to:
1. Configure policy parameters
2. Update policy configuration at runtime
3. Monitor policy metrics via Prometheus
"""

import time
import logging
from vua.backend import (
    TieredBackend,
    PolicyConfig,
    MockGPURAMBackend,
    MockPMDKBackend,
    FileSystemBackend
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Create backends for each tier
    backends = [
        MockGPURAMBackend(),  # Tier 0 (fastest)
        MockPMDKBackend(),    # Tier 1
        FileSystemBackend("/tmp/vua_cache")  # Tier 2 (slowest)
    ]
    
    # Configure tiers
    tier_configs = [
        {
            'name': 'gpu',
            'capacity_count': 100,
            'capacity_bytes': 1024 * 1024 * 1024,  # 1GB
            'promotion_threshold': 2,
            'demotion_threshold': 1
        },
        {
            'name': 'pmem',
            'capacity_count': 1000,
            'capacity_bytes': 10 * 1024 * 1024 * 1024,  # 10GB
            'promotion_threshold': 3,
            'demotion_threshold': 1
        },
        {
            'name': 'disk',
            'capacity_count': 10000,
            'promotion_threshold': 5,
            'demotion_threshold': 2
        }
    ]
    
    # Create policy configuration
    policy_config = PolicyConfig(
        learning_rate=0.2,          # Faster learning
        exploration_rate=0.15,      # More exploration
        initial_lru_weight=0.6,     # Bias towards recency
        initial_lfu_weight=0.4,     # Less frequency bias
        ghost_cache_ttl=1800,       # 30 minutes
        min_weight=0.2,             # Minimum 20% for each policy
        max_weight=0.8              # Maximum 80% for each policy
    )
    
    # Create TieredBackend with policy configuration
    backend = TieredBackend(
        backends=backends,
        tier_configs=tier_configs,
        policy_config=policy_config,
        adjustment_interval=100  # Adjust thresholds more frequently
    )
    
    # Start Prometheus server
    backend.start_prometheus_server(port=8000)
    logger.info("Started Prometheus server on port 8000")
    logger.info("View metrics at http://localhost:8000/metrics")
    
    # Simulate some workload
    logger.info("Simulating workload...")
    for i in range(1000):
        # Every 250 operations, update policy configuration
        if i > 0 and i % 250 == 0:
            logger.info(f"Operation {i}: Updating policy configuration")
            new_config = PolicyConfig(
                learning_rate=0.1 + (i/1000),  # Gradually increase learning rate
                exploration_rate=max(0.05, 0.15 - (i/2000)),  # Gradually decrease exploration
                initial_lru_weight=backend.policy.lru_weight,  # Keep current weights
                initial_lfu_weight=backend.policy.lfu_weight,
                ghost_cache_ttl=1800,
                min_weight=0.2,
                max_weight=0.8
            )
            backend.update_policy_config(new_config)
            
        # Simulate data access patterns
        group_hash = f"group_{i % 100}"  # 100 unique groups
        data = f"data_{i}".encode()
        tokens = f"tokens_{i}".encode()
        
        # Write data
        backend.put(group_hash, data, tokens)
        
        # Read data with varying patterns
        if i % 2 == 0:  # Frequent access to even-numbered groups
            backend.get(f"group_{(i//2) % 50}")
        if i % 5 == 0:  # Less frequent access to every 5th group
            backend.get(f"group_{(i//5) % 20}")
        
        # Small delay to simulate real workload
        time.sleep(0.01)
        
        # Log current policy weights periodically
        if i % 100 == 0:
            logger.info(
                f"Operation {i}: LRU weight = {backend.policy.lru_weight:.3f}, "
                f"LFU weight = {backend.policy.lfu_weight:.3f}"
            )
    
    logger.info("Workload simulation complete")
    logger.info("Prometheus metrics server still running. Press Ctrl+C to exit.")
    
    # Keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")

if __name__ == "__main__":
    main() 