#!/usr/bin/env python3
"""
Benchmarking script for VUA TieredBackend with LeCAR policy.

Allows testing different tier configurations, policy parameters, and workload patterns
to evaluate performance and policy adaptation.
"""

import argparse
import time
import random
import os
import shutil
import logging
import csv
import math
from collections import defaultdict

# Add src directory to path to import vua backend
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
sys.path.insert(0, src_dir)

try:
    from vua.backend import (
        TieredBackend,
        PolicyConfig,
        LeCAR,
        MockGPURAMBackend,
        MockPMDKBackend,
        FileSystemBackend,
        CacheEntry
    )
except ImportError as e:
    print(f"Error importing VUA modules: {e}")
    print(f"Ensure the script is run from the workspace root or add src to PYTHONPATH.")
    sys.exit(1)

# Optional: numpy for Zipfian distribution
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not found. Zipfian workload unavailable.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("benchmark")

def setup_backend(args):
    """Creates the TieredBackend based on parsed arguments."""
    test_dir = args.fs_path
    if os.path.exists(test_dir):
        logger.info(f"Cleaning up existing test directory: {test_dir}")
        shutil.rmtree(test_dir)
    os.makedirs(os.path.join(test_dir, "disk_tier"))
    
    backends = [
        MockGPURAMBackend(),
        MockPMDKBackend(),
        FileSystemBackend(os.path.join(test_dir, "disk_tier"))
    ]
    
    tier_configs = [
        {'name': 'ram', 'capacity_count': args.cap_t0},
        {'name': 'pmem', 'capacity_count': args.cap_t1},
        {'name': 'disk', 'capacity_count': args.cap_t2}
    ]
    
    policy_config = PolicyConfig(
        learning_rate=args.lr,
        exploration_rate=args.er,
        initial_lru_weight=args.w_lru,
        initial_lfu_weight=args.w_lfu,
        ghost_cache_ttl=args.ttl,
        min_weight=args.w_min,
        max_weight=args.w_max,
        base_promotion_score_threshold=args.p_thresh,
        base_demotion_score_threshold=args.d_thresh
    )
    policy = LeCAR(policy_config)
    
    tiered_backend = TieredBackend(
        backends=backends,
        tier_configs=tier_configs,
        policy=policy,
        adjustment_interval=args.adj_interval
    )
    logger.info("TieredBackend initialized.")
    logger.info(f"Policy Config: {policy_config}")
    logger.info(f"Tier Configs: {tier_configs}")
    return tiered_backend

def generate_workload(workload_type, num_ops, num_items, read_ratio, zipf_param=1.1):
    """Generates a sequence of (operation, key) tuples."""
    logger.info(f"Generating workload: type={workload_type}, ops={num_ops}, items={num_items}, read%={read_ratio*100:.1f}")
    keys = [f"item_{i}" for i in range(num_items)]
    
    if workload_type == 'random':
        for _ in range(num_ops):
            op_type = 'get' if random.random() < read_ratio else 'put'
            key = random.choice(keys)
            yield (op_type, key)
            
    elif workload_type == 'sequential':
        i = 0
        for _ in range(num_ops):
            op_type = 'get' if random.random() < read_ratio else 'put'
            key = keys[i % num_items]
            yield (op_type, key)
            i += 1
            
    elif workload_type == 'zipfian':
        if not NUMPY_AVAILABLE:
            raise RuntimeError("Numpy is required for Zipfian workload.")
        # Generate samples from Zipfian distribution
        # Note: numpy.random.zipf generates values k >= 1.
        # We sample indices from 0 to num_items-1, so we sample zipf(a, num_items)
        # and map values > num_items back into range (or use other methods)
        # A simpler way is to generate probabilities and use choices.
        probs = [(1.0 / (i + 1)**zipf_param) for i in range(num_items)]
        total_prob = sum(probs)
        normalized_probs = [p / total_prob for p in probs]
        
        sampled_keys = np.random.choice(keys, size=num_ops, p=normalized_probs)
        
        for i in range(num_ops):
            op_type = 'get' if random.random() < read_ratio else 'put'
            yield (op_type, sampled_keys[i])
    else:
        raise ValueError(f"Unknown workload type: {workload_type}")

def run_benchmark(backend, workload_generator, num_ops):
    """Runs the benchmark and collects metrics."""
    logger.info(f"Starting benchmark run: {num_ops} operations...")
    start_time = time.time()
    
    total_gets = 0
    total_puts = 0
    total_hits = 0
    latencies = []
    
    # Dummy data for puts
    put_data = b'x' * 100
    put_tokens = b'y' * 10
    
    i = 0
    for op_type, key in workload_generator:
        if i >= num_ops: break
        i += 1
        
        op_start_time = time.time()
        if op_type == 'get':
            total_gets += 1
            result = backend.get(key)
            if result is not None:
                total_hits += 1
        elif op_type == 'put':
            total_puts += 1
            backend.put(key, put_data, put_tokens)
        op_end_time = time.time()
        latencies.append(op_end_time - op_start_time)
        
        if i % (num_ops // 10) == 0: # Log progress every 10%
             logger.info(f"Progress: {i}/{num_ops} ({i/num_ops*100:.0f}%)")
             
    end_time = time.time()
    total_time = end_time - start_time
    ops_per_sec = num_ops / total_time if total_time > 0 else 0
    hit_rate = total_hits / total_gets if total_gets > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    # Collect final policy state if possible
    final_lru_weight = backend.policy.lru_weight if hasattr(backend.policy, 'lru_weight') else 'N/A'
    final_lfu_weight = backend.policy.lfu_weight if hasattr(backend.policy, 'lfu_weight') else 'N/A'

    # Collect tier usage
    tier_usage_counts = {i: backend.tier_usage[i]['count'] for i in range(len(backend.backends))}

    logger.info("Benchmark run finished.")
    
    results = {
        'total_time_s': total_time,
        'ops_per_sec': ops_per_sec,
        'total_ops': num_ops,
        'total_gets': total_gets,
        'total_puts': total_puts,
        'total_hits': total_hits,
        'hit_rate': hit_rate,
        'avg_latency_ms': avg_latency * 1000,
        'final_lru_weight': final_lru_weight,
        'final_lfu_weight': final_lfu_weight,
        'final_tier_usage': tier_usage_counts
    }
    return results

def print_results(results, args):
    """Prints a summary of the benchmark results."""
    print("\n--- Benchmark Summary ---")
    print(f"Workload: {args.workload_type}, Ops: {args.num_ops}, Items: {args.num_items}, Read%: {args.read_ratio*100:.1f}")
    print(f"Policy Config: LR={args.lr}, ER={args.er}, PThresh={args.p_thresh}, DThresh={args.d_thresh}, Weights(LRU/LFU)=({args.w_lru}/{args.w_lfu}) TTL={args.ttl}")
    print(f"Tier Caps (0/1/2): {args.cap_t0}/{args.cap_t1}/{args.cap_t2}")
    print(f"Total Time: {results['total_time_s']:.2f} s")
    print(f"Operations/Sec: {results['ops_per_sec']:.2f}")
    print(f"Avg Op Latency: {results['avg_latency_ms']:.3f} ms")
    print(f"Overall Hit Rate: {results['hit_rate']:.4f} ({results['total_hits']}/{results['total_gets']}) ")
    print(f"Final Policy Weights (LRU/LFU): {results['final_lru_weight']:.3f}/{results['final_lfu_weight']:.3f}")
    print(f"Final Tier Usage (Count): {results['final_tier_usage']}")
    print("-------------------------")

def save_results_csv(results, args, filename):
    """Appends results to a CSV file."""
    fieldnames = [
        'timestamp', 'workload_type', 'num_ops', 'num_items', 'read_ratio', 'zipf_param',
        'cap_t0', 'cap_t1', 'cap_t2', 
        'lr', 'er', 'w_lru', 'w_lfu', 'ttl', 'w_min', 'w_max', 'p_thresh', 'd_thresh',
        'adj_interval',
        'total_time_s', 'ops_per_sec', 'total_gets', 'total_puts', 'total_hits',
        'hit_rate', 'avg_latency_ms', 
        'final_lru_weight', 'final_lfu_weight', 'final_tier_usage'
    ]
    
    row = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'workload_type': args.workload_type,
        'num_ops': args.num_ops,
        'num_items': args.num_items,
        'read_ratio': args.read_ratio,
        'zipf_param': args.zipf_param if args.workload_type == 'zipfian' else 'N/A',
        'cap_t0': args.cap_t0,
        'cap_t1': args.cap_t1,
        'cap_t2': args.cap_t2,
        'lr': args.lr,
        'er': args.er,
        'w_lru': args.w_lru,
        'w_lfu': args.w_lfu,
        'ttl': args.ttl,
        'w_min': args.w_min,
        'w_max': args.w_max,
        'p_thresh': args.p_thresh,
        'd_thresh': args.d_thresh,
        'adj_interval': args.adj_interval,
        **results # Merge in calculated results
    }
    
    file_exists = os.path.isfile(filename)
    try:
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        logger.info(f"Results appended to {filename}")
    except IOError as e:
        logger.error(f"Failed to write results to CSV {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark VUA TieredBackend with LeCAR policy.")
    
    # Workload Args
    parser.add_argument("-n", "--num-ops", type=int, default=10000, help="Total number of operations")
    parser.add_argument("-i", "--num-items", type=int, default=1000, help="Number of unique items")
    parser.add_argument("-w", "--workload-type", choices=['random', 'sequential', 'zipfian'], default='random', help="Workload access pattern")
    parser.add_argument("-r", "--read-ratio", type=float, default=0.8, help="Ratio of GET operations (0.0 to 1.0)")
    parser.add_argument("--zipf-param", type=float, default=1.1, help="Zipf distribution parameter (alpha > 1.0)")
    
    # Tier Args
    parser.add_argument("--cap-t0", type=int, default=100, help="Capacity (count) of Tier 0 (RAM)")
    parser.add_argument("--cap-t1", type=int, default=500, help="Capacity (count) of Tier 1 (PMEM)")
    parser.add_argument("--cap-t2", type=int, default=1000, help="Capacity (count) of Tier 2 (Disk)")
    parser.add_argument("--fs-path", type=str, default="/tmp/vua_benchmark_fs", help="Path for FileSystemBackend tier")
    
    # Policy Args (matching PolicyConfig)
    parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate")
    parser.add_argument("--er", type=float, default=0.0, help="Exploration Rate (default 0 for benchmarks)") # Default 0 for benchmark predictability
    parser.add_argument("--w-lru", type=float, default=0.5, help="Initial LRU weight")
    parser.add_argument("--w-lfu", type=float, default=0.5, help="Initial LFU weight")
    parser.add_argument("--ttl", type=int, default=3600, help="Ghost Cache TTL (seconds)")
    parser.add_argument("--w-min", type=float, default=0.1, help="Min policy weight")
    parser.add_argument("--w-max", type=float, default=0.9, help="Max policy weight")
    parser.add_argument("--p-thresh", type=float, default=0.7, help="Base Promotion Score Threshold")
    parser.add_argument("--d-thresh", type=float, default=0.3, help="Base Demotion Score Threshold")
    
    # Backend Arg
    parser.add_argument("--adj-interval", type=int, default=1000, help="Feedback adjustment interval (operations)")
    
    # Output Arg
    parser.add_argument("-o", "--output-csv", type=str, default="benchmark_results.csv", help="CSV file to append results")

    args = parser.parse_args()

    # Basic validation
    if not (0.0 <= args.read_ratio <= 1.0):
        parser.error("Read ratio must be between 0.0 and 1.0")
    if args.workload_type == 'zipfian' and not NUMPY_AVAILABLE:
         parser.error("Numpy is required for Zipfian workload. Please install it (`pip install numpy`).")
    if args.zipf_param <= 1.0 and args.workload_type == 'zipfian':
         parser.error("Zipfian parameter (alpha) must be > 1.0")

    try:
        backend = setup_backend(args)
        workload_gen = generate_workload(args.workload_type, args.num_ops, args.num_items, args.read_ratio, args.zipf_param)
        results = run_benchmark(backend, workload_gen, args.num_ops)
        print_results(results, args)
        if args.output_csv:
            save_results_csv(results, args, args.output_csv)
    except Exception as e:
        logger.exception(f"An error occurred during benchmark: {e}")
    finally:
        # Cleanup filesystem backend dir
        if os.path.exists(args.fs_path):
            logger.info(f"Cleaning up test directory: {args.fs_path}")
            # shutil.rmtree(args.fs_path) # Optional: uncomment to auto-cleanup
        pass 

if __name__ == "__main__":
    main() 