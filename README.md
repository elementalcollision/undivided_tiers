<p align="center">
  <img src="static/vast_stacked_logo_hex_white.jpg" alt="VAST DATA Logo" width="50%"/>
</p>

---

# VUA

VUA is a system for storing and retrieving key-value caches of deep learning models, utilizing a directory structure derived from token values.

## Developer Quick Start

1. **Understanding VUA**

   VUA splits tokens into groups defined by a fixed split factor (see `VUAConfig.split_factor`). Tokens are converted to directory names where cache data is stored. The `put()` method splits key-value caches and stores fragmented tensor data into these directories, while `get_closest()` fetches the closest matching cache fragment based on supplied token prefixes.

2. **Exploring the Codebase**

   - **`src/vua/core.py`**: Contains the primary VUA implementation with `put()` and `get_closest()`, plus token-to-path conversion logic.
   - **`src/vua/serdes.py`**: Provides tensor serialization/deserialization functions (`tensor_to_bytes` and `bytes_to_tensor`).
   - **`src/vua/backend.py`**: Defines the storage backend abstraction (`StorageBackend`) and implementations like `FileSystemBackend`, `MockPMDKBackend`, `PMDKBackend` (simulated), and `TieredBackend`.
   - **`tests/test_vua.py`**: Offers tests to validate core logic, backends, and tiering.

3. **Running the Tests**

   Run the tests by executing:
   ```bash
   uv run python -m unittest discover -s tests
   ```
4. **Using VUA in Your Project**

   - Create a VUA instance by providing a configuration (`VUAConfig`), a root path, and optionally a custom storage backend (defaults to `FileSystemBackend`).
   - Utilize `put()` to store computed key-value caches and `get_closest()` to retrieve cached data based on token queries.
   - **Batched Operations:** VUA supports batched put and get operations. If you provide a 2D tensor of tokens to `put()`, it processes each sequence in parallel. Similarly, calling `get_closest()` with a list of token tensors returns a list of corresponding `ClosestKV` results.

   **Example (Default Filesystem Backend):**

   ```python
   import os
   import torch
   from vua.core import VUA, VUAConfig

   # Set up cache storage directory
   cache_dir = "./vua_cache"
   os.makedirs(cache_dir, exist_ok=True)

   # Create a VUA instance
   vua = VUA(VUAConfig, cache_dir)

   # Generate sample tokens ensuring the length is divisible by split_factor
   tokens = torch.randint(0, 0xFFFF, (1, 512), dtype=torch.uint16)
   trimmed_tokens = VUAConfig.trim_to_split_factor(tokens)

   # Create a sample key-value cache (for demonstration purposes)
   # This is a list with one layer and two tensor pairs (keys and values)
   kvcache = [[torch.randn(1, 2, trimmed_tokens.size(1), 64), torch.randn(1, 2, trimmed_tokens.size(1), 64)]]

   # Store the cache using put()
   vua.put(trimmed_tokens, kvcache)

   # Retrieve the cache using get_closest()
   result = vua.get_closest(trimmed_tokens, device="cpu")
   print("Retrieved tokens:", result.tokens)
   print("Retrieved data:", result.data)
   ```

   **Example (Tiered Backend):**

   ```python
   # See example/tiered-backend-example.py for a full example
   from vua.backend import TieredBackend, MockPMDKBackend, FileSystemBackend
   # ... imports ...
   storage_path = "./storage_tier"
   os.makedirs(storage_path, exist_ok=True)
   tier_configs = [
       {'name': 'dram', 'capacity_count': 100, ...},
       {'name': 'storage', 'capacity_count': 1000, ...},
   ]
   backends = [MockPMDKBackend(), FileSystemBackend(storage_path)]
   tiered_backend = TieredBackend(backends, tier_configs)
   # Inject the tiered backend
   vua = VUA(VUAConfig, ".", backend=tiered_backend) # root_path less relevant here
   # ... use vua.put() and vua.get_closest() ...
   ```

5. **Examples directory**

   The examples directory contains:

   - Usage with 'transformers' library (`on-transformers.py`)
     ```
     uv run python ./example/on-transformers.py
     ```
   - Example demonstrating the TieredBackend (`tiered-backend-example.py`)
     - This example now accepts command-line arguments to configure the backend types (mock_gpu, mock_dram, fs, pmdk) and paths for each tier. Run with `-h` for details.
     ```
     uv run python ./example/tiered-backend-example.py
     # Example with PMDK for tier 1:
     # uv run python ./example/tiered-backend-example.py --tier1-backend pmdk --pmdk-path /mnt/pmem/my_pool.pmdk
     ```
   - Usage with experimental vLLM connector (`vllm_kv_connector_0.py`, `serve-vllm.sh`, etc.)
     ```
     uv run ./example/serve-vllm.sh
     ```

6. **Debugging and Logging**

   VUA leverages Python's `logging` module for detailed debug output. Configure custom log handlers during development to monitor directory navigation and cache operations effectively.

# Repairing Symlinks in the Cache Directory

If your VUA cache directory has missing or broken `parent` symlinks (for example, due to interrupted writes or manual changes), you can repair them using the provided CLI tool:

```
python scripts/repair_symlinks.py /path/to/vua_cache --log-level DEBUG
```

- Replace `/path/to/vua_cache` with the path to your cache directory.
- The `--log-level` flag is optional and can be set to `DEBUG`, `INFO`, `WARNING`, or `ERROR`.

This script will scan all group directories in the cache, check for missing or broken `parent` symlinks, and repair them where possible. It is safe to run multiple times.

# License

VUA is released under the Apache License Version 2.0.

See the file LICENSE for more details.

## File and Directory Structure

- `src/vua/core.py`: Main VUA logic and API
- `src/vua/serdes.py`: Tensor serialization/deserialization
- `src/vua/backend.py`: Storage backend abstractions, including FileSystemBackend, PMDKBackend, MockPMDKBackend, and TieredBackend (Colloid-inspired tiering logic)
- `tests/`: Unit tests for VUA, backends, and tiering
- `scripts/`: CLI and utility scripts
- `example/`, `samples/`, `static/`: Usage examples, sample data, and static assets

## Tiered Backend and Colloid-Inspired Tiering

VUA now supports multi-tier cache management using a Colloid-inspired algorithm:
- Data is managed across multiple tiers (e.g., GPU RAM, DRAM, CXL/PMEM, Storage) using the `TieredBackend` abstraction.
- Promotion, demotion, and eviction policies are based on access frequency, recency, and dynamic thresholds.
- Metadata is tracked for each cache fragment, with advanced heuristics activated as cache fills.
- The system adapts to access patterns and tier pressure, keeping hot data in the fastest available memory.
- **Metrics:** Key performance indicators (hits, misses, promotions, demotions, evictions, tier usage) and aggregated statistics (distributions of fragment age, access count, size) are exported for monitoring via Prometheus (requires optional `prometheus-client` dependency).

See `src/vua/backend.py` and `project.md` for details on the tiering logic and configuration.

## Dynamic Policy Tuning in TieredBackend

The TieredBackend now supports dynamic adjustment of key policy parameters:
- **promotion_threshold**: Adjusted based on hit rate for each tier.
- **demotion_threshold**: Adjusted based on demotion/eviction rates and tier usage.
- **watermark**: Adjusted based on tier usage relative to capacity.

These adjustments are made automatically at runtime, logged for traceability, and exposed as Prometheus metrics for observability. Comprehensive tests validate that these thresholds and watermarks adjust as expected under different usage scenarios.

## Key Features

- **Key-Value Caching:** Stores and retrieves deep learning model KV caches.
- **Token-Based Directory Structure:** Efficiently organizes cache fragments based on token sequences.
- **Batched Operations:** Supports parallel processing of multiple sequences.
- **Pluggable Storage Backends:** Allows using different storage types:
    - `FileSystemBackend`: Standard file system storage.
    - `PMDKBackend`: (Experimental) Integration with Intel PMDK for persistent memory (requires `py-pmemobj`).
    - Mock Backends: `MockGPURAMBackend`, `MockPMDKBackend` for testing.
- **Tiered Storage (`TieredBackend`):** Manages data across multiple storage tiers (e.g., RAM, PMEM, Disk).
    - **Adaptive Policy (LeCAR):** Uses a learning cache policy (LeCAR) combining LRU and LFU with adaptive weights based on ghost cache hits.
    - **Dynamic Tuning:** Automatically adjusts policy parameters (e.g., exploration rate, promotion/demotion score thresholds) based on observed metrics like hit rates and tier usage via a feedback loop (`_feedback_adjust_thresholds`).
    - **Configurable Policy:** LeCAR policy parameters (`learning_rate`, `exploration_rate`, initial weights, score thresholds, etc.) are configurable via `PolicyConfig`.
    - **Prometheus Metrics:** Exports detailed metrics for tier usage, cache performance, policy state (weights, ghost cache size), and score distributions.

## Benchmarking Policy Performance

A benchmarking script is available to evaluate the performance and adaptation of the `TieredBackend` with the `LeCAR` policy under various conditions.

**Location:** `scripts/benchmark_policy.py`

**Purpose:**
- Measure key performance metrics (OPS, latency, hit rate).
- Observe how policy weights and tier usage evolve under different workloads.
- Compare the effectiveness of different policy configurations.
- Gather data to refine tuning heuristics.

**How to Run:**
- Execute from the workspace root: `python scripts/benchmark_policy.py [OPTIONS]`
- Use `python scripts/benchmark_policy.py -h` to see all available options.

**Key Configuration Options:**
- **Workload:** `-n` (num ops), `-i` (num items), `-w` (type: `random`, `sequential`, `zipfian`), `-r` (read ratio), `--zipf-param`.
- **Tier Capacities:** `--cap-t0`, `--cap-t1`, `--cap-t2` (item counts).
- **Policy (`PolicyConfig`):** `--lr` (learning rate), `--er` (exploration rate), `--w-lru`/`--w-lfu` (initial weights), `--ttl` (ghost TTL), `--w-min`/`--w-max` (weight bounds), `--p-thresh` (base promotion threshold), `--d-thresh` (base demotion threshold).
- **Backend:** `--adj-interval` (feedback loop frequency).
- **Output:** `-o` (CSV file to append results).

**Example:**
```bash
# Requires numpy for zipfian
# pip install numpy 
python scripts/benchmark_policy.py -n 20000 -i 2000 -w zipfian --zipf-param 1.2 --cap-t0 150 --cap-t1 600 --cap-t2 1500 --lr 0.15 --er 0.05 --p-thresh 0.65 --d-thresh 0.35 -o zipfian_test.csv
```

**Relevant Files:**
- `src/vua/backend.py`: Contains `TieredBackend`, `LeCAR`, `PolicyConfig`.
- `scripts/benchmark_policy.py`: The benchmarking script itself.

