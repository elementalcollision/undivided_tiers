# VUA Tiering Enhancement & PMDK/CXL Integration

## Overview / Background

VUA currently stores key-value cache data on the filesystem. To support next-generation memory architectures and improve performance, we plan to add tiered storage capabilities, including support for CXL-attached memory and persistent memory via Intel PMDK.

## Objectives
- Enable VUA to store and retrieve cache data across multiple memory/storage tiers (DRAM, CXL/PMEM, disk).
- Integrate with Intel PMDK to leverage persistent memory and CXL memory types.
- Provide a configurable tiering policy (e.g., LRU, size, access frequency).
- Maintain compatibility with existing file-based storage.
- [x] **Dynamic Policy Tuning:**
    - The system now dynamically adjusts `promotion_threshold`, `demotion_threshold`, and `watermark` for each tier based on real-time feedback:
        - `promotion_threshold` is adjusted based on hit rate.
        - `demotion_threshold` is adjusted based on demotion/eviction rates and tier usage.
        - `watermark` is adjusted based on tier usage relative to capacity.
    - All adjustments are logged and exposed as Prometheus metrics for observability.
- [x] **Expanded Dynamic Policy Tuning:**
    - Implemented dynamic adjustment logic for `promotion_threshold`, `demotion_threshold`, and `watermark` in `TieredBackend`.
    - Added Prometheus metrics and logging for all threshold and watermark changes.
    - Updated code and documentation to reflect these new features.
    - **Comprehensive tests** now validate dynamic adjustment of all thresholds and watermarks (see `tests/test_vua.py`).
    - The `README.md` has been updated to highlight these features for users.

## Requirements
- Support for pluggable storage backends (filesystem, PMDK, etc.).
- Python integration with PMDK (via bindings or extension).
- Tiering logic to move data between memory types based on policy.
- Configuration options for tiering and backend selection.
- Robust testing on supported hardware.

## High-Level Design
- Introduce a `StorageBackend` abstraction in VUA.
- Implement `FileSystemBackend` (current) and `PMDKBackend` (new).
- Add tiering logic and policy configuration to VUAConfig or a new config file.
- Ensure serialization format compatibility across backends.

## Implementation Plan
1. Research PMDK APIs and Python integration options.
2. Refactor VUA to use a pluggable storage backend.
3. Implement and test the PMDK backend.
4. Add tiering logic and configuration.
5. Validate on CXL/PMEM hardware.
6. Document usage and configuration.

## GPU Server Integration (via SSH)

To enable large-scale benchmarking and real-world validation, the project will be integrated with a remote GPU-based server accessible over SSH. This section tracks the steps and requirements for successful deployment and execution on a GPU server.

### Checklist
- [ ] **Server Preparation**
    - [ ] Ensure the GPU server is accessible via SSH and credentials are available.
    - [ ] Confirm Python, CUDA, and required libraries (PyTorch, PMDK, etc.) are installed.
    - [ ] Set up a Python virtual environment on the server for isolation.
- [ ] **Code Deployment**
    - [ ] Use `rsync`, `scp`, or a Git workflow to deploy the codebase to the server.
    - [ ] Ensure all dependencies are installed (`pip install -r requirements.txt` or equivalent).
- [ ] **Configuration**
    - [ ] Update configuration files (e.g., `VUAConfig`, environment variables) for the server environment (paths, backend selection, GPU device, etc.).
    - [ ] Set up any required environment variables for CUDA, PMDK, and Prometheus.
- [ ] **Remote Execution**
    - [ ] Use SSH to run scripts and tests on the server (e.g., `ssh user@server 'cd /path/to/project && python scripts/benchmark_policy.py'`).
    - [ ] Optionally, use `tmux` or `screen` for persistent sessions.
- [ ] **Monitoring & Logging**
    - [ ] Set up Prometheus metrics endpoint to be accessible (open port or use SSH tunneling).
    - [ ] Configure logging to files for later retrieval and analysis.
- [ ] **Automation (Optional)**
    - [ ] Write helper scripts for deployment, remote execution, and log retrieval.
    - [ ] Consider using tools like `fabric`, `paramiko`, or GitHub Actions for automation.
- [ ] **Validation**
    - [ ] Run integration and performance tests on the GPU server.
    - [ ] Collect and analyze results, update tuning as needed.

---

## Risks & Open Questions
- Python/PMDK integration complexity.
- Hardware/OS support for CXL memory.
- Performance and reliability of tiering logic.
- Compatibility with existing VUA data.

## References & Resources
- [Intel PMDK GitHub](https://github.com/pmem/pmdk)
- [PMDK Documentation](https://pmem.io/pmdk/)
- [CXL Consortium](https://www.computeexpresslink.org/)

## Progress Log
- [x] Draft design doc and create project.md (complete)
- [x] Research PMDK Python integration (in progress)
    - Investigated available Python bindings for PMDK:
        - [py-pmemobj](https://github.com/pmem/py-pmemobj): Python bindings for libpmemobj (transactional object store).
        - [py-pmem](https://github.com/pmem/py-pmem): Python bindings for libpmem (low-level persistent memory support).
    - Plan: Prototype minimal usage of these bindings (create pool, persist object, read back) and evaluate suitability for VUA backend abstraction.
    - Will update this log with findings, issues, and next steps as research continues.
- [x] Refactor VUA for backend abstraction (complete)
    - The VUA class now accepts a backend parameter and uses the StorageBackend interface for all cache fragment storage and retrieval.
    - This enables easy swapping between filesystem, mock, and future PMDK backends, and is a foundation for tiering and further enhancements.
- [x] Prototype PMDK backend (complete)
    - Implemented PMDKBackend as a simulated backend using a dedicated directory (pmdk_pool) to mimic PMDK pool/object storage.
    - This allows for development and testing of backend integration and tiering logic without requiring CXL/PMEM hardware.
    - The class includes TODOs and docstrings for future replacement with real PMDK logic.
- [x] Implement tiering logic (complete)
    - Implemented Colloid-inspired tiering logic in TieredBackend, including promotion, demotion, eviction, and proactive demotion based on LRU/LFU and dynamic thresholds.
    - Added metadata schema and activation logic for advanced fields as cache fills.
    - Integrated snoop filter and per-tier thresholds.
    - Wrote comprehensive unit tests for insertion, retrieval, promotion, demotion, and eviction across tiers using mock backends.
    - Updated README and codebase to reflect new backend and tiering structure.
    - [x] Implement Proactive Demotion Logic (`_proactive_demote`)
        - [x] Define per-tier watermark configurations (bytes/count thresholds).
        - [x] Implement logic to check current tier usage against watermarks.
        - [x] Determine target usage level after demotion.
        - [x] Loop to find coldest fragments (`_find_coldest_fragment`) and demote them (`_demote`) until usage is below target.
        - [x] Add relevant logging.
        - [x] Consider trigger frequency and potential recursive effects.
- [x] Refined PMDKBackend implementation (error handling, type hints)
- [ ] Test and validate
- [ ] Document and release
- Refactoring of `PMDKBackend` will proceed next.
  - Refactored `PMDKBackend` to use the `pmemobj` Python binding.
  - Implemented pool creation/opening, persistent dictionary for storage, and transactional puts.
  - Added `close()` method and basic error handling.
  - TODOs remain for specific PMDK error handling (e.g., OOM) and require testing on actual PMEM/CXL hardware.
- [ ] Complete PMDK Backend Implementation (Expanded)
  - [ ] **1.1:** Set up PMDK Pool Management System
  - [ ] **1.2:** Implement Basic PMDK Object Storage Structure
  - [ ] **1.3:** Implement Transactional Put Operations
  - [ ] **1.4:** Implement Transactional Get Operations
  - [ ] **1.5:** Implement Error Handling and Resource Management
  - [ ] **Review LeCAR/TieredBackend Scoring & Eviction Logic**
    - [ ] **1:** Review `LeCAR._get_lru_score` (index errors, division by zero, normalization).
    - [ ] **2:** Review `LeCAR._get_lfu_score` (division by zero, normalization).
    - [ ] **3:** Review `LeCAR._get_score` (potential `None` values, type errors, clamping).
    - [ ] **4:** Review `LeCAR.select_victim` (empty lists, `min` key usage, score comparison issues, fallback).
    - [ ] **5:** Review `TieredBackend._find_coldest_fragment` (call to `policy.select_victim`, empty tier handling).
    - [ ] **6:** Review `TieredBackend._update_metadata` & `CacheEntry` creation (data consistency).
    - [ ] **7:** Review `TieredBackend._demote` & `TieredBackend._promote` (state updates, policy calls).
    - [ ] **8:** Summarize findings & identify potential fixes for benchmark error.
- [x] **Expanded Dynamic Policy Tuning:**
    - Implemented dynamic adjustment logic for `promotion_threshold`, `demotion_threshold`, and `watermark` in `TieredBackend`.
    - Added Prometheus metrics and logging for all threshold and watermark changes.
    - Updated code and documentation to reflect these new features.
    - **Comprehensive tests** now validate dynamic adjustment of all thresholds and watermarks (see `tests/test_vua.py`).
    - The `README.md` has been updated to highlight these features for users.
- [x] **Dynamic Policy Tuning Implementation:**
    - Integrated LeCAR (Learning Cache Admission and Replacement) policy into `TieredBackend`
    - Implemented adaptive threshold adjustments based on policy weights
    - Added policy-based promotion/demotion decisions
    - Enhanced victim selection using policy scoring
    - Updated Prometheus metrics to track policy effectiveness
    - Completed integration of policy feedback loops with threshold adjustments
- [x] **Policy Configuration Enhancement:**
    - Added `PolicyConfig` dataclass for structured configuration.
    - Implemented runtime configuration updates via `update_config` method.
    - Made policy parameters (learning rate, exploration, weights, TTL) configurable.
    - Included basic `policy_config_demo.py` example.
- [x] **Advanced Metrics & Monitoring (Policy Level):**
    - Added Prometheus metrics for policy weights, ghost cache stats, and exploration.
    - Implemented score distribution tracking using Histograms.
    - Integrated policy eviction tracking (`policy.evict`) into `TieredBackend`.
- [x] **Testing & Validation (LeCAR Unit Tests):**
    - Created `tests/test_policy.py`.
    - Implemented unit tests covering initialization, scoring, decisions, victim selection, learning, ghost cache, and configuration updates for `LeCAR`.
- [x] **Policy Tuning & Optimization (Setup):**
    - Removed redundant thresholds from `TieredBackend` config.
    - Made internal policy thresholds (`base_promotion_score_threshold`, `base_demotion_score_threshold`) configurable via `PolicyConfig`.
    - Refactored `TieredBackend._feedback_adjust_thresholds` to target `PolicyConfig` parameters.
    - Implemented initial heuristics for tuning exploration rate and score thresholds.
- [x] **Policy Tuning & Optimization (Benchmarking Script):**
    - Created `scripts/benchmark_policy.py` with argparse for configuration.
    - Implemented workload generation (random, sequential, zipfian).
    - Added performance metrics collection (time, OPS, hit rate, latency).
    - Included CSV output for results logging.

## Colloid-Inspired Tiering Algorithm Outline

### Core Principle
- Dynamically balance hot (frequently accessed) data across memory tiers (GPU, DRAM, CXL/PMEM, Storage) based on access frequency and estimated latency, minimizing overload and maximizing performance.

### Data Structures
- Tier Metadata Table: For each cache fragment, track:
  - Current tier (e.g., GPU, DRAM, PMEM, Storage)
  - Access count or recency (e.g., LRU, LFU, or custom)
  - Last access timestamp
  - Estimated access latency for each tier
- Tier Capacity/Pressure: Track current usage and capacity for each tier.

### Algorithm Steps
- **On Access (get):**
  - Update access count/recency for the fragment.
  - If in a lower tier and higher tier has capacity, promote fragment.
  - If higher tier is full, evict/demote coldest fragment, then promote hot fragment.
- **On Insert (put):**
  - Place new fragments in highest available tier.
  - If tier is full, evict/demote coldest fragment.
- **Eviction Policy:**
  - Use LRU, LFU, or hybrid. Optionally use latency estimates for prioritization.
- **Tier Pressure Management:**
  - Monitor usage/pressure. If overloaded, proactively demote cold fragments.

### Promotion/Demotion Triggers
- Promotion: On access, if fragment is hot and higher tier is available.
- Demotion: On insert or when tier is full, evict coldest fragment to next lower tier.

### Metrics and Feedback
- Track hit/miss rates and access latencies per tier.
- Optionally use feedback to adjust thresholds dynamically.

### Integration Points
- Implement in TieredBackend class.
- Use a metadata table (e.g., Python dict) to track fragment state.
- Integrate with put, get, and exists methods.

### Tiering Policy Decisions
- **Capacity Measurement:**
  - Use both fragment count and bytes to determine if a tier is full.
  - If explicit limits are not set, query the framework/runtime (e.g., PyTorch for GPU RAM) to determine available memory and set capacity accordingly.
- **Promotion Policy:**
  - Promotion requires multiple accesses (not just a single access) to avoid redundant promotions from LLM tokenization patterns.
- **Demotion Policy:**
  - Demotion occurs on eviction (when a tier is full and a new fragment must be inserted).
  - Proactive demotion is also triggered when usage exceeds a tunable watermark (e.g., 90% full).
- **Thresholds:**
  - Thresholds are dynamic and adaptive, using heuristics to adjust based on observed access patterns (e.g., hit/miss rates, access frequency distributions).

Prototyping of the metadata table and tier management logic will proceed with these policies.

### Metadata Schema for Tiered Cache Management
- **Always present:**
  - `tier`: int — Index of the current tier (0 = fastest, N = slowest)
  - `access_count`: int — Number of accesses since last promotion/demotion or insertion
  - `last_access`: float — Timestamp of the last access
  - `size_bytes`: int — Size of the fragment in bytes (data + tokens)
  - `insert_time`: float — Timestamp when the fragment was first inserted
- **Advanced fields (triggered as cache fills):**
  - `history`: list[float] — Recent access timestamps (for moving average, burst detection, etc.)
  - `promotion_threshold`: int — Per-tier (default), can be overridden per-fragment if needed
  - `demotion_threshold`: int — Per-tier (default), can be overridden per-fragment if needed
  - `latency_estimates`: dict — Estimated access latency for each tier
  - `eviction_score`: float — Calculated score for fine-grained eviction/promotion
  - `snoop_filter`: dict — Info about adjacent tiers (capacity, readiness)
- **Activation:**
  - Advanced fields are only populated and used when the cache is approaching or exceeding a configurable fill watermark (e.g., 80–90% full).
  - This enables lightweight operation when the cache is mostly empty, and fine-grained control as pressure increases.
- **Thresholds:**
  - Start per-tier (configurable in `tier_configs`).
  - Allow for future adaptation to per-fragment or global if testing shows benefit.
- **Snoop Filter:**
  - Each fragment's metadata (or the tier manager) can query the tier above and below for current usage/capacity and readiness to accept promoted/demoted fragments.
  - This enables smarter decisions for promotion/demotion.

---

Prototyping of this logic will begin next.

## Next Steps: Dynamic Policy Tuning Refinement

### Current Status
- Base LeCAR policy implementation complete with configuration and advanced metrics.
- Unit tests for LeCAR policy logic are in place.
- Feedback loop mechanism implemented to tune policy parameters.
- Initial benchmarking script (`scripts/benchmark_policy.py`) created.

### Remaining Tasks
1.  **Policy Tuning & Optimization (Refinement & Benchmarking)**
    - [ ] **Run Benchmarks:** Execute `benchmark_policy.py` with various configurations (workloads, tiers, policy params).
    - [ ] **Analyze Results:** Study the output CSV to understand performance trade-offs and policy behavior.
    - [ ] **Refine Heuristics:** Update tuning logic in `_feedback_adjust_thresholds` based on benchmark analysis.
    - [ ] **Adaptive Exploration (Optional):** Investigate/implement more advanced adaptation for exploration rate.
    - [ ] **Workload-Specific Tuning (Optional):** Explore methods for detecting and adapting to different workload patterns.
2.  **Testing & Validation (Integration)**
    - [ ] Create integration tests for `TieredBackend` with `LeCAR` focusing on the dynamic tuning behavior over time.
    - [ ] Test edge cases related to tuning (e.g., rapid changes, hitting bounds).
3.  **Documentation & Examples**
    - [ ] Document policy configuration options and tuning heuristics.
    - [ ] Provide example configurations optimized for different scenarios (e.g., high-hit-rate, high-churn).
    - [ ] Add monitoring/dashboard setup guides for policy-specific metrics.
    - [ ] Include performance tuning recommendations based on benchmarking.

### Implementation Priority
1. Policy Tuning & Optimization (Run Benchmarks & Analyze)
2. Testing & Validation (Integration for tuning)
3. Documentation & Examples

---

Execution of Benchmarking runs and analysis will proceed next.

## Next Steps: Real PMDK Integration

### Objectives
- Replace the simulated file-based logic in `