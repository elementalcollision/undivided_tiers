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
- [x] Refined PMDKBackend implementation (error handling, type hints)
- [ ] Test and validate
- [ ] Document and release
- Refactoring of `PMDKBackend` will proceed next.
  - Refactored `PMDKBackend` to use the `pmemobj` Python binding.
  - Implemented pool creation/opening, persistent dictionary for storage, and transactional puts.
  - Added `close()` method and basic error handling.
  - TODOs remain for specific PMDK error handling (e.g., OOM) and require testing on actual PMEM/CXL hardware.
- [x] **Expanded Dynamic Policy Tuning:**
    - Implemented dynamic adjustment logic for `promotion_threshold`, `demotion_threshold`, and `watermark` in `TieredBackend`.
    - Added Prometheus metrics and logging for all threshold and watermark changes.
    - Updated code and documentation to reflect these new features.
    - **Comprehensive tests** now validate dynamic adjustment of all thresholds and watermarks (see `tests/test_vua.py`).
    - The `README.md` has been updated to highlight these features for users.

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

## Next Steps: Dynamic Policy Tuning

### Objectives
- Implement feedback loops to dynamically adjust promotion/demotion thresholds and watermarks based on observed metrics (e.g., hit/miss rates, tier pressure, access patterns).
- Experiment with different scoring functions for eviction and promotion (e.g., weighted LRU/LFU, recency/frequency hybrids).
- Explore adding a learning or heuristic module to predict hot/cold fragments and adapt policies in real time.
- Ensure all policy parameters are configurable and, where possible, tunable at runtime.
- Document the impact of dynamic tuning on cache efficiency and system performance.

---

Research and code generation for dynamic policy tuning will proceed next.

## Next Steps: Real PMDK Integration

### Objectives
- Replace the simulated file-based logic in `src/vua/backend.py::PMDKBackend` with calls to a real PMDK Python binding (likely `py-pmemobj`).
- Implement PMDK pool creation/opening (`pmemobj_create`/`pmemobj_open`) in the backend's constructor.
- Utilize persistent memory allocation (e.g., `pmemobj_alloc`) and transactional updates within the PMDK pool for `put`, `get`, and `exists` operations, using the `group_hash` as a key.
- Add robust error handling for PMDK-specific issues (pool errors, allocation failures).
- Ensure the implementation remains modular, encapsulating PMDK specifics within `PMDKBackend` and adhering to the `StorageBackend` interface, allowing for potential future integration of other CXL SDKs via separate backend classes.
- Add necessary configuration options (pool path, size, layout name) for the `PMDKBackend`.

---

Refactoring of `PMDKBackend` will proceed next.

## Next Steps: Prometheus Integration for Metrics

### Objectives
- Export key TieredBackend metrics (hits, misses, promotions, demotions, evictions, tier usage, dynamic thresholds) to Prometheus.
- Enable real-time monitoring and offline analysis of cache performance and policy effectiveness.

### Approach
- Utilize the `prometheus-client` Python library.
- Define Prometheus Counters and Gauges within `TieredBackend`, labeled by tier, for aggregated metrics (hits, misses, usage, thresholds, etc.).
- Define Prometheus `Histogram` or `Summary` metrics to track distributions of key per-fragment metadata (e.g., fragment age, access counts, size) per tier.
- Calculate and update aggregated/distribution metrics periodically.
- Update metrics within relevant backend methods.
- Expose metrics via an optional HTTP endpoint for Prometheus scraping.
- **Note:** Avoid exporting raw per-fragment metadata directly to Prometheus due to cardinality concerns; use aggregated statistics and distributions instead.

### Implementation Steps
1. Add `prometheus-client` as an optional dependency.
2. Define Prometheus metric objects (Counters, Gauges, Histograms/Summaries) in `TieredBackend` constructor.
3. Increment/set counters and basic gauges in relevant backend methods.
4. Implemented logic to calculate and update aggregated/distribution metrics (Histograms/Summaries, advanced metadata counts, avg scores) periodically (e.g., in `export_metrics`).
5. Add option to example script to start Prometheus HTTP server.
6. Update documentation.

---

Implementation of Prometheus integration (including aggregated metrics) is complete.

- [x] Implement Prometheus Integration for Metrics (within TieredBackend)
    - Defined Counters, Gauges, Histograms for key metrics.
    - Integrated metric updates into backend operations.
    - Added helper methods for exporting aggregates and starting HTTP server.
    - Remaining: Update example scripts, documentation. 