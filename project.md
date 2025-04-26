# VUA Tiering Enhancement & PMDK/CXL Integration

## Overview / Background

VUA currently stores key-value cache data on the filesystem. To support next-generation memory architectures and improve performance, we plan to add tiered storage capabilities, including support for CXL-attached memory and persistent memory via Intel PMDK.

## Objectives
- Enable VUA to store and retrieve cache data across multiple memory/storage tiers (DRAM, CXL/PMEM, disk).
- Integrate with Intel PMDK to leverage persistent memory and CXL memory types.
- Provide a configurable tiering policy (e.g., LRU, size, access frequency).
- Maintain compatibility with existing file-based storage.

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
- [ ] Draft design doc and create project.md (in progress)
- [ ] Research PMDK Python integration
- [ ] Refactor VUA for backend abstraction
- [ ] Prototype PMDK backend
- [ ] Implement tiering logic
- [ ] Test and validate
- [ ] Document and release 