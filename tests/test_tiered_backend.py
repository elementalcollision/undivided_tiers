import unittest
import time
import os
import shutil
import logging
from typing import List

# Assuming backend.py is in src/vua relative to the workspace root
from src.vua.backend import (
    TieredBackend,
    PolicyConfig,
    LeCAR, # Use LeCAR policy for integration tests
    MockGPURAMBackend,
    MockPMDKBackend,
    FileSystemBackend,
    CacheEntry
)

# Configure logging for tests (optional)
# logging.basicConfig(level=logging.DEBUG)

class TestTieredBackendIntegration(unittest.TestCase):

    def setUp(self):
        """Set up a TieredBackend with mock backends and LeCAR policy."""
        self.test_dir = "/tmp/test_vua_tiered_backend"
        # Clean up previous test runs
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        self.backends = [
            MockGPURAMBackend(),  # Tier 0: Fast mock RAM
            MockPMDKBackend(),    # Tier 1: Slower mock PMEM
            FileSystemBackend(os.path.join(self.test_dir, "disk_tier")) # Tier 2: Slowest disk
        ]
        
        self.tier_configs = [
            # Tier 0: Small capacity for easy testing, no thresholds needed here as policy decides
            {'name': 'ram', 'capacity_count': 3},
            # Tier 1: Medium capacity
            {'name': 'pmem', 'capacity_count': 5},
            # Tier 2: Large capacity, no promotion from here
            {'name': 'disk', 'capacity_count': 10}
        ]
        
        # Use default LeCAR policy for these tests
        self.policy_config = PolicyConfig(exploration_rate=0) # Disable exploration for predictable tests
        self.policy = LeCAR(self.policy_config)
        
        self.tiered_backend = TieredBackend(
            backends=self.backends,
            tier_configs=self.tier_configs,
            policy=self.policy,
            adjustment_interval=1000 # High interval to avoid interference during tests
        )

        # Helper to create data
        self.create_data = lambda i: (f"data_{i}".encode(), f"tokens_{i}".encode())

    def tearDown(self):
        """Clean up test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_put_get_simple(self):
        """Test basic put and get in the top tier."""
        data, tokens = self.create_data(1)
        group_hash = "simple_1"
        
        # Put should go to tier 0
        self.tiered_backend.put(group_hash, data, tokens)
        self.assertTrue(self.backends[0].exists(group_hash))
        self.assertFalse(self.backends[1].exists(group_hash))
        self.assertFalse(self.backends[2].exists(group_hash))
        self.assertEqual(self.tiered_backend.metadata[group_hash]['tier'], 0)
        
        # Get should retrieve from tier 0
        retrieved_data, retrieved_tokens = self.tiered_backend.get(group_hash)
        self.assertEqual(retrieved_data, data)
        self.assertEqual(retrieved_tokens, tokens)
        # Check policy was updated (access count should increase)
        self.assertEqual(self.policy._lfu_scores[group_hash], 1)

    # --- Demotion Tests ---
    def test_put_fills_tier0_demotes_to_tier1(self):
        """Test items demote from tier 0 to tier 1 when tier 0 is full."""
        tier0_capacity = self.tier_configs[0]['capacity_count'] # Should be 3
        
        # Fill tier 0
        hashes = []
        for i in range(tier0_capacity):
            group_hash = f"fill_t0_{i}"
            hashes.append(group_hash)
            data, tokens = self.create_data(i)
            self.tiered_backend.put(group_hash, data, tokens)
            self.assertTrue(self.backends[0].exists(group_hash), f"{group_hash} should be in tier 0")
            self.assertEqual(self.tiered_backend.metadata[group_hash]['tier'], 0)
        
        self.assertEqual(len(self.tiered_backend.metadata), tier0_capacity)
        self.assertEqual(self.tiered_backend.tier_usage[0]['count'], tier0_capacity)
        
        # Put one more item - should cause demotion
        data_extra, tokens_extra = self.create_data(tier0_capacity)
        hash_extra = f"fill_t0_{tier0_capacity}"
        self.tiered_backend.put(hash_extra, data_extra, tokens_extra)
        
        # The new item should be in tier 0
        self.assertTrue(self.backends[0].exists(hash_extra))
        self.assertEqual(self.tiered_backend.metadata[hash_extra]['tier'], 0)
        
        # Check tier 0 usage (should still be at capacity)
        self.assertEqual(self.tiered_backend.tier_usage[0]['count'], tier0_capacity)
        
        # One item should have been demoted to tier 1
        tier1_hashes = [h for h, m in self.tiered_backend.metadata.items() if m['tier'] == 1]
        self.assertEqual(len(tier1_hashes), 1, f"Expected 1 item in tier 1, found {len(tier1_hashes)}")
        demoted_hash = tier1_hashes[0]
        self.assertTrue(self.backends[1].exists(demoted_hash), f"Demoted hash {demoted_hash} not found in tier 1 backend")
        self.assertFalse(self.backends[0].exists(demoted_hash), f"Demoted hash {demoted_hash} should not be in tier 0 backend")
        self.assertEqual(self.tiered_backend.tier_usage[1]['count'], 1)
        
        # Verify the policy's select_victim was likely called (indirect check)
        # This relies on the default behavior of LeCAR evicting the LRU item initially
        # The first item put (hashes[0]) should be the demoted one if no accesses happened
        self.assertEqual(demoted_hash, hashes[0])

    def test_demotion_cascades_to_tier2(self):
        """Test items demote from tier 1 to tier 2 when tier 1 is full."""
        tier0_cap = self.tier_configs[0]['capacity_count'] # 3
        tier1_cap = self.tier_configs[1]['capacity_count'] # 5
        total_cap_01 = tier0_cap + tier1_cap # 8
        
        # Fill tier 0 and tier 1
        hashes = []
        for i in range(total_cap_01):
            group_hash = f"cascade_{i}"
            hashes.append(group_hash)
            data, tokens = self.create_data(i)
            self.tiered_backend.put(group_hash, data, tokens)
            
        # Verify counts
        self.assertEqual(self.tiered_backend.tier_usage[0]['count'], tier0_cap)
        self.assertEqual(self.tiered_backend.tier_usage[1]['count'], tier1_cap)
        self.assertEqual(self.tiered_backend.tier_usage[2]['count'], 0)
        self.assertEqual(len(self.tiered_backend.metadata), total_cap_01)

        # Put one more item - should cause cascade demotion to tier 2
        hash_extra = f"cascade_{total_cap_01}"
        data_extra, tokens_extra = self.create_data(total_cap_01)
        self.tiered_backend.put(hash_extra, data_extra, tokens_extra)
        
        # New item in tier 0
        self.assertTrue(self.backends[0].exists(hash_extra))
        self.assertEqual(self.tiered_backend.metadata[hash_extra]['tier'], 0)
        
        # Check tier counts (should remain at capacity for 0 and 1)
        self.assertEqual(self.tiered_backend.tier_usage[0]['count'], tier0_cap)
        self.assertEqual(self.tiered_backend.tier_usage[1]['count'], tier1_cap)
        
        # One item should now be in tier 2
        tier2_hashes = [h for h, m in self.tiered_backend.metadata.items() if m['tier'] == 2]
        self.assertEqual(len(tier2_hashes), 1, f"Expected 1 item in tier 2, found {len(tier2_hashes)}")
        demoted_hash = tier2_hashes[0]
        self.assertTrue(self.backends[2].exists(demoted_hash), f"Demoted hash {demoted_hash} not found in tier 2 backend")
        self.assertFalse(self.backends[1].exists(demoted_hash), f"Demoted hash {demoted_hash} should not be in tier 1 backend")
        self.assertEqual(self.tiered_backend.tier_usage[2]['count'], 1)
        
        # Verify the twice-demoted item (should be the very first item put)
        self.assertEqual(demoted_hash, hashes[0])

    # --- Promotion Tests ---
    def test_get_promotes_item_from_tier1_to_tier0(self):
        """Test accessing an item in tier 1 promotes it to tier 0."""
        # Put item directly into tier 1 (by filling tier 0 first)
        tier0_capacity = self.tier_configs[0]['capacity_count']
        for i in range(tier0_capacity):
            self.tiered_backend.put(f"fill_{i}", *self.create_data(i))
        
        hash_to_promote = "promo_1"
        data, tokens = self.create_data(100)
        self.tiered_backend.put(hash_to_promote, data, tokens)
        
        # Verify it's initially in tier 1
        self.assertFalse(self.backends[0].exists(hash_to_promote))
        self.assertTrue(self.backends[1].exists(hash_to_promote))
        self.assertEqual(self.tiered_backend.metadata[hash_to_promote]['tier'], 1)
        
        # Access the item - policy should decide to promote
        # Tier 1 promo threshold is 2, LeCAR default score needs to be > 0.6
        # Let's ensure score is high enough by accessing multiple times or mocking policy
        # For simplicity, let's just access it once, default config should promote
        retrieved_data, retrieved_tokens = self.tiered_backend.get(hash_to_promote)
        self.assertEqual(retrieved_data, data)

        # Check if promotion occurred
        # Need a slight delay or re-check as promotion might happen slightly after get returns
        # In mock backends, it should be immediate
        self.assertTrue(self.backends[0].exists(hash_to_promote), "Item should be promoted to tier 0")
        self.assertFalse(self.backends[1].exists(hash_to_promote), "Item should be removed from tier 1 after promotion")
        self.assertEqual(self.tiered_backend.metadata[hash_to_promote]['tier'], 0)
        self.assertEqual(self.tiered_backend.tier_usage[0]['count'], tier0_capacity) # Tier 0 should be full again
        # Tier 1 count should decrease by 1 (promo) and increase by 1 (demotion from tier 0)
        self.assertEqual(self.tiered_backend.tier_usage[1]['count'], 1) 

    def test_promotion_cascades_demotion(self):
        """Test promoting an item forces demotion from the target tier if full."""
        tier0_cap = self.tier_configs[0]['capacity_count'] # 3
        
        # 1. Fill tier 0 completely
        fill_hashes = []
        for i in range(tier0_cap):
            f_hash = f"fill_{i}"
            fill_hashes.append(f_hash)
            self.tiered_backend.put(f_hash, *self.create_data(i))
        self.assertEqual(self.tiered_backend.tier_usage[0]['count'], tier0_cap)
        
        # 2. Put item to be promoted into tier 1
        hash_to_promote = "promo_cascade"
        data, tokens = self.create_data(100)
        self.tiered_backend.put(hash_to_promote, data, tokens) # This forces one item from tier 0 to tier 1
        original_demoted_hash = fill_hashes[0] # fill_0 should now be in tier 1
        self.assertEqual(self.tiered_backend.metadata[hash_to_promote]['tier'], 0) # New item goes to tier 0
        self.assertEqual(self.tiered_backend.metadata[original_demoted_hash]['tier'], 1) # fill_0 demoted
        self.assertTrue(self.backends[1].exists(original_demoted_hash))
        # Now manually move the item-to-promote to tier 1 for the test scenario
        self.tiered_backend.metadata[hash_to_promote]['tier'] = 1
        self.tiered_backend.backends[1].put(hash_to_promote, data, tokens)
        self.tiered_backend.backends[0].put(hash_to_promote, b"", b"") # Clear from tier 0
        self.tiered_backend._update_tier_usage(0, -1, -(len(data)+len(tokens)))
        self.tiered_backend._update_tier_usage(1, 1, len(data)+len(tokens))
        # State: tier0=[f1, f2], tier1=[f0, pC] (pC=promo_cascade) 
        self.assertEqual(self.tiered_backend.tier_usage[0]['count'], tier0_cap -1)
        self.assertEqual(self.tiered_backend.tier_usage[1]['count'], 2)

        # 3. Access item in tier 1 to trigger promotion
        self.tiered_backend.get(hash_to_promote)
        
        # 4. Verify promotion and cascaded demotion
        self.assertTrue(self.backends[0].exists(hash_to_promote)) # Promoted item now in tier 0
        self.assertEqual(self.tiered_backend.metadata[hash_to_promote]['tier'], 0)
        self.assertFalse(self.backends[1].exists(hash_to_promote)) # Should be gone from tier 1
        
        # Tier 0 should be full (promo_cascade + 2 original fillers)
        self.assertEqual(self.tiered_backend.tier_usage[0]['count'], tier0_cap)
        
        # One of the original fillers (fill_1 or fill_2) must have been demoted from tier 0 to tier 1
        tier1_hashes = [h for h, m in self.tiered_backend.metadata.items() if m['tier'] == 1]
        # Tier 1 should now contain original_demoted_hash (fill_0) + one newly demoted hash
        self.assertEqual(len(tier1_hashes), 2)
        self.assertIn(original_demoted_hash, tier1_hashes) # fill_0 should still be there
        # Find the newly demoted hash
        newly_demoted = [h for h in tier1_hashes if h != original_demoted_hash][0]
        self.assertIn(newly_demoted, [fill_hashes[1], fill_hashes[2]]) # Should be fill_1 or fill_2
        self.assertTrue(self.backends[1].exists(newly_demoted))

    # --- Eviction Tests ---
    def test_eviction_from_last_tier(self):
        """Test items are evicted entirely when demoted from the last tier."""
        tier0_cap = self.tier_configs[0]['capacity_count'] # 3
        tier1_cap = self.tier_configs[1]['capacity_count'] # 5
        tier2_cap = self.tier_configs[2]['capacity_count'] # 10
        total_cap = tier0_cap + tier1_cap + tier2_cap # 18
        
        # Fill all tiers
        hashes = []
        for i in range(total_cap):
            group_hash = f"evict_{i}"
            hashes.append(group_hash)
            data, tokens = self.create_data(i)
            self.tiered_backend.put(group_hash, data, tokens)
            
        # Verify counts
        self.assertEqual(self.tiered_backend.tier_usage[0]['count'], tier0_cap)
        self.assertEqual(self.tiered_backend.tier_usage[1]['count'], tier1_cap)
        self.assertEqual(self.tiered_backend.tier_usage[2]['count'], tier2_cap)
        self.assertEqual(len(self.tiered_backend.metadata), total_cap)
        
        # Mock policy.evict to check if it gets called
        evicted_from_policy = []
        def mock_evict(entry):
            evicted_from_policy.append(entry.group_hash)
            # Call original evict if needed for ghost cache testing
            # super(type(self.policy), self.policy).evict(entry)
        original_evict = self.policy.evict # Keep original if needed later
        self.policy.evict = mock_evict
        
        # Put one more item - should cause eviction from tier 2
        hash_extra = f"evict_{total_cap}"
        data_extra, tokens_extra = self.create_data(total_cap)
        self.tiered_backend.put(hash_extra, data_extra, tokens_extra)

        # Restore original evict method
        self.policy.evict = original_evict

        # Verify counts (tiers 0, 1, 2 should be at capacity)
        self.assertEqual(self.tiered_backend.tier_usage[0]['count'], tier0_cap)
        self.assertEqual(self.tiered_backend.tier_usage[1]['count'], tier1_cap)
        self.assertEqual(self.tiered_backend.tier_usage[2]['count'], tier2_cap)
        
        # Verify total metadata count hasn't increased
        self.assertEqual(len(self.tiered_backend.metadata), total_cap)
        
        # Verify the first item put (hashes[0]) was evicted
        evicted_hash = hashes[0]
        self.assertNotIn(evicted_hash, self.tiered_backend.metadata)
        self.assertFalse(self.backends[0].exists(evicted_hash))
        self.assertFalse(self.backends[1].exists(evicted_hash))
        self.assertFalse(self.backends[2].exists(evicted_hash))
        
        # Verify policy.evict was called with the correct hash
        self.assertEqual(len(evicted_from_policy), 1)
        self.assertEqual(evicted_from_policy[0], evicted_hash)

    # --- More integration tests will be added below ---

if __name__ == '__main__':
    unittest.main() 