import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import unittest
import torch
import logging
import tempfile
import shutil
from unittest.mock import patch, MagicMock, call

from vua.core import VUA, VUAConfig
from vua.serdes import tensor_to_bytes, bytes_to_tensor
from vua.backend import PMDKBackend, TieredBackend, MockPMDKBackend, FileSystemBackend

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

def generate_rand_kvcache(n_layers, seq_len, batch_size, num_heads, head_size):
    layers = []
    for i in range(0, n_layers):
        s = []
        for kv in [0, 1]:
            size = (batch_size, num_heads, seq_len, head_size)
            t = torch.randn(size, dtype=torch.float16)
            s.append(t)
        layers.append(s)
    return layers


class TestVUAConfig(unittest.TestCase):
    def test_tokens_to_paths(self):
        # Create a tensor of tokens that is divisible by split_factor
        tokens = torch.arange(VUAConfig.split_factor * 2)
        paths = VUAConfig.tokens_to_paths(tokens)
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(isinstance(p, str) for p in paths),
            "Each token group should be converted to a string path component")

    def test_trim_to_split_factor(self):
        # Create a tensor of tokens that isn't divisible by split_factor
        tokens = torch.arange(100)
        trimmed = VUAConfig.trim_to_split_factor(tokens)
        self.assertEqual(len(trimmed) % VUAConfig.split_factor, 0,
            "Trimmed tensor length should be divisible by split_factor")


    def test_put_get(self):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            vua_path = os.path.join(temp_dir, 'vua')
            os.mkdir(vua_path)

            # Create a tensor of tokens that isn't divisible by split_factor
            logger.info(f"VUAPath: {vua_path}")
            cache = VUA(VUAConfig, vua_path)

            nr_tokens = 3000
            tokens = torch.randint(low=0, high=0xffff, size=(1, nr_tokens), dtype=torch.uint16)
            half_tokens = tokens[:, :nr_tokens//2]
            quarter_tokens = tokens[:, :nr_tokens//4]
            trimmed = VUAConfig.trim_to_split_factor(tokens)
            trimmed_half = VUAConfig.trim_to_split_factor(half_tokens)
            trimmed_quarter = VUAConfig.trim_to_split_factor(quarter_tokens)
            trimmed_quarter_plus_other = VUAConfig.trim_to_split_factor(torch.cat([quarter_tokens, tokens[:, 3*nr_tokens//4:nr_tokens]], dim=1))
            self.assertEqual(len(trimmed_half) % VUAConfig.split_factor, 0,
                "Trimmed tensor length should be divisible by split_factor")

            kvcache = generate_rand_kvcache(32, half_tokens.size(1), 1, 32, 16)
            cache.put(trimmed_half, kvcache)

            logger.info("---- Doing a get with a double length query")
            # The prefix with which we originally did 'put', yields half that prefix.
            res = cache.get_closest(trimmed, device="cuda:0")
            self.assertEqual(torch.equal(res.tokens.to("cpu"), trimmed_half), True)

            logger.info("---- :: Doing a get with half of it, only gets us the half")
            res = cache.get_closest(trimmed_quarter, device="cuda:0")
            self.assertEqual(torch.equal(res.tokens.to("cpu"), trimmed_quarter), True)

            logger.info("---- :: Doing a get with half of it plus other tokens, only gets us the half")
            res = cache.get_closest(trimmed_quarter_plus_other, device="cuda:0")
            self.assertEqual(torch.equal(res.tokens.to("cpu"), trimmed_quarter), True)

            logger.info("---- :: Batched get")
            res = cache.get_closest([trimmed_quarter,
                                     trimmed_quarter_plus_other], device="cuda:0")
            self.assertNotEqual(res[0], None)
            self.assertNotEqual(res[1], None)

            logger.info("---- :: Batched put")

            batched_seqs = torch.randint(low=0, high=0xffff,
                                         size=(3, cache.config().split_factor * 10), dtype=torch.uint16)
            batched_kvcache = generate_rand_kvcache(24, batched_seqs.size(1), batched_seqs.size(0), 8, 16)
            cache.put(batched_seqs, batched_kvcache)

    def test_put_get_mock_backend(self):
        """Test VUA put/get with the MockPMDKBackend."""
        mock_backend = MockPMDKBackend()
        cache = VUA(VUAConfig, ".", backend=mock_backend) # root_path is irrelevant for mock
        tokens = torch.arange(VUAConfig.split_factor * 2)
        kvcache = generate_rand_kvcache(1, VUAConfig.split_factor * 2, 1, 1, 1)
        cache.put(tokens, kvcache)
        res = cache.get_closest(tokens, device="cpu")
        self.assertIsNotNone(res)
        # Further checks could compare res.data with original kvcache (after slicing/serdes)

    def test_put_get_tiered_backend(self):
        """Test VUA put/get with the TieredBackend using mocks."""
        tier_configs = [
            {'name': 'tier0', 'capacity_count': 1, 'capacity_bytes': 1000},
            {'name': 'tier1', 'capacity_count': 1, 'capacity_bytes': 1000},
        ]
        backends = [MockPMDKBackend(), MockPMDKBackend()]
        tiered_backend = TieredBackend(backends, tier_configs)
        cache = VUA(VUAConfig, ".", backend=tiered_backend)
        tokens1 = torch.arange(VUAConfig.split_factor)
        tokens2 = torch.arange(VUAConfig.split_factor, VUAConfig.split_factor * 2)
        kvcache1 = generate_rand_kvcache(1, VUAConfig.split_factor, 1, 1, 1)
        kvcache2 = generate_rand_kvcache(1, VUAConfig.split_factor, 1, 1, 1)
        cache.put(tokens1, kvcache1) # Goes to tier 0
        cache.put(tokens2, kvcache2) # Should demote tokens1 to tier 1
        res1 = cache.get_closest(tokens1, device="cpu")
        res2 = cache.get_closest(tokens2, device="cpu")
        self.assertIsNotNone(res1)
        self.assertIsNotNone(res2)
        self.assertEqual(tiered_backend.metadata['testhash0']['tier'], 1) # Check if demoted
        self.assertEqual(tiered_backend.metadata['testhash1']['tier'], 0) # Check if in top tier

    def test_repair_symlinks(self):
        """Test the repair_symlinks functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vua_path = os.path.join(temp_dir, 'vua')
            os.mkdir(vua_path)
            cache = VUA(VUAConfig, vua_path)
            tokens = torch.arange(VUAConfig.split_factor * 3)
            kvcache = generate_rand_kvcache(1, VUAConfig.split_factor * 3, 1, 1, 1)
            cache.put(tokens, kvcache)
            # Manually break a symlink
            path_components = VUAConfig.tokens_to_paths(tokens)
            link_path = os.path.join(vua_path, path_components[1], "parent")
            if os.path.exists(link_path):
                os.remove(link_path)
            self.assertFalse(os.path.exists(link_path))
            # Run repair
            cache.repair_symlinks()
            # Check if symlink was restored
            self.assertTrue(os.path.islink(link_path))
            target = os.readlink(link_path)
            expected_target = os.path.join("..", path_components[0])
            self.assertEqual(target, expected_target)


class TestSerdes(unittest.TestCase):
    def test_tensor_serialization(self):
        # Create a random tensor and test that serializing and deserializing
        # returns a similar tensor
        x = torch.randn(100, 5, 10000)
        b = tensor_to_bytes(x, "tensor")
        x_rec = bytes_to_tensor(b, "tensor")
        self.assertTrue(torch.allclose(x, x_rec.float(), atol=1e-6),
                        "Deserialized tensor does not match the original")


# Constants for PMDK tests
TEST_POOL_PATH = "/test/pmdk_pool.pmem"
TEST_POOL_SIZE = 1024 * 1024 * 32 # 32MB

class TestPMDKBackend(unittest.TestCase):
    # We mock pmemobj for most tests, so setUp/tearDown manage mocks not files
    def setUp(self):
        # Patch pmemobj for the duration of the test class if it's None
        # If pmemobj *is* available, tests might try to use it unless explicitly mocked per-test
        self.pmemobj_patcher = patch('vua.backend.pmemobj', spec=True) # Use spec for better mocking
        self.mock_pmemobj = self.pmemobj_patcher.start()
        # Mock the error class if it exists
        self.pmemerror_patcher = patch('vua.backend.PmemError', spec=True)
        MockPmemErrorType = self.pmemerror_patcher.start() # Get the mocked type
        # Define a local dummy exception if PmemError was None
        if MockPmemErrorType is None:
            class DummyPmemError(Exception): pass
            self.MockPmemError = DummyPmemError
        else:
            # If PmemError was mocked, use the mock type for isinstance checks etc.
            # For assertRaises, we need an actual exception type. Let's ensure MockPmemErrorType
            # behaves like a type. If it's a MagicMock, we might need a different approach,
            # but spec=True should make it reasonable.
            # We'll assume the mocked type works with assertRaises for now.
            self.MockPmemError = MockPmemErrorType

        # Prevent tests from trying to access the filesystem unless specifically intended
        self.os_path_exists_patcher = patch('os.path.exists')
        self.mock_os_path_exists = self.os_path_exists_patcher.start()

    def tearDown(self):
        self.pmemobj_patcher.stop()
        self.pmemerror_patcher.stop()
        self.os_path_exists_patcher.stop()

    # --- Tests for __init__ (Task 1.1) ---
    def test_init_imports_missing(self):
        """Test that ImportError is raised if pmemobj is None."""
        with patch('vua.backend.pmemobj', None):
             with self.assertRaisesRegex(ImportError, "PMDK pmemobj binding not found"):
                  PMDKBackend(TEST_POOL_PATH)

    def test_init_creates_pool(self):
        """Test pool creation when file doesn't exist and create=True."""
        self.mock_os_path_exists.return_value = False
        mock_pool_instance = MagicMock()
        self.mock_pmemobj.create.return_value = mock_pool_instance

        backend = PMDKBackend(TEST_POOL_PATH, pool_size=TEST_POOL_SIZE, create=True)

        self.mock_os_path_exists.assert_called_once_with(TEST_POOL_PATH)
        self.mock_pmemobj.create.assert_called_once_with(TEST_POOL_PATH, PMDKBackend.LAYOUT_NAME, TEST_POOL_SIZE)
        self.mock_pmemobj.open.assert_not_called()
        # Check if the transaction for root initialization was called
        mock_pool_instance.transaction.assert_called_once()
        self.assertEqual(backend.pool, mock_pool_instance)

    def test_init_opens_existing_pool(self):
        """Test opening an existing pool when create=True."""
        self.mock_os_path_exists.return_value = True
        mock_pool_instance = MagicMock()
        self.mock_pmemobj.open.return_value = mock_pool_instance

        backend = PMDKBackend(TEST_POOL_PATH, create=True)

        self.mock_os_path_exists.assert_called_once_with(TEST_POOL_PATH)
        self.mock_pmemobj.open.assert_called_once_with(TEST_POOL_PATH, PMDKBackend.LAYOUT_NAME)
        self.mock_pmemobj.create.assert_not_called()
        self.assertEqual(backend.pool, mock_pool_instance)

    def test_init_opens_existing_pool_no_create(self):
        """Test opening an existing pool when create=False."""
        self.mock_os_path_exists.return_value = True
        mock_pool_instance = MagicMock()
        self.mock_pmemobj.open.return_value = mock_pool_instance

        backend = PMDKBackend(TEST_POOL_PATH, create=False)

        self.mock_os_path_exists.assert_called_once_with(TEST_POOL_PATH)
        self.mock_pmemobj.open.assert_called_once_with(TEST_POOL_PATH, PMDKBackend.LAYOUT_NAME)
        self.mock_pmemobj.create.assert_not_called()
        self.assertEqual(backend.pool, mock_pool_instance)

    def test_init_raises_filenotfound_if_no_pool_and_create_false(self):
        """Test FileNotFoundError is raised if pool doesn't exist and create=False."""
        self.mock_os_path_exists.return_value = False
        with self.assertRaisesRegex(FileNotFoundError, "PMDK pool file not found and create=False"):
            PMDKBackend(TEST_POOL_PATH, create=False)
        self.mock_os_path_exists.assert_called_once_with(TEST_POOL_PATH)
        self.mock_pmemobj.open.assert_not_called()
        self.mock_pmemobj.create.assert_not_called()

    def test_init_raises_permission_error_on_create(self):
        """Test PermissionError is caught and raised during create."""
        self.mock_os_path_exists.return_value = False
        self.mock_pmemobj.create.side_effect = PermissionError("Test permission denied")
        with self.assertRaisesRegex(PermissionError, "Test permission denied"):
            PMDKBackend(TEST_POOL_PATH, create=True)
        self.mock_pmemobj.create.assert_called_once()

    def test_init_raises_permission_error_on_open(self):
        """Test PermissionError is caught and raised during open."""
        self.mock_os_path_exists.return_value = True
        self.mock_pmemobj.open.side_effect = PermissionError("Test permission denied open")
        with self.assertRaisesRegex(PermissionError, "Test permission denied open"):
            PMDKBackend(TEST_POOL_PATH, create=False)
        self.mock_pmemobj.open.assert_called_once()

    def test_init_raises_pmem_error_on_create(self):
        """Test specific PmemError is caught and raised during create."""
        self.mock_os_path_exists.return_value = False
        # Instantiate the error type we stored in setUp
        test_error = self.MockPmemError("PMDK create failed")
        self.mock_pmemobj.create.side_effect = test_error
        # Use the stored type with assertRaises
        with self.assertRaises(self.MockPmemError) as cm:
            PMDKBackend(TEST_POOL_PATH, create=True)
        # Check the raised exception instance
        self.assertIsInstance(cm.exception, self.MockPmemError)
        self.mock_pmemobj.create.assert_called_once()

    def test_init_raises_pmem_error_on_open(self):
        """Test specific PmemError is caught and raised during open."""
        self.mock_os_path_exists.return_value = True
        # Instantiate the error type
        test_error = self.MockPmemError("PMDK open failed")
        self.mock_pmemobj.open.side_effect = test_error
        # Use the stored type with assertRaises
        with self.assertRaises(self.MockPmemError) as cm:
            PMDKBackend(TEST_POOL_PATH, create=False)
        # Check the raised exception instance
        self.assertIsInstance(cm.exception, self.MockPmemError)
        self.mock_pmemobj.open.assert_called_once()

    def test_close_closes_pool_and_clears_ref(self):
        """Test that close() calls pool.close() and sets self.pool to None."""
        self.mock_os_path_exists.return_value = True
        mock_pool_instance = MagicMock()
        self.mock_pmemobj.open.return_value = mock_pool_instance

        backend = PMDKBackend(TEST_POOL_PATH, create=False)
        self.assertEqual(backend.pool, mock_pool_instance)

        backend.close()

        mock_pool_instance.close.assert_called_once()
        self.assertIsNone(backend.pool)

    def test_close_handles_already_closed(self):
        """Test that close() does nothing if pool is already None."""
        self.mock_os_path_exists.return_value = True
        mock_pool_instance = MagicMock()
        self.mock_pmemobj.open.return_value = mock_pool_instance

        backend = PMDKBackend(TEST_POOL_PATH, create=False)
        backend.pool = None # Simulate already closed

        backend.close() # Should not raise error
        mock_pool_instance.close.assert_not_called()

    def test_close_handles_close_error(self):
        """Test that close() logs error but sets pool to None even if pool.close() fails."""
        self.mock_os_path_exists.return_value = True
        mock_pool_instance = MagicMock()
        # Instantiate the error type
        mock_pool_instance.close.side_effect = self.MockPmemError("Failed to close")
        self.mock_pmemobj.open.return_value = mock_pool_instance

        backend = PMDKBackend(TEST_POOL_PATH, create=False)
        # Use assertLogs to check log messages
        with self.assertLogs(backend.logger, level='ERROR') as log_cm:
            backend.close()
        self.assertIn("Error closing PMDK pool", log_cm.output[0])
        mock_pool_instance.close.assert_called_once()
        self.assertIsNone(backend.pool)

    def test_del_closes_pool(self):
        """Test that the pool is closed when the backend object is deleted."""
        self.mock_os_path_exists.return_value = True
        mock_pool_instance = MagicMock()
        self.mock_pmemobj.open.return_value = mock_pool_instance

        backend = PMDKBackend(TEST_POOL_PATH, create=False)
        mock_pool_ref = backend.pool # Keep a ref to check close call

        del backend # Trigger __del__

        mock_pool_ref.close.assert_called_once()

    # --- Tests for put/get/exists (Task 1.3, 1.4) --- 
    # These tests now use the mocked pmemobj
    def test_put_and_exists(self):
        # Setup mock pool and root dict for this test
        mock_pool = MagicMock()
        mock_root = MagicMock(spec=dict) # Mock PersistentDict
        mock_pool.transaction.return_value.__enter__.return_value = None # Mock transaction context
        mock_pool.root = mock_root
        self.mock_pmemobj.open.return_value = mock_pool
        self.mock_os_path_exists.return_value = True

        backend = PMDKBackend(TEST_POOL_PATH, create=False)
        group_hash = "testhash"
        data = b"testdata"
        tokens = b"testtokens"
        
        # Mock the PersistentBytes class within the scope of put
        with patch('vua.backend.pmemobj.PersistentBytes', side_effect=lambda x: x) as mock_pbytes:
            backend.put(group_hash, data, tokens)
        
        # Check transaction was used
        mock_pool.transaction.assert_called_once()
        # Check PersistentBytes was called for data and tokens
        mock_pbytes.assert_has_calls([call(data), call(tokens)])
        # Check item was added to the mock root dict
        mock_root.__setitem__.assert_called_once_with(group_hash, (data, tokens))

        # Test exists
        mock_root.__contains__.return_value = True # Simulate key exists
        self.assertTrue(backend.exists(group_hash))
        mock_root.__contains__.assert_called_with(group_hash)

    def test_get_returns_correct_data(self):
        # Setup mock pool and root dict
        mock_pool = MagicMock()
        mock_root = MagicMock(spec=dict)
        mock_pool.root = mock_root
        self.mock_pmemobj.open.return_value = mock_pool
        self.mock_os_path_exists.return_value = True
        
        backend = PMDKBackend(TEST_POOL_PATH, create=False)
        group_hash = "testhash2"
        data = b"somedata"
        tokens = b"sometokens"
        
        # Simulate data exists in the pool root
        # Return bytes directly, as PersistentBytes was mocked away in put test
        mock_root.get.return_value = (data, tokens)
        
        result = backend.get(group_hash)
        
        mock_root.get.assert_called_once_with(group_hash)
        self.assertIsNotNone(result)
        data_out, tokens_out = result
        self.assertEqual(data_out, data)
        self.assertEqual(tokens_out, tokens)

    def test_get_returns_none_for_missing(self):
        # Setup mock pool and root dict
        mock_pool = MagicMock()
        mock_root = MagicMock(spec=dict)
        mock_pool.root = mock_root
        self.mock_pmemobj.open.return_value = mock_pool
        self.mock_os_path_exists.return_value = True

        backend = PMDKBackend(TEST_POOL_PATH, create=False)

        # Simulate key not found
        mock_root.get.return_value = None
        mock_root.__contains__.return_value = False

        self.assertIsNone(backend.get("nonexistent"))
        mock_root.get.assert_called_once_with("nonexistent")
        
        self.assertFalse(backend.exists("nonexistent"))
        mock_root.__contains__.assert_called_once_with("nonexistent")


class TestTieredBackend(unittest.TestCase):
    def setUp(self):
        # Use small capacities for fast testing
        self.tier_configs = [
            {'name': 'tier0', 'capacity_count': 2, 'capacity_bytes': 1000, 'promotion_threshold': 2, 'demotion_threshold': 1, 'watermark': 0.9},
            {'name': 'tier1', 'capacity_count': 2, 'capacity_bytes': 1000, 'promotion_threshold': 2, 'demotion_threshold': 1, 'watermark': 0.9},
            {'name': 'tier2', 'capacity_count': 2, 'capacity_bytes': 1000, 'promotion_threshold': 2, 'demotion_threshold': 1, 'watermark': 0.9},
        ]
        self.backends = [MockPMDKBackend(), MockPMDKBackend(), MockPMDKBackend()]
        self.tiered = TieredBackend(self.backends, self.tier_configs)

    def test_insert_and_retrieve(self):
        self.tiered.put('frag1', b'data1', b'tokens1')
        result = self.tiered.get('frag1')
        self.assertIsNotNone(result)
        self.assertEqual(result[0], b'data1')
        self.assertEqual(result[1], b'tokens1')
        meta = self.tiered.metadata['frag1']
        self.assertEqual(meta['tier'], 0)

    def test_promotion_on_repeated_access(self):
        self.tiered.put('frag2', b'data2', b'tokens2')
        # Access enough times to trigger promotion
        for _ in range(3):
            self.tiered.get('frag2')
        # Should be promoted to tier 0 (already there), so demote to tier 1 and test promotion
        self.tiered._demote('frag2', 0, 1)
        meta = self.tiered.metadata['frag2']
        self.assertEqual(meta['tier'], 1)
        # Access enough times to trigger promotion back to tier 0
        for _ in range(3):
            self.tiered.get('frag2')
        meta = self.tiered.metadata['frag2']
        self.assertEqual(meta['tier'], 0)

    def test_demotion_on_capacity(self):
        # Fill tier 0
        self.tiered.put('frag3', b'data3', b'tokens3')
        self.tiered.put('frag4', b'data4', b'tokens4')
        # Next insert should demote coldest
        self.tiered.put('frag5', b'data5', b'tokens5')
        # One of the first two should be in tier 1
        tiers = [self.tiered.metadata[f]['tier'] for f in ['frag3', 'frag4', 'frag5']]
        self.assertIn(1, tiers)
        self.assertIn(0, tiers)

    def test_eviction_from_lowest_tier(self):
        # Fill all tiers
        self.tiered.put('frag6', b'data6', b'tokens6')
        self.tiered.put('frag7', b'data7', b'tokens7')
        self.tiered.put('frag8', b'data8', b'tokens8')
        self.tiered.put('frag9', b'data9', b'tokens9')
        self.tiered.put('frag10', b'data10', b'tokens10')
        self.tiered.put('frag11', b'data11', b'tokens11')
        # Next insert should evict from system
        self.tiered.put('frag12', b'data12', b'tokens12')
        # Only 6 fragments should remain (2 per tier)
        self.assertEqual(len(self.tiered.metadata), 6)
        # The evicted fragment should not be retrievable
        all_frags = set(self.tiered.metadata.keys())
        for f in ['frag6', 'frag7', 'frag8', 'frag9', 'frag10', 'frag11', 'frag12']:
            if f not in all_frags:
                self.assertIsNone(self.tiered.get(f))

    def test_proactive_demotion_watermark(self):
        """Test proactive demotion when tier usage exceeds watermark."""
        # Configure tier 0 to have capacity 2, watermark 0.5 (so >1 item triggers demotion)
        self.tier_configs[0]['watermark'] = 0.5
        self.tiered.put('fragA', b'dataA', b'tokensA') # Should stay
        self.tiered.put('fragB', b'dataB', b'tokensB') # Should trigger demotion of fragA
        self.assertEqual(self.tiered.metadata['fragA']['tier'], 1)
        self.assertEqual(self.tiered.metadata['fragB']['tier'], 0)

    def test_dynamic_threshold_adjustment(self):
        """Test that thresholds adjust based on hit rate (simple case)."""
        self.tiered._demote('fragC', 0, 1) # Move to tier 1
        initial_promo_threshold = self.tiered.tier_configs[1].get('promotion_threshold', 1)
        # Simulate low hit rate by missing frequently
        for _ in range(self.tiered.adjustment_interval):
            self.tiered.get('nonexistent')
        # Check if promotion threshold decreased
        new_promo_threshold = self.tiered.tier_configs[1]['promotion_threshold']
        self.assertLess(new_promo_threshold, initial_promo_threshold)

    def test_dynamic_demotion_threshold_adjustment(self):
        """Test that demotion_threshold adjusts based on demotion/eviction rate and usage."""
        tier = 0
        config = self.tiered.tier_configs[tier]
        initial_demotion_threshold = config.get('demotion_threshold', 1)
        # Simulate high usage and high demotion rate
        self.tiered.metrics[tier]['demotions'] = 10
        self.tiered.metrics[tier]['evictions'] = 2
        self.tiered.tier_usage[tier]['count'] = config['capacity_count']  # High usage
        self.tiered._feedback_adjust_thresholds()
        decreased = config['demotion_threshold'] < initial_demotion_threshold
        # Simulate low usage and low demotion rate
        config['demotion_threshold'] = initial_demotion_threshold  # Reset
        self.tiered.metrics[tier]['demotions'] = 0
        self.tiered.metrics[tier]['evictions'] = 10
        self.tiered.tier_usage[tier]['count'] = 0  # Low usage
        self.tiered._feedback_adjust_thresholds()
        increased = config['demotion_threshold'] > initial_demotion_threshold
        self.assertTrue(decreased or increased, "Demotion threshold should adjust dynamically.")

    def test_dynamic_watermark_adjustment(self):
        """Test that watermark adjusts based on sustained usage."""
        tier = 0
        config = self.tiered.tier_configs[tier]
        initial_watermark = config.get('watermark', 0.9)
        # Simulate high usage to trigger increase
        self.tiered.tier_usage[tier]['count'] = config['capacity_count']
        self.tiered._feedback_adjust_thresholds()
        increased = config['watermark'] > initial_watermark
        # Simulate low usage to trigger decrease
        config['watermark'] = initial_watermark  # Reset
        self.tiered.tier_usage[tier]['count'] = 0
        self.tiered._feedback_adjust_thresholds()
        decreased = config['watermark'] < initial_watermark
        self.assertTrue(increased or decreased, "Watermark should adjust dynamically.")


if __name__ == '__main__':
    unittest.main()
