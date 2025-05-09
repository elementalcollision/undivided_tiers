import threading
import os
import logging
import torch
import hashlib
import tempfile
import shutil
from typing import NamedTuple, Tuple, List


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


from . import serdes
from .backend import FileSystemBackend, StorageBackend


class SplitFactorError(Exception):
    pass


class VUAConfig:
    split_factor = 200

    @classmethod
    def tokens_to_paths(cls, tokens):
        """
        Convert a tensor of tokens into a list of hash directory names, where each hash
        includes all tokens from the beginning up to the current group.

        Parameters:
            tokens (Tensor): A PyTorch or numpy-compatible tensor of tokens
            with shape [n] or [1, n].

        Returns:
            List[str]: A list of directory names, each representing a hash of all tokens
            up to that group.

        Raises:
            SplitFactorError: If the total number of tokens is not divisible by
            the split_factor.
        """

        tokens = tokens.squeeze()
        token_list = list(tokens)
        if len(token_list) % cls.split_factor != 0:
            raise SplitFactorError(len(token_list), cls.split_factor)
        num_groups = len(token_list) // cls.split_factor
        path_components = []
        prev = ""
        for i in range(num_groups):
            # Each hash includes all tokens from the beginning up to this group
            group_tokens = token_list[: (i + 1) * cls.split_factor]
            hex_tokens = [format(token, 'x') for token in group_tokens]
            component = ",".join(hex_tokens)
            component = hashlib.sha1((prev + component).encode('utf-8')).hexdigest()
            prev = "hash:" + component + ":"
            path_components.append(component)
        return path_components  # List of strings

    @classmethod
    def trim_to_split_factor(cls, tokens):
        """Trim a tokens tensor so that its length is divisible by split_factor.

        Parameters:
            tokens (Tensor): A PyTorch or numpy-compatible tensor of tokens
            with shape [n] or [1, n].

        Returns:
            Tensor: The trimmed tensor.
        """
        tokens = tokens.squeeze()
        return tokens[:len(tokens) - (len(tokens) % cls.split_factor)]


class ClosestKV(NamedTuple):
    """
    tokens (Tensor): A PyTorch tensor of shape:
        tensor.Size([1, seq_len])

    data (List[Tuple[Tensor, Tensor]]): a KVCache in the struct of
        Transformers library. The list of layers, each layer has a 2-tuple
        for keys and values, and each keys or values tensor is of the
        following shape:

        tensor.Size([1, num_heads, seq_len, head_dim])
    """

    data: torch.Tensor
    tokens: List[Tuple[torch.Tensor, torch.Tensor]]


class VUA:
    def __init__(self, config, root_path, backend: StorageBackend = None):
        self._config = config
        self._root_path = root_path
        if backend is None:
            self._backend = FileSystemBackend(root_path)
        else:
            self._backend = backend

    def config(self) -> VUAConfig:
        """
        Return configruration for this VUA instance.
        """
        return self._config

    def put(self, tokens, data):
        """
        Save split kvcache data and tokens into a nested directory structure
        derived from the token values.

        The tokens are processed to generate directory path components.
        Directories are created as needed, ensuring that the generated path
        does not exceed system limitations. The data and tokens are stored in
        files named '_data' and '_tokens' respectively within each node.

        Parameters:
            tokens (Tensor): A PyTorch tensor of shape:
                tensor.Size([batch_head, seq_len])
                if provided `tensor.Size([seq_len])`, a 1-batch is assumed.

            data (List[Tuple[Tensor, Tensor]]): a KVCache in the struct of
                Transformers library. The list of layers, each layer has a 2-tuple
                for keys and values, and each keys or values tensor is of the
                following shape:

                tensor.Size([batch_head, num_heads, seq_len, head_dim])

            seq_len must be a multiple of config().split_factor

        Returns:
            None if the root path does not exist, otherwise None.
        """

        if tokens.dim() > 2:
            raise Exception(f"input token tensor dimension too big {tokens.dim()}")

        if tokens.dim() == 2:
            threads = []
            assert data[0][0].size(0) == tokens.size(0), "number token sequences should match batch_head"
            for i in range(tokens.size(0)):
                split_kvcache = [[kv[i].unsqueeze(0) for kv in layer] for layer in data]
                t = threading.Thread(target=self.put, args=(tokens[i], split_kvcache))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            return

        logger.debug(f"put with {len(tokens)} tokens")
        tokens = tokens.squeeze()
        path_components = self._config.tokens_to_paths(tokens)

        def save_group(group_hash, group_idx):
            group_dir = os.path.join(self._root_path, group_hash)
            if os.path.exists(group_dir):
                return

            group_dir_tmp = group_dir + ".tmp"
            try:
                os.mkdir(group_dir_tmp)
                logger.debug(f"group #{group_idx}: dir created {group_dir_tmp}")
            except OSError:
                pass

            if group_idx > 0:
                parent_hash = path_components[group_idx - 1]
                parent_link = os.path.join(group_dir_tmp, "parent")
                try:
                    os.symlink(os.path.join("..", parent_hash), parent_link)
                except FileExistsError:
                    pass

            sliced_group = []
            for layer in data:
                x = []
                for t in layer:
                    t2 = t[:, :, group_idx*self._config.split_factor:(group_idx+1)*self._config.split_factor, :]
                    x.append(t2.clone())
                sliced_group.append(torch.stack(x))

            sliced_group_bytes = serdes.tensor_to_bytes(torch.stack(sliced_group), group_hash + ".data")
            sliced_tokens_bytes = serdes.tensor_to_bytes(tokens[group_idx*self._config.split_factor:
                                       (group_idx+1)*self._config.split_factor].clone(), group_hash + ".tokens")

            # Use backend to store data and tokens
            self._backend.put(group_hash, sliced_group_bytes, sliced_tokens_bytes)

            # Atomically rename the temporary directory to the final hash path
            try:
                # Replace os.rename with shutil.move for potentially better handling
                shutil.move(group_dir_tmp, group_dir)
                # os.rename(group_dir_tmp, group_dir)
            except OSError as e:
                # Handle potential race condition or other errors during rename
                logger.error(f"Error renaming temp dir {group_dir_tmp} to {group_dir}: {e}")
                # Attempt cleanup of temp dir if rename failed
                try:
                    shutil.rmtree(group_dir_tmp)
                except OSError as cleanup_e:
                    logger.error(f"Failed to cleanup temp dir {group_dir_tmp} after rename error: {cleanup_e}")
                # Re-raise the original error so the operation failure is known
                raise e
            except Exception as e:
                logger.error(f"Unexpected error during rename/move of {group_dir_tmp}: {e}")
                # Attempt cleanup
                try:
                    shutil.rmtree(group_dir_tmp)
                except OSError as cleanup_e:
                     logger.error(f"Failed to cleanup temp dir {group_dir_tmp} after unexpected error: {cleanup_e}")
                raise e

        threads = []
        for group_idx, group_hash in enumerate(path_components):
            t = threading.Thread(target=save_group, args=(group_hash, group_idx))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    def get_closest(self, tokens, device):
        """
        Reconstruct KVCaches from stored fragments based on a token path lookup.

        It supports both a single token tensor and a list of token tensors.
        When provided a list, it returns a list where each element is the
        result corresponding to the best match for the input token tensor; when
        provided a single tensor, it returns a ClosestKV instance or None if no
        match is found.

        Parameters:
            tokens (Tensor or List[Tensor]): A PyTorch tensor of tokens
            (1-dimensional) or a list thereof. device (Device): The device to
            load the stored tensors onto (e.g., 'cpu' or 'cuda:<num>').

        Returns:
            Union[ClosestKV, None] or List[Union[ClosestKV, None]]: For a
            single token tensor input, returns a ClosestKV instance or None;
            for a list of token tensors, returns a list with an entry for each
            tensor.
        """

        if isinstance(tokens, torch.Tensor):
            if tokens.dim() >= 2:
                raise Exception(f"input token tensor dimension too big {tokens.dim()}")
            logger.debug(f"get with tokens.size()={tokens.size()} tokens")
        elif isinstance(tokens, list) and \
                all(isinstance(t, torch.Tensor) and t.dim() == 1 for t in tokens):
            tokens_groups = tokens
            threads = []
            results = [None] * len(tokens_groups)

            for i, token_list in enumerate(tokens_groups):
                def worker(token_list, device, i=i):
                    results[i] = self.get_closest(token_list, device)
                t = threading.Thread(target=worker, args=(token_list, device))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            return results

        results = []
        tokens = tokens.squeeze()
        tokens = self._config.trim_to_split_factor(tokens)
        logger.debug(f"tokens shape: {tokens.size()}")
        path_components = self._config.tokens_to_paths(tokens)

        threads = []

        def load_group(group_hash, group_idx):
            # Use backend to load data and tokens
            loaded = self._backend.get(group_hash)
            if loaded is None:
                logger.debug(f"group {group_hash} not found")
                return
            data_bytes, tokens_bytes = loaded
            tokens_tensor = serdes.bytes_to_tensor(tokens_bytes, group_hash + ".tokens")
            data_tensor = serdes.bytes_to_tensor(data_bytes, group_hash + ".data")
            results.append((group_idx, group_hash, (tokens_tensor, data_tensor)))

        for group_idx, group_hash in enumerate(path_components):
            t = threading.Thread(target=load_group, args=(group_hash, group_idx))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        if not results:
            return None

        results.sort()
        cache_groups = []
        tokens_groups = []

        for (group_idx, _, (tokens, data)) in results:
            tokens_groups.append(tokens)
            cache_groups.append(data.pin_memory().to(device=device, non_blocking=True))

        data = []
        num_layers = len(cache_groups[0])
        for layer_idx in range(num_layers):
            keys = []
            values = []
            for group in cache_groups:
                keys.append(group[layer_idx][0])
                values.append(group[layer_idx][1])
            combined_key = torch.cat(keys, dim=2)
            combined_value = torch.cat(values, dim=2)
            data.append((combined_key, combined_value))

        logger.debug("data combination ended")

        tokens = torch.cat(tokens_groups, dim=0).to(device=device)
        return ClosestKV(tokens=tokens, data=data)

    def repair_symlinks(self):
        """
        Traverse all group directories in the cache root, check the 'parent' symlink in each,
        and repair or report missing/broken symlinks. Logs actions taken. Safe to run multiple times.
        """
        import os
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"Starting symlink repair in {self._root_path}")

        # List all directories in the root path
        group_dirs = [d for d in os.listdir(self._root_path)
                      if os.path.isdir(os.path.join(self._root_path, d)) and not d.endswith('.tmp')]
        group_dirs.sort()  # Sort for deterministic order

        for group_dir in group_dirs:
            full_group_dir = os.path.join(self._root_path, group_dir)
            parent_link = os.path.join(full_group_dir, "parent")
            if os.path.exists(parent_link):
                # Check if it's a symlink and if it points to a valid directory
                if not os.path.islink(parent_link):
                    logger.warning(f"'{parent_link}' exists but is not a symlink. Skipping.")
                    continue
                target = os.readlink(parent_link)
                target_path = os.path.normpath(os.path.join(full_group_dir, target))
                if not os.path.isdir(target_path):
                    logger.warning(f"'{parent_link}' is a broken symlink. Removing and attempting repair.")
                    os.remove(parent_link)
                else:
                    # Symlink exists and is valid
                    continue
            # Try to infer the parent directory (by convention, one level up, any sibling hash)
            # This is a best-effort guess: look for a sibling directory and link to it if possible
            siblings = [d for d in group_dirs if d != group_dir]
            if siblings:
                # Pick the lexicographically previous sibling as parent, if possible
                idx = group_dirs.index(group_dir)
                if idx > 0:
                    parent_hash = group_dirs[idx - 1]
                    rel_path = os.path.join("..", parent_hash)
                    try:
                        os.symlink(rel_path, parent_link)
                        logger.info(f"Created missing symlink: {parent_link} -> {rel_path}")
                    except Exception as e:
                        logger.error(f"Failed to create symlink {parent_link} -> {rel_path}: {e}")
                else:
                    logger.info(f"No parent for first group directory: {group_dir}")
            else:
                logger.info(f"No siblings found for {group_dir}, cannot infer parent.")
