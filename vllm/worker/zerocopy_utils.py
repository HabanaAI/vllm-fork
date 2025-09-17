# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# vllm/worker/zerocopy_utils.py
import uuid
import weakref
from multiprocessing import shared_memory
from typing import Dict, Tuple

import numpy as np
import torch


class SharedTensorPool:
    """Manages shared memory blocks for zero-copy tensor transfer."""

    def __init__(self):
        self.active_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self._finalizer = weakref.finalize(self, self._cleanup_all)
    # --- Add these constants at the top of the file or class ---
    # Special dtype string for bf16 which numpy doesn't support.
    ENC_BF16_STR = "!bf16"

    def put_tensor(self, tensor: torch.Tensor) -> Tuple[str, dict]:
        """Put tensor in shared memory and return handle."""
        # print(f">>>>>>> SharedTensorPool: Putting tensor shape={tensor.shape}, dtype={tensor.dtype}")
        
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        tensor_id = str(uuid.uuid4())

        # --- FIXED: Handle bfloat16 by converting to float16 view for transfer ---
        original_dtype = tensor.dtype
        if original_dtype == torch.bfloat16:
            tensor_for_transfer = tensor.view(torch.float16)
            # print(f">>>>>>> SharedTensorPool: Converting bfloat16 tensor to float16 view for transfer")
        else:
            tensor_for_transfer = tensor

        # Convert to numpy
        if tensor_for_transfer.device.type == "hpu":
            np_array = tensor_for_transfer.cpu().numpy()
        else:
            np_array = tensor_for_transfer.numpy()

        # Create shared memory
        shm = shared_memory.SharedMemory(create=True, size=np_array.nbytes)

        # Copy data
        shared_array = np.ndarray(np_array.shape,
                                dtype=np_array.dtype,
                                buffer=shm.buf)
        shared_array[:] = np_array[:]

        self.active_blocks[tensor_id] = shm

        return tensor_id, {
            'shape': list(tensor.shape), # Store the *original* shape
            'dtype': str(original_dtype), # Store the *original* dtype (e.g., 'torch.bfloat16')
            'device': str(tensor.device),
            'shm_name': shm.name
        }

    def get_tensor(self, tensor_id: str, metadata: dict) -> torch.Tensor:
        """Retrieve tensor from shared memory (zero-copy)."""
        shm = shared_memory.SharedMemory(name=metadata['shm_name'])

        # Get the original dtype
        original_dtype_str = metadata['dtype']
        
        # Map numpy dtype
        dtype_str = original_dtype_str.replace('torch.', '')
        if dtype_str == 'float32':
            np_dtype = np.float32
        elif dtype_str == 'float16':
            np_dtype = np.float16
        elif dtype_str == 'bfloat16':  # <-- ADD THIS CONDITION
            np_dtype = np.float16      # <-- MAP TO float16 for NumPy
        elif dtype_str == 'int64':
            np_dtype = np.int64
        else:
            np_dtype = np.dtype(dtype_str)

        shared_array = np.ndarray(metadata['shape'],
                                dtype=np_dtype,
                                buffer=shm.buf)
        tensor = torch.from_numpy(shared_array)

        # --- FIXED: Convert back to bfloat16 if that was the original dtype ---
        if original_dtype_str == 'torch.bfloat16':
            tensor = tensor.view(torch.bfloat16)
            # print(f">>>>>>> SharedTensorPool: Converted tensor back to bfloat16")

        # Attach metadata for cleanup
        tensor._shm = shm
        tensor._tensor_id = tensor_id

        return tensor

    def _cleanup_all(self):
        for shm in self.active_blocks.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
