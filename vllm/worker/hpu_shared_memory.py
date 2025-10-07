# vllm/worker/hpu_shared_memory.py

import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Optional, Tuple
import numpy as np
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


class SharedTensorRingBuffer:
    """Ring buffer for tensor data in shared memory - initialized once, reused forever."""
    
    def __init__(self, buffer_size: int, max_batch_size: int, vocab_size: int, 
                 attach_info: Optional[dict] = None):
        """
        Initialize ring buffer in create or attach mode.
        
        Args:
            buffer_size: Number of slots in the ring buffer
            max_batch_size: Maximum batch size for token tensors
            vocab_size: Vocabulary size for logprob tensors
            attach_info: If provided, attach to existing shared memory (Engine mode)
                        If None, create new shared memory (Worker mode)
        """
        self.buffer_size = buffer_size
        self.max_batch_size = max_batch_size
        self.vocab_size = vocab_size
        
        self.token_shms = []
        self.logprob_shms = []
        
        if attach_info is None:
            # === CREATE MODE (Worker Process) ===
            logger.info(f"Creating shared memory ring buffer: {buffer_size} slots, "
                       f"max_batch={max_batch_size}, vocab={vocab_size}")
            
            token_size = max_batch_size * 4  # int32
            logprob_size = max_batch_size * vocab_size * 4  # float32
            
            for i in range(buffer_size):
                # Token IDs
                token_shm = shared_memory.SharedMemory(
                    create=True, size=token_size)
                self.token_shms.append(token_shm)
                logger.debug(f"Created token shared memory slot {i}: {token_shm.name}")
                
                # Logprobs (optional)
                logprob_shm = shared_memory.SharedMemory(
                    create=True, size=logprob_size)
                self.logprob_shms.append(logprob_shm)
                logger.debug(f"Created logprob shared memory slot {i}: {logprob_shm.name}")
            
            # Control: current write/read positions (shared atomic counters)
            self.write_idx = mp.Value('i', 0)
            self.read_idx = mp.Value('i', 0)
            
            logger.info("Shared memory ring buffer created successfully")
            
        else:
            # === ATTACH MODE (Engine Process) ===
            logger.info(f"Attaching to existing shared memory ring buffer: {buffer_size} slots")
            
            for name in attach_info['token_shm_names']:
                self.token_shms.append(shared_memory.SharedMemory(name=name))
                logger.debug(f"Attached to token shared memory: {name}")
                
            for name in attach_info['logprob_shm_names']:
                self.logprob_shms.append(shared_memory.SharedMemory(name=name))
                logger.debug(f"Attached to logprob shared memory: {name}")
            
            # These are proxy objects that can be passed through queues/pipes
            self.write_idx = attach_info['write_idx']
            self.read_idx = attach_info['read_idx']
            
            logger.info("Successfully attached to shared memory ring buffer")
    
    def get_attach_info(self) -> dict:
        """Returns the necessary info for another process to attach."""
        return {
            "token_shm_names": [shm.name for shm in self.token_shms],
            "logprob_shm_names": [shm.name for shm in self.logprob_shms],
            "write_idx": self.write_idx,
            "read_idx": self.read_idx,
            "buffer_size": self.buffer_size,
            "max_batch_size": self.max_batch_size,
            "vocab_size": self.vocab_size,
        }
    
    def write_slot(self, token_ids_cpu: torch.Tensor, 
                   logprobs_cpu: Optional[torch.Tensor]) -> int:
        """Write to next available slot. Returns slot index."""
        with self.write_idx.get_lock():
            slot = self.write_idx.value
            self.write_idx.value = (slot + 1) % self.buffer_size
        
        # Zero-copy write to shared memory
        token_np = token_ids_cpu.numpy().astype(np.int32).flatten()
        token_array = np.ndarray(
            token_np.shape, dtype=np.int32, buffer=self.token_shms[slot].buf)
        np.copyto(token_array, token_np)
        
        if logprobs_cpu is not None:
            logprob_np = logprobs_cpu.numpy().astype(np.float32).flatten()
            logprob_array = np.ndarray(
                logprob_np.shape, dtype=np.float32, buffer=self.logprob_shms[slot].buf)
            np.copyto(logprob_array, logprob_np)
        
        return slot
    
    def read_slot(self, slot: int, token_shape: tuple, 
                  logprob_shape: Optional[tuple]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Read from specific slot."""
        # Calculate flat size
        token_flat_size = int(np.prod(token_shape))
        
        token_array = np.ndarray(
            (token_flat_size,), dtype=np.int32, buffer=self.token_shms[slot].buf)
        token_ids = torch.from_numpy(np.copy(token_array)).reshape(token_shape)
        
        logprobs = None
        if logprob_shape is not None:
            logprob_flat_size = int(np.prod(logprob_shape))
            logprob_array = np.ndarray(
                (logprob_flat_size,), dtype=np.float32, buffer=self.logprob_shms[slot].buf)
            logprobs = torch.from_numpy(np.copy(logprob_array)).reshape(logprob_shape)
        
        return token_ids, logprobs
    
    def cleanup(self):
        """Clean up shared memory (worker only)."""
        logger.info("Cleaning up shared memory ring buffer")
        for shm in self.token_shms + self.logprob_shms:
            try:
                shm.close()
                shm.unlink()
            except Exception as e:
                logger.warning(f"Error cleaning up shared memory: {e}")
    
    def detach(self):
        """Detach from shared memory without destroying it (engine only)."""
        logger.info("Detaching from shared memory ring buffer")
        for shm in self.token_shms + self.logprob_shms:
            try:
                shm.close()
            except Exception as e:
                logger.warning(f"Error detaching from shared memory: {e}")