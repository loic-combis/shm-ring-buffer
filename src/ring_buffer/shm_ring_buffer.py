import struct

from typing import TypeVar
from multiprocessing import shared_memory

from .abstract_ring_buffer import RingBuffer

T = TypeVar("T")

class ShmRingBuffer(RingBuffer[T]):
    def __init__(self, slots: int, item_size: int, create: bool, buf_name: str, read_idx_name: str, write_idx_name: str) -> None:
        super().__init__(slots)
        self._item_size = item_size

        self._packer = struct.Struct('=q')

        # Shared memory space for read/write indexes.
        self._r_idx_shm = shared_memory.SharedMemory(name=read_idx_name, create=create, size=8)
        self._w_idx_shm = shared_memory.SharedMemory(name=write_idx_name, create=create, size=8)

        # Shared memory space for the ring buffer elements.
        self._buf_shm = shared_memory.SharedMemory(name=buf_name, create=create, size=self._slots * self._item_size)
        
    @property
    def _read_idx(self) -> int:
        """Get current read index."""
        if self._r_idx_shm.buf is None:
             raise ValueError("Cannot get read index: shared memory is closed or not initialized")
        
        return self._packer.unpack_from(self._r_idx_shm.buf, 0)[0]

    @_read_idx.setter
    def _read_idx(self, value: int) -> None:
        """Set current read index."""
        if self._r_idx_shm.buf is None:
             raise ValueError("Cannot set read index: shared memory is closed or not initialized")
       
        self._packer.pack_into(self._r_idx_shm.buf, 0, value)

    @property
    def _write_idx(self) -> int:
        """Get current write index."""
        if self._w_idx_shm.buf is None:
             raise ValueError("Cannot get write index: shared memory is closed or not initialized")
        
        return self._packer.unpack_from(self._w_idx_shm.buf, 0)[0]

    @_write_idx.setter
    def _write_idx(self, value: int) -> None:
        """Set current write index."""
        if self._w_idx_shm.buf is None:
             raise ValueError("Cannot set write index: shared memory is closed or not initialized")
        
        self._packer.pack_into(self._w_idx_shm.buf, 0, value)


    def close(self):
        """Cleanup resources if necessary."""
        self._r_idx_shm.close()
        self._r_idx_shm.unlink()
        
        self._w_idx_shm.close()
        self._w_idx_shm.unlink()

        self._buf_shm.close()
        self._buf_shm.unlink()

        