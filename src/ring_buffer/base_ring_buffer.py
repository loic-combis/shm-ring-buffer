from typing import TypeVar

from .abstract_ring_buffer import RingBuffer

T = TypeVar("T")

class BaseRingBuffer(RingBuffer[T]):
    def __init__(self, slots: int) -> None:
        super().__init__(slots)

        self._r_idx_shm: int = 0
        self._w_idx_shm: int = 0

        
    @property
    def _r_idx(self) -> int:
        """Get current read index."""
        return self._r_idx_shm

    @_r_idx.setter
    def _r_idx(self, value: int) -> None:
        """Set current read index."""
        self._r_idx_shm = value

    @property
    def _w_idx(self) -> int:
        """Get current write index."""
        return self._w_idx_shm

    @_w_idx.setter
    def _w_idx(self, value: int) -> None:
        """Set current write index."""
        self._w_idx_shm = value