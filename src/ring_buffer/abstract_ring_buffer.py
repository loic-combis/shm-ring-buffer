from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional

T = TypeVar("T")

class RingBuffer(ABC, Generic[T]):
    """
    Abstract Base Class for a Ring Buffer.
    
    Manages the logic for circular indexing, but delegates 
    storage and state management to subclasses.
    """
    def __init__(self, slots: int) -> None:
        # We use one slot as a sentinel to distinguish full vs empty
        # This simplifies lock-free logic.
        self._slots = slots + 1

    @property
    @abstractmethod
    def _r_idx(self) -> int:
        """Get current read index."""
        pass

    @_r_idx.setter
    @abstractmethod
    def _r_idx(self, value: int) -> None:
        """Set current read index."""
        pass

    @property
    @abstractmethod
    def _w_idx(self) -> int:
        """Get current write index."""
        pass

    @_w_idx.setter
    @abstractmethod
    def _w_idx(self, value: int) -> None:
        """Set current write index."""
        pass

    # --- Core Logic (Shared across all implementations) ---

    @property
    def capacity(self) -> int:
        """Total capacity of the buffer (excluding sentinel)."""
        return self._slots - 1

    def can_read(self) -> bool:
        """Check if there is data available to read."""
        return self._r_idx != self._w_idx

    def can_write(self) -> bool:
        """Check if there is space available to write."""
        return (self._w_idx + 1) % self._slots != self._r_idx

    def write(self, data: T) -> bool:
        """
        Writes data to the buffer if space is available.
        Returns True if successful, False if buffer was full.
        """
        if not self.can_write():
            return False

        # 1. Write the data to the current slot
        self._write(self._w_idx, data)

        # 2. Advance the write index
        self._w_idx = (self._w_idx + 1) % self._slots
        return True

    def read(self) -> Optional[T]:
        """
        Reads (peeks) at the next available element without removing it.
        Returns None if buffer is empty.
        """
        if not self.can_read():
            return None
        
        return self._read(self._r_idx)

    def release(self) -> bool:
        """
        Advances the read index, effectively freeing the slot 
        that was just read.
        """
        if not self.can_read():
            return False

        self._r_idx = (self._r_idx + 1) % self._slots
        return True

    # --- Abstract Storage Methods ---

    @abstractmethod
    def _write(self, index: int, data: T) -> None:
        """
        Store 'data' into the buffer at the given 'index'.
        """
        pass

    @abstractmethod
    def _read(self, index: int) -> T:
        """
        Retrieve data from the buffer at the given 'index'.
        """
        pass