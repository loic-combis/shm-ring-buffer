try:
    import numpy as np
except ImportError:
    raise ImportError(
        "\n\n"
        "Missing dependency: 'numpy'\n"
        "To use NumpyRingBuffer, you must install the numpy extra:\n"
        "    pip install ring-buffer[numpy]\n"
    ) from None


from typing import Optional

from ring_buffer.base_ring_buffer import BaseRingBuffer 


class NumpyRingBuffer(BaseRingBuffer[np.ndarray]):
    def __init__(self, 
                slots: int, 
                dtype: np.dtype, 
                shape: Optional[tuple[int, ...]] = None) -> None:
        
        
        super().__init__(slots=slots)

        item_size: int

        if shape is None:
            item_size = dtype.itemsize
        else:
            item_size = int(np.prod(shape) * dtype.itemsize)

        self._buf = bytearray(item_size * self._slots)

        self._elts: list[np.ndarray] = []

        # Pre-allocate numpy views
        for i in range(self._slots):
            view = np.ndarray(
                shape=shape or (),
                dtype=dtype,
                buffer=self._buf,
                offset=i * item_size,
            )

            self._elts.append(view)


    def _write(self, index: int, data: np.ndarray) -> None:
        """
        Store 'data' into the buffer at the given 'index'.
        """
        # Get the view for the current slot
        view = self._elts[index]
        view[()] = data


    def _read(self, index: int) -> np.ndarray:
        """
        Retrieve data from the buffer at the given 'index'.
        """
        return self._elts[index]