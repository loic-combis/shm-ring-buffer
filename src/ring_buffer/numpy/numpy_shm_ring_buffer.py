try:
    import numpy as np
except ImportError:
    raise ImportError(
        "\n\n"
        "Missing dependency: 'numpy'\n"
        "To use NumpyShmRingBuffer, you must install the numpy extra:\n"
        "    pip install ring-buffer[numpy]\n"
    ) from None
from typing import Optional

from ring_buffer.shm_ring_buffer import ShmRingBuffer 


class NumpyShmRingBuffer(ShmRingBuffer[np.ndarray]):
    def __init__(self, 
                slots: int, 
                buf_shm_name: str, 
                read_idx_shm_name: str, 
                write_idx_shm_name: str,
                create: bool,
                dtype: np.dtype, 
                shape: Optional[tuple[int, ...]] = None) -> None:
        
        dtype = np.dtype(dtype)
        item_size: int

        if shape is None:
            item_size = dtype.itemsize
        else:
            item_size = int(np.prod(shape) * dtype.itemsize)
        
        super().__init__(
            slots=slots, 
            item_size=item_size, 
            buf_name=buf_shm_name, 
            read_idx_name=read_idx_shm_name, 
            write_idx_name=write_idx_shm_name, 
            create=create
        )

        self._elts: list[np.ndarray] = []

        # Pre-allocate numpy views
        for i in range(self._slots):
            view = np.ndarray(
                shape=shape or (),
                dtype=dtype,
                buffer=self._buf_shm.buf,
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