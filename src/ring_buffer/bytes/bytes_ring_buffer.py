from ring_buffer.base_ring_buffer import RingBuffer 


class BytesRingBuffer(RingBuffer[bytes]):
    def __init__(self, 
                slots: int, 
                item_size: int,
            ) -> None:

        super().__init__(slots=slots)

        self._buf = bytearray(item_size * self._slots)
        self._view = memoryview(self._buf)

        self._elts: list[memoryview] = []

        # Pre-allocate memory views for read/write easiness.
        for i in range(self._slots):
            start = i * item_size
            end = start + item_size
            self._elts.append(self._view[start:end])


    def _write(self, index: int, data: bytes) -> None:
        """
        Store 'data' into the buffer at the given 'index'.
        """
        # Get the view for the current slot
        self._elts[index][:] = data


    def _read(self, index: int) -> bytes:
        """
        Retrieve data from the buffer at the given 'index'.
        """
        return self._elts[index]