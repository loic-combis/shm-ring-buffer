from ring_buffer.shm_ring_buffer import ShmRingBuffer 


class BytesShmRingBuffer(ShmRingBuffer[bytes]):
    def __init__(self, 
                slots: int, 
                item_size: int,
                buf_shm_name: str, 
                read_idx_shm_name: str, 
                write_idx_shm_name: str,
                create: bool
            ) -> None:

        super().__init__(
            slots=slots, 
            item_size=item_size, 
            buf_name=buf_shm_name, 
            read_idx_name=read_idx_shm_name, 
            write_idx_name=write_idx_shm_name, 
            create=create
        )

        if self._buf_shm.buf is None:
            raise ValueError("Shared memory buffer is not initialized")

        self._elts: list[memoryview] = []

        # Pre-allocate memory views for read/write easiness.
        for i in range(self._slots):
            start = i * item_size
            end = start + item_size
            self._elts.append(self._buf_shm.buf[start:end])


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