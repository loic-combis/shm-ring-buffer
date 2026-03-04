import uuid

from ring_buffer import BytesShmRingBuffer


def _unique_names():
    """Generate unique shared memory names to avoid collisions between tests."""
    uid = uuid.uuid4().hex[:8]
    return f"buf_{uid}", f"ri_{uid}", f"wi_{uid}"


def test_initialization():
    """Verify capacity and initial empty state."""
    buf_name, ri_name, wi_name = _unique_names()
    buf = BytesShmRingBuffer(5, 10, buf_name, ri_name, wi_name, create=True)
    try:
        assert buf.capacity == 5
        assert buf.can_read() is False
        assert buf.can_write() is True
        assert buf.read() is None
    finally:
        buf.close()


def test_basic_write_read():
    """Test simple FIFO write-read-release cycle."""
    buf_name, ri_name, wi_name = _unique_names()
    buf = BytesShmRingBuffer(5, 4, buf_name, ri_name, wi_name, create=True)
    try:
        data = b"abcd"
        assert buf.write(data) is True
        assert buf.can_read() is True
        assert bytes(buf.read()) == data
        assert buf.release() is True
        assert buf.can_read() is False
    finally:
        buf.close()


def test_fill_to_capacity():
    """Verify we can fill exactly 'capacity' items, then no more."""
    buf_name, ri_name, wi_name = _unique_names()
    capacity = 3
    buf = BytesShmRingBuffer(capacity, 4, buf_name, ri_name, wi_name, create=True)
    try:
        for i in range(capacity):
            assert buf.write(f"{i:04d}".encode()) is True

        assert buf.can_write() is False
        assert buf.write(b"full") is False
    finally:
        buf.close()


def test_fill_and_drain():
    """Fill buffer to capacity, then read all items in FIFO order."""
    buf_name, ri_name, wi_name = _unique_names()
    capacity = 4
    buf = BytesShmRingBuffer(capacity, 2, buf_name, ri_name, wi_name, create=True)
    try:
        items = [f"{i}x".encode() for i in range(capacity)]
        for item in items:
            buf.write(item)
        for item in items:
            assert bytes(buf.read()) == item
            buf.release()
        assert buf.can_read() is False
    finally:
        buf.close()


def test_circular_wrap_around():
    """Ensure indices wrap around correctly."""
    buf_name, ri_name, wi_name = _unique_names()
    capacity = 3
    buf = BytesShmRingBuffer(capacity, 2, buf_name, ri_name, wi_name, create=True)
    try:
        for i in range(capacity):
            buf.write(f"{i}a".encode())
        for _ in range(capacity):
            buf.release()

        assert buf.write(b"xx") is True
        assert buf.write(b"yy") is True
        assert bytes(buf.read()) == b"xx"
        buf.release()
        assert bytes(buf.read()) == b"yy"
        buf.release()
        assert buf.can_read() is False
    finally:
        buf.close()


def test_interleaved_write_read():
    """Interleave writes and reads to simulate streaming usage."""
    buf_name, ri_name, wi_name = _unique_names()
    buf = BytesShmRingBuffer(2, 3, buf_name, ri_name, wi_name, create=True)
    try:
        for i in range(10):
            data = f"{i:03d}".encode()
            assert buf.write(data) is True
            assert bytes(buf.read()) == data
            buf.release()
    finally:
        buf.close()


def test_release_on_empty():
    """Releasing an empty buffer should return False."""
    buf_name, ri_name, wi_name = _unique_names()
    buf = BytesShmRingBuffer(2, 4, buf_name, ri_name, wi_name, create=True)
    try:
        assert buf.release() is False
    finally:
        buf.close()


def test_read_on_empty():
    """Reading an empty buffer should return None."""
    buf_name, ri_name, wi_name = _unique_names()
    buf = BytesShmRingBuffer(2, 4, buf_name, ri_name, wi_name, create=True)
    try:
        assert buf.read() is None
    finally:
        buf.close()


def test_read_without_release():
    """Read (peek) should not advance the read pointer."""
    buf_name, ri_name, wi_name = _unique_names()
    buf = BytesShmRingBuffer(2, 2, buf_name, ri_name, wi_name, create=True)
    try:
        buf.write(b"hi")
        assert bytes(buf.read()) == b"hi"
        assert bytes(buf.read()) == b"hi"
        assert buf.can_read() is True
    finally:
        buf.close()


def test_shared_memory_visibility():
    """A second buffer attached to the same shm should see writes from the first."""
    buf_name, ri_name, wi_name = _unique_names()
    writer = BytesShmRingBuffer(4, 4, buf_name, ri_name, wi_name, create=True)
    reader = BytesShmRingBuffer(4, 4, buf_name, ri_name, wi_name, create=False)
    try:
        writer.write(b"abcd")
        assert reader.can_read() is True
        assert bytes(reader.read()) == b"abcd"
        reader.release()

        writer.write(b"efgh")
        assert bytes(reader.read()) == b"efgh"
    finally:
        # Only the creator unlinks; close reader first to release its views
        elts = getattr(reader, "_elts", None)
        if elts is not None:
            elts.clear()
        reader._r_idx_shm.close()
        reader._w_idx_shm.close()
        reader._buf_shm.close()
        writer.close()


def test_multiple_wrap_cycles():
    """Run multiple full fill-drain cycles to stress wrap-around logic."""
    buf_name, ri_name, wi_name = _unique_names()
    capacity = 3
    buf = BytesShmRingBuffer(capacity, 1, buf_name, ri_name, wi_name, create=True)
    try:
        for cycle in range(5):
            for i in range(capacity):
                assert buf.write(bytes([cycle * capacity + i])) is True
            for i in range(capacity):
                assert bytes(buf.read()) == bytes([cycle * capacity + i])
                buf.release()
    finally:
        buf.close()


def test_capacity_one():
    """Edge case: single-slot buffer."""
    buf_name, ri_name, wi_name = _unique_names()
    buf = BytesShmRingBuffer(1, 3, buf_name, ri_name, wi_name, create=True)
    try:
        assert buf.capacity == 1
        assert buf.write(b"abc") is True
        assert buf.can_write() is False
        assert bytes(buf.read()) == b"abc"
        buf.release()
        assert buf.write(b"xyz") is True
        assert bytes(buf.read()) == b"xyz"
    finally:
        buf.close()
