from ring_buffer import BytesRingBuffer


def test_initialization():
    """Verify capacity and initial empty state."""
    buf = BytesRingBuffer(slots=5, item_size=10)
    assert buf.capacity == 5
    assert buf.can_read() is False
    assert buf.can_write() is True
    assert buf.read() is None


def test_basic_write_read():
    """Test simple FIFO write-read-release cycle."""
    buf = BytesRingBuffer(slots=5, item_size=4)
    data = b"abcd"
    assert buf.write(data) is True
    assert buf.can_read() is True
    assert bytes(buf.read()) == data
    assert buf.release() is True
    assert buf.can_read() is False


def test_fill_to_capacity():
    """Verify we can fill exactly 'capacity' items, then no more."""
    capacity = 3
    buf = BytesRingBuffer(slots=capacity, item_size=4)

    for i in range(capacity):
        assert buf.write(f"{i:04d}".encode()) is True

    assert buf.can_write() is False
    assert buf.write(b"full") is False


def test_fill_and_drain():
    """Fill buffer to capacity, then read all items in FIFO order."""
    capacity = 4
    buf = BytesRingBuffer(slots=capacity, item_size=2)
    items = [f"{i}x".encode() for i in range(capacity)]

    for item in items:
        buf.write(item)

    for item in items:
        assert bytes(buf.read()) == item
        buf.release()

    assert buf.can_read() is False


def test_circular_wrap_around():
    """Ensure indices wrap around correctly."""
    capacity = 3
    buf = BytesRingBuffer(slots=capacity, item_size=2)

    # Fill and drain to move indices forward
    for i in range(capacity):
        buf.write(f"{i}a".encode())
    for _ in range(capacity):
        buf.release()

    # Now write again — indices must wrap around the internal array
    assert buf.write(b"xx") is True
    assert buf.write(b"yy") is True
    assert bytes(buf.read()) == b"xx"
    buf.release()
    assert bytes(buf.read()) == b"yy"
    buf.release()
    assert buf.can_read() is False


def test_interleaved_write_read():
    """Interleave writes and reads to simulate streaming usage."""
    buf = BytesRingBuffer(slots=2, item_size=3)

    for i in range(10):
        data = f"{i:03d}".encode()
        assert buf.write(data) is True
        assert bytes(buf.read()) == data
        buf.release()


def test_release_on_empty():
    """Releasing an empty buffer should return False."""
    buf = BytesRingBuffer(slots=2, item_size=4)
    assert buf.release() is False


def test_read_on_empty():
    """Reading an empty buffer should return None."""
    buf = BytesRingBuffer(slots=2, item_size=4)
    assert buf.read() is None


def test_read_without_release():
    """Read (peek) should not advance the read pointer."""
    buf = BytesRingBuffer(slots=2, item_size=2)
    buf.write(b"hi")
    assert bytes(buf.read()) == b"hi"
    assert bytes(buf.read()) == b"hi"  # Same data again
    assert buf.can_read() is True


def test_write_overwrites_slot_after_release():
    """After release, the slot can be reused with new data."""
    buf = BytesRingBuffer(slots=1, item_size=4)

    buf.write(b"aaaa")
    assert bytes(buf.read()) == b"aaaa"
    buf.release()

    buf.write(b"bbbb")
    assert bytes(buf.read()) == b"bbbb"


def test_data_integrity_different_values():
    """Write distinct values and verify FIFO ordering is preserved."""
    capacity = 5
    buf = BytesRingBuffer(slots=capacity, item_size=4)
    values = [b"\x00\x01\x02\x03", b"\x04\x05\x06\x07",
              b"\x08\x09\x0a\x0b", b"\x0c\x0d\x0e\x0f",
              b"\x10\x11\x12\x13"]

    for v in values:
        buf.write(v)

    for v in values:
        assert bytes(buf.read()) == v
        buf.release()


def test_multiple_wrap_cycles():
    """Run multiple full fill-drain cycles to stress wrap-around logic."""
    capacity = 3
    buf = BytesRingBuffer(slots=capacity, item_size=1)

    for cycle in range(5):
        for i in range(capacity):
            assert buf.write(bytes([cycle * capacity + i])) is True
        for i in range(capacity):
            assert bytes(buf.read()) == bytes([cycle * capacity + i])
            buf.release()


def test_capacity_one():
    """Edge case: single-slot buffer."""
    buf = BytesRingBuffer(slots=1, item_size=3)
    assert buf.capacity == 1

    assert buf.write(b"abc") is True
    assert buf.can_write() is False
    assert bytes(buf.read()) == b"abc"
    buf.release()

    assert buf.write(b"xyz") is True
    assert bytes(buf.read()) == b"xyz"
