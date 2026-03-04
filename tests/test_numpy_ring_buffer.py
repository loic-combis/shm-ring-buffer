import numpy as np

from ring_buffer.numpy import NumpyRingBuffer


def test_initialization():
    """Verify capacity and initial empty state."""
    buf = NumpyRingBuffer(slots=5, dtype=np.float64)
    assert buf.capacity == 5
    assert buf.can_read() is False
    assert buf.can_write() is True
    assert buf.read() is None


def test_basic_write_read_scalar():
    """Test write/read with scalar (0-d) arrays."""
    buf = NumpyRingBuffer(slots=3, dtype=np.float64)
    buf.write(np.float64(42.0))
    result = buf.read()
    assert float(result) == 42.0
    buf.release()
    assert buf.can_read() is False


def test_basic_write_read_1d():
    """Test write/read with 1-d arrays."""
    buf = NumpyRingBuffer(slots=3, dtype=np.float32, shape=(4,))
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buf.write(data)
    result = buf.read()
    np.testing.assert_array_equal(result, data)
    buf.release()
    assert buf.can_read() is False


def test_basic_write_read_2d():
    """Test write/read with 2-d arrays."""
    buf = NumpyRingBuffer(slots=3, dtype=np.int32, shape=(2, 3))
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    buf.write(data)
    result = buf.read()
    np.testing.assert_array_equal(result, data)


def test_fill_to_capacity():
    """Verify we can fill exactly 'capacity' items, then no more."""
    capacity = 3
    buf = NumpyRingBuffer(slots=capacity, dtype=np.float64, shape=(2,))

    for i in range(capacity):
        assert buf.write(np.array([float(i), float(i + 10)], dtype=np.float64)) is True

    assert buf.can_write() is False
    assert buf.write(np.array([99.0, 99.0], dtype=np.float64)) is False


def test_fill_and_drain():
    """Fill buffer to capacity, then read all items in FIFO order."""
    capacity = 4
    buf = NumpyRingBuffer(slots=capacity, dtype=np.float64, shape=(2,))
    items = [np.array([float(i), float(i * 10)], dtype=np.float64) for i in range(capacity)]

    for item in items:
        buf.write(item)

    for item in items:
        np.testing.assert_array_equal(buf.read(), item)
        buf.release()

    assert buf.can_read() is False


def test_circular_wrap_around():
    """Ensure indices wrap around correctly."""
    capacity = 3
    buf = NumpyRingBuffer(slots=capacity, dtype=np.int32, shape=(2,))

    for i in range(capacity):
        buf.write(np.array([i, i + 10], dtype=np.int32))
    for _ in range(capacity):
        buf.release()

    a = np.array([100, 200], dtype=np.int32)
    b = np.array([300, 400], dtype=np.int32)
    buf.write(a)
    buf.write(b)
    np.testing.assert_array_equal(buf.read(), a)
    buf.release()
    np.testing.assert_array_equal(buf.read(), b)
    buf.release()
    assert buf.can_read() is False


def test_interleaved_write_read():
    """Interleave writes and reads to simulate streaming usage."""
    buf = NumpyRingBuffer(slots=2, dtype=np.float64, shape=(3,))

    for i in range(10):
        data = np.array([float(i), float(i * 2), float(i * 3)], dtype=np.float64)
        assert buf.write(data) is True
        np.testing.assert_array_equal(buf.read(), data)
        buf.release()


def test_release_on_empty():
    """Releasing an empty buffer should return False."""
    buf = NumpyRingBuffer(slots=2, dtype=np.float64)
    assert buf.release() is False


def test_read_on_empty():
    """Reading an empty buffer should return None."""
    buf = NumpyRingBuffer(slots=2, dtype=np.float64)
    assert buf.read() is None


def test_read_without_release():
    """Read (peek) should not advance the read pointer."""
    buf = NumpyRingBuffer(slots=2, dtype=np.float64, shape=(2,))
    data = np.array([1.0, 2.0], dtype=np.float64)
    buf.write(data)
    np.testing.assert_array_equal(buf.read(), data)
    np.testing.assert_array_equal(buf.read(), data)
    assert buf.can_read() is True


def test_zero_copy_read():
    """Read returns a view into the buffer, not a copy."""
    buf = NumpyRingBuffer(slots=2, dtype=np.float64, shape=(2,))
    data = np.array([1.0, 2.0], dtype=np.float64)
    buf.write(data)

    view1 = buf.read()
    view2 = buf.read()
    assert view1 is view2  # Same object, not a copy


def test_different_dtypes():
    """Verify support for various numpy dtypes."""
    for dtype in [np.int8, np.int16, np.int32, np.int64,
                  np.float32, np.float64, np.uint8, np.uint16]:
        buf = NumpyRingBuffer(slots=2, dtype=dtype, shape=(3,))
        data = np.array([1, 2, 3], dtype=dtype)
        buf.write(data)
        np.testing.assert_array_equal(buf.read(), data)


def test_multiple_wrap_cycles():
    """Run multiple full fill-drain cycles to stress wrap-around logic."""
    capacity = 3
    buf = NumpyRingBuffer(slots=capacity, dtype=np.int32)

    for cycle in range(5):
        for i in range(capacity):
            val = np.int32(cycle * capacity + i)
            assert buf.write(val) is True
        for i in range(capacity):
            assert int(buf.read()) == cycle * capacity + i
            buf.release()


def test_capacity_one():
    """Edge case: single-slot buffer."""
    buf = NumpyRingBuffer(slots=1, dtype=np.float64, shape=(2,))
    assert buf.capacity == 1

    data = np.array([1.0, 2.0], dtype=np.float64)
    assert buf.write(data) is True
    assert buf.can_write() is False
    np.testing.assert_array_equal(buf.read(), data)
    buf.release()

    data2 = np.array([3.0, 4.0], dtype=np.float64)
    assert buf.write(data2) is True
    np.testing.assert_array_equal(buf.read(), data2)


def test_structured_dtype():
    """Test with a structured (record) dtype."""
    dt = np.dtype([("x", np.float32), ("y", np.float32), ("id", np.int32)])
    buf = NumpyRingBuffer(slots=3, dtype=dt)

    record = np.array((1.5, 2.5, 42), dtype=dt)
    buf.write(record)
    result = buf.read()
    assert result["x"] == np.float32(1.5)
    assert result["y"] == np.float32(2.5)
    assert result["id"] == 42
    buf.release()

    record2 = np.array((3.0, 4.0, 99), dtype=dt)
    buf.write(record2)
    result2 = buf.read()
    assert result2["x"] == np.float32(3.0)
    assert result2["id"] == 99


def test_structured_dtype_array():
    """Test with an array of structured records."""
    dt = np.dtype([("pos", np.float64, (3,)), ("label", np.uint8)])
    buf = NumpyRingBuffer(slots=3, dtype=dt, shape=(2,))

    data = np.array([
        ([1.0, 2.0, 3.0], 10),
        ([4.0, 5.0, 6.0], 20),
    ], dtype=dt)
    buf.write(data)
    result = buf.read()
    np.testing.assert_array_equal(result["pos"], data["pos"])
    np.testing.assert_array_equal(result["label"], data["label"])


def test_structured_dtype_fill_drain():
    """Fill and drain with structured dtype to verify FIFO ordering."""
    dt = np.dtype([("a", np.int16), ("b", np.float32)])
    capacity = 4
    buf = NumpyRingBuffer(slots=capacity, dtype=dt)

    items = [np.array((i, float(i * 0.1)), dtype=dt) for i in range(capacity)]
    for item in items:
        buf.write(item)

    for item in items:
        result = buf.read()
        assert result["a"] == item["a"]
        np.testing.assert_almost_equal(float(result["b"]), float(item["b"]), decimal=5)
        buf.release()

    assert buf.can_read() is False


def test_structured_dtype_wrap_around():
    """Verify wrap-around works correctly with structured dtypes."""
    dt = np.dtype([("val", np.int32)])
    capacity = 3
    buf = NumpyRingBuffer(slots=capacity, dtype=dt)

    # Fill and drain to advance indices
    for i in range(capacity):
        buf.write(np.array((i,), dtype=dt))
    for _ in range(capacity):
        buf.release()

    # Write again to force wrap
    for i in range(capacity):
        buf.write(np.array((i + 100,), dtype=dt))
    for i in range(capacity):
        assert buf.read()["val"] == i + 100
        buf.release()


def test_complex_dtype():
    """Test with complex number dtype."""
    buf = NumpyRingBuffer(slots=3, dtype=np.complex128, shape=(2,))
    data = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex128)
    buf.write(data)
    np.testing.assert_array_equal(buf.read(), data)


def test_bool_dtype():
    """Test with boolean dtype."""
    buf = NumpyRingBuffer(slots=3, dtype=np.bool_, shape=(4,))
    data = np.array([True, False, True, False], dtype=np.bool_)
    buf.write(data)
    np.testing.assert_array_equal(buf.read(), data)
