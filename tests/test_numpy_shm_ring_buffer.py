import uuid

import numpy as np

from ring_buffer.numpy import NumpyShmRingBuffer


def _unique_names():
    """Generate unique shared memory names to avoid collisions between tests."""
    uid = uuid.uuid4().hex[:8]
    return f"buf_{uid}", f"ri_{uid}", f"wi_{uid}"


def test_initialization():
    """Verify capacity and initial empty state."""
    buf_name, ri_name, wi_name = _unique_names()
    buf = NumpyShmRingBuffer(5, buf_name, ri_name, wi_name, create=True,
                             dtype=np.float64)
    try:
        assert buf.capacity == 5
        assert buf.can_read() is False
        assert buf.can_write() is True
        assert buf.read() is None
    finally:
        buf.close()


def test_basic_write_read_scalar():
    """Test write/read with scalar (0-d) arrays."""
    buf_name, ri_name, wi_name = _unique_names()
    buf = NumpyShmRingBuffer(3, buf_name, ri_name, wi_name, create=True,
                             dtype=np.float64)
    try:
        buf.write(np.float64(42.0))
        result = buf.read()
        assert float(result) == 42.0
        buf.release()
        assert buf.can_read() is False
    finally:
        buf.close()


def test_basic_write_read_1d():
    """Test write/read with 1-d arrays."""
    buf_name, ri_name, wi_name = _unique_names()
    buf = NumpyShmRingBuffer(3, buf_name, ri_name, wi_name, create=True,
                             dtype=np.float32, shape=(4,))
    try:
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf.write(data)
        np.testing.assert_array_equal(buf.read(), data)
        buf.release()
        assert buf.can_read() is False
    finally:
        buf.close()


def test_basic_write_read_2d():
    """Test write/read with 2-d arrays."""
    buf_name, ri_name, wi_name = _unique_names()
    buf = NumpyShmRingBuffer(3, buf_name, ri_name, wi_name, create=True,
                             dtype=np.int32, shape=(2, 3))
    try:
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        buf.write(data)
        np.testing.assert_array_equal(buf.read(), data)
    finally:
        buf.close()


def test_fill_to_capacity():
    """Verify we can fill exactly 'capacity' items, then no more."""
    buf_name, ri_name, wi_name = _unique_names()
    capacity = 3
    buf = NumpyShmRingBuffer(capacity, buf_name, ri_name, wi_name, create=True,
                             dtype=np.float64, shape=(2,))
    try:
        for i in range(capacity):
            assert buf.write(np.array([float(i), float(i + 10)], dtype=np.float64)) is True
        assert buf.can_write() is False
        assert buf.write(np.array([99.0, 99.0], dtype=np.float64)) is False
    finally:
        buf.close()


def test_fill_and_drain():
    """Fill buffer to capacity, then read all items in FIFO order."""
    buf_name, ri_name, wi_name = _unique_names()
    capacity = 4
    buf = NumpyShmRingBuffer(capacity, buf_name, ri_name, wi_name, create=True,
                             dtype=np.float64, shape=(2,))
    try:
        items = [np.array([float(i), float(i * 10)], dtype=np.float64) for i in range(capacity)]
        for item in items:
            buf.write(item)
        for item in items:
            np.testing.assert_array_equal(buf.read(), item)
            buf.release()
        assert buf.can_read() is False
    finally:
        buf.close()


def test_circular_wrap_around():
    """Ensure indices wrap around correctly."""
    buf_name, ri_name, wi_name = _unique_names()
    capacity = 3
    buf = NumpyShmRingBuffer(capacity, buf_name, ri_name, wi_name, create=True,
                             dtype=np.int32, shape=(2,))
    try:
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
    finally:
        buf.close()


def test_interleaved_write_read():
    """Interleave writes and reads to simulate streaming usage."""
    buf_name, ri_name, wi_name = _unique_names()
    buf = NumpyShmRingBuffer(2, buf_name, ri_name, wi_name, create=True,
                             dtype=np.float64, shape=(3,))
    try:
        for i in range(10):
            data = np.array([float(i), float(i * 2), float(i * 3)], dtype=np.float64)
            assert buf.write(data) is True
            np.testing.assert_array_equal(buf.read(), data)
            buf.release()
    finally:
        buf.close()


def test_release_on_empty():
    """Releasing an empty buffer should return False."""
    buf_name, ri_name, wi_name = _unique_names()
    buf = NumpyShmRingBuffer(2, buf_name, ri_name, wi_name, create=True,
                             dtype=np.float64)
    try:
        assert buf.release() is False
    finally:
        buf.close()


def test_read_on_empty():
    """Reading an empty buffer should return None."""
    buf_name, ri_name, wi_name = _unique_names()
    buf = NumpyShmRingBuffer(2, buf_name, ri_name, wi_name, create=True,
                             dtype=np.float64)
    try:
        assert buf.read() is None
    finally:
        buf.close()


def test_read_without_release():
    """Read (peek) should not advance the read pointer."""
    buf_name, ri_name, wi_name = _unique_names()
    buf = NumpyShmRingBuffer(2, buf_name, ri_name, wi_name, create=True,
                             dtype=np.float64, shape=(2,))
    try:
        data = np.array([1.0, 2.0], dtype=np.float64)
        buf.write(data)
        np.testing.assert_array_equal(buf.read(), data)
        np.testing.assert_array_equal(buf.read(), data)
        assert buf.can_read() is True
    finally:
        buf.close()


def test_shared_memory_visibility():
    """A second buffer attached to the same shm should see writes from the first."""
    buf_name, ri_name, wi_name = _unique_names()
    writer = NumpyShmRingBuffer(4, buf_name, ri_name, wi_name, create=True,
                                dtype=np.float64, shape=(2,))
    reader = NumpyShmRingBuffer(4, buf_name, ri_name, wi_name, create=False,
                                dtype=np.float64, shape=(2,))
    try:
        data1 = np.array([1.0, 2.0], dtype=np.float64)
        writer.write(data1)
        assert reader.can_read() is True
        np.testing.assert_array_equal(reader.read(), data1)
        reader.release()

        data2 = np.array([3.0, 4.0], dtype=np.float64)
        writer.write(data2)
        np.testing.assert_array_equal(reader.read(), data2)
    finally:
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
    buf = NumpyShmRingBuffer(capacity, buf_name, ri_name, wi_name, create=True,
                             dtype=np.int32)
    try:
        for cycle in range(5):
            for i in range(capacity):
                val = np.int32(cycle * capacity + i)
                assert buf.write(val) is True
            for i in range(capacity):
                assert int(buf.read()) == cycle * capacity + i
                buf.release()
    finally:
        buf.close()


def test_capacity_one():
    """Edge case: single-slot buffer."""
    buf_name, ri_name, wi_name = _unique_names()
    buf = NumpyShmRingBuffer(1, buf_name, ri_name, wi_name, create=True,
                             dtype=np.float64, shape=(2,))
    try:
        assert buf.capacity == 1
        data = np.array([1.0, 2.0], dtype=np.float64)
        assert buf.write(data) is True
        assert buf.can_write() is False
        np.testing.assert_array_equal(buf.read(), data)
        buf.release()

        data2 = np.array([3.0, 4.0], dtype=np.float64)
        assert buf.write(data2) is True
        np.testing.assert_array_equal(buf.read(), data2)
    finally:
        buf.close()


def test_structured_dtype():
    """Test with a structured (record) dtype."""
    buf_name, ri_name, wi_name = _unique_names()
    dt = np.dtype([("x", np.float32), ("y", np.float32), ("id", np.int32)])
    buf = NumpyShmRingBuffer(3, buf_name, ri_name, wi_name, create=True, dtype=dt)
    try:
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
    finally:
        buf.close()


def test_structured_dtype_array():
    """Test with an array of structured records."""
    buf_name, ri_name, wi_name = _unique_names()
    dt = np.dtype([("pos", np.float64, (3,)), ("label", np.uint8)])
    buf = NumpyShmRingBuffer(3, buf_name, ri_name, wi_name, create=True,
                             dtype=dt, shape=(2,))
    try:
        data = np.array([
            ([1.0, 2.0, 3.0], 10),
            ([4.0, 5.0, 6.0], 20),
        ], dtype=dt)
        buf.write(data)
        result = buf.read()
        np.testing.assert_array_equal(result["pos"], data["pos"])
        np.testing.assert_array_equal(result["label"], data["label"])
    finally:
        buf.close()


def test_structured_dtype_fill_drain():
    """Fill and drain with structured dtype to verify FIFO ordering."""
    buf_name, ri_name, wi_name = _unique_names()
    dt = np.dtype([("a", np.int16), ("b", np.float32)])
    capacity = 4
    buf = NumpyShmRingBuffer(capacity, buf_name, ri_name, wi_name, create=True,
                             dtype=dt)
    try:
        items = [np.array((i, float(i * 0.1)), dtype=dt) for i in range(capacity)]
        for item in items:
            buf.write(item)

        for item in items:
            result = buf.read()
            assert result["a"] == item["a"]
            np.testing.assert_almost_equal(float(result["b"]), float(item["b"]), decimal=5)
            buf.release()

        assert buf.can_read() is False
    finally:
        buf.close()


def test_structured_dtype_wrap_around():
    """Verify wrap-around works correctly with structured dtypes."""
    buf_name, ri_name, wi_name = _unique_names()
    dt = np.dtype([("val", np.int32)])
    capacity = 3
    buf = NumpyShmRingBuffer(capacity, buf_name, ri_name, wi_name, create=True,
                             dtype=dt)
    try:
        for i in range(capacity):
            buf.write(np.array((i,), dtype=dt))
        for _ in range(capacity):
            buf.release()

        for i in range(capacity):
            buf.write(np.array((i + 100,), dtype=dt))
        for i in range(capacity):
            assert buf.read()["val"] == i + 100
            buf.release()
    finally:
        buf.close()


def test_complex_dtype():
    """Test with complex number dtype."""
    buf_name, ri_name, wi_name = _unique_names()
    buf = NumpyShmRingBuffer(3, buf_name, ri_name, wi_name, create=True,
                             dtype=np.complex128, shape=(2,))
    try:
        data = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex128)
        buf.write(data)
        np.testing.assert_array_equal(buf.read(), data)
    finally:
        buf.close()


def test_bool_dtype():
    """Test with boolean dtype."""
    buf_name, ri_name, wi_name = _unique_names()
    buf = NumpyShmRingBuffer(3, buf_name, ri_name, wi_name, create=True,
                             dtype=np.bool_, shape=(4,))
    try:
        data = np.array([True, False, True, False], dtype=np.bool_)
        buf.write(data)
        np.testing.assert_array_equal(buf.read(), data)
    finally:
        buf.close()
