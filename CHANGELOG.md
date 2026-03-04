# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-04

### Added
- `BytesRingBuffer` for in-memory byte stream ring buffers.
- `BytesShmRingBuffer` for shared memory byte stream ring buffers.
- `NumpyRingBuffer` for in-memory NumPy array ring buffers.
- `NumpyShmRingBuffer` for shared memory NumPy array ring buffers.
- Zero-copy reads via buffer views.
- Lock-free Single-Producer Single-Reader (SPSR) design.
- Optional NumPy dependency (`pip install ring-buffer[numpy]`).
