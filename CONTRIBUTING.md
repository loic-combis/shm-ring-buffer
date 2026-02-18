# Contributing to Ring Buffer
First off, thank you for considering contributing! Projects like this thrive on community involvement, whether you're fixing a bug or implementing a new feature like GPU support.

## 🚀 How to Contribute
1. Report Bugs or Request Features
If you find a bug or have an idea for an improvement, please open an issue.

Check if the issue already exists before opening a new one.

Provide a clear description and, if possible, a code snippet to reproduce the bug.

2. Submit a Pull Request (PR)
Fork the repository and create your branch from main.

Install dependencies (including dev dependencies) using poetry install.

Write your code. Ensure your changes follow the existing style and the SPSR (Single-Producer Single-Reader) design philosophy.

Test your changes. If you add a feature, please include a test case.

Open a PR with a clear description of what you changed and why.

## 🛠 Development Guidelines
Performance First: This library is designed for high-performance IPC. Avoid adding locks or heavy abstractions that might slow down the data path.

Typing: We use Python 3.11+ features. Please ensure your code is fully type-hinted.

Docstrings: Use clear docstrings for any new public-facing methods.

## 🗺 Current Focus
We are currently looking for help with:

- CuPy/CUDA implementation: Integrating GPU-backed buffers.

- Benchmarking: Expanding the performance test suite against other IPC methods.

## Questions?
If you're unsure about an implementation detail, feel free to start a discussion in an issue before writing any code. We're happy to help!