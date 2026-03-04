#!/usr/bin/env python3
"""Test runner for ring-buffer — discovers and runs all test_* functions."""

import importlib
import inspect
import sys
import time
import traceback

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

TEST_MODULES = [
    "tests.test_bytes_ring_buffer",
    "tests.test_bytes_shm_ring_buffer",
    "tests.test_numpy_ring_buffer",
    "tests.test_numpy_shm_ring_buffer",
]


def discover_tests(module):
    """Return all callables whose name starts with 'test_'."""
    return [
        (name, obj)
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("test_")
    ]


def run_tests():
    total = 0
    passed = 0
    failed = 0
    errors: list[tuple[str, str, str]] = []  # (module, test, traceback)

    module_results: list[tuple[str, list[tuple[str, bool, float, str]]]] = []

    for mod_name in TEST_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            tb = traceback.format_exc()
            print(f"\n{RED}  ERROR importing {mod_name}{RESET}")
            print(f"{DIM}{tb}{RESET}")
            errors.append((mod_name, "<import>", tb))
            failed += 1
            total += 1
            continue

        tests = discover_tests(mod)
        results: list[tuple[str, bool, float, str]] = []

        for name, func in tests:
            total += 1
            t0 = time.perf_counter()
            try:
                func()
                elapsed = time.perf_counter() - t0
                passed += 1
                results.append((name, True, elapsed, ""))
            except Exception:
                elapsed = time.perf_counter() - t0
                tb = traceback.format_exc()
                failed += 1
                results.append((name, False, elapsed, tb))
                errors.append((mod_name, name, tb))

        module_results.append((mod_name, results))

    # ── Display ──────────────────────────────────────────────────────
    print()
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  Ring Buffer Test Results{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")

    for mod_name, results in module_results:
        short_name = mod_name.replace("tests.", "")
        mod_passed = sum(1 for _, ok, _, _ in results if ok)
        mod_total = len(results)
        status = f"{GREEN}ALL PASSED{RESET}" if mod_passed == mod_total else f"{RED}{mod_total - mod_passed} FAILED{RESET}"
        print(f"\n  {BOLD}{short_name}{RESET}  ({mod_passed}/{mod_total}) {status}")
        print(f"  {'-' * 60}")

        for name, ok, elapsed, _ in results:
            icon = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
            ms = elapsed * 1000
            print(f"    {icon}  {name:<45} {DIM}{ms:6.1f}ms{RESET}")

    # ── Failure details ──────────────────────────────────────────────
    if errors:
        print(f"\n{BOLD}{'=' * 70}{RESET}")
        print(f"{RED}{BOLD}  Failure Details{RESET}")
        print(f"{BOLD}{'=' * 70}{RESET}")
        for mod_name, test_name, tb in errors:
            print(f"\n  {RED}{mod_name}::{test_name}{RESET}")
            for line in tb.strip().splitlines():
                print(f"    {DIM}{line}{RESET}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    if failed == 0:
        print(f"  {GREEN}{BOLD}{passed}/{total} tests passed{RESET}")
    else:
        print(f"  {RED}{BOLD}{failed}/{total} tests failed{RESET}, {GREEN}{passed} passed{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
