import os
import sys

import pytest

from scripts.measure_generation import MacProcessUsage, SafetyGuard, parse_swap_usage, parse_vm_stat


@pytest.mark.parametrize("text,expected", [
    ("total = 6144.00M  used = 4799.94M  free = 1344.06M", int(4799.94 * 1024 ** 2)),
    ("total = 8G used = 1.5G", int(1.5 * 1024 ** 3)),
    ("used = 0M", 0),
])
def test_swap_parser_uses_binary_units(text, expected):
    assert parse_swap_usage(text) == expected


def test_vm_stat_parser_preserves_page_counts_and_size():
    result = parse_vm_stat('Mach Virtual Memory Statistics: (page size of 16384 bytes)\nSwapouts: 42.\n"Translation faults": 2,000.\n')
    assert result == {"page_size_bytes": 16384, "Swapouts": 42, "Translation faults": 2000}


def test_guard_allows_preexisting_swap_and_transient_pressure():
    guard = SafetyGuard(5_000_000_000)
    warning = {"system": {"pressure_level": 4, "swap_used_bytes": 5_000_000_000}}
    normal = {"system": {"pressure_level": 1, "swap_used_bytes": 5_000_000_000}}
    assert guard.check(warning, 0) is None
    assert guard.check(warning, 9) is None
    assert guard.check(normal, 10) is None
    assert guard.check(warning, 12) is None
    assert guard.check(warning, 22) == "critical_memory_pressure_for_10_seconds"


def test_guard_stops_on_large_footprint_or_sustained_warning_and_swap_growth():
    guard = SafetyGuard(1_000_000_000)
    large = {"process": {"physical_footprint_bytes": 14_000_000_001},
             "system": {"pressure_level": 1, "swap_used_bytes": 1_000_000_000}}
    assert guard.check(large, 0) == "process_physical_footprint_limit"
    warning = {"system": {"pressure_level": 2, "swap_used_bytes": 3_000_000_000}}
    assert guard.check(warning, 10) is None
    assert guard.check(warning, 40) == "warning_pressure_and_2GB_swap_growth_for_30_seconds"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS process accounting API")
def test_macos_process_usage_matches_live_process():
    usage = MacProcessUsage().read(os.getpid())
    assert usage["rss_bytes"] > 0
    assert usage["physical_footprint_bytes"] > 0
    assert usage["peak_physical_footprint_bytes"] >= usage["physical_footprint_bytes"]
