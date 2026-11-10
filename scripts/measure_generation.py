"""Measure a macOS generation process and stop on sustained unsafe pressure."""

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time


_USAGE_FIELDS = """
ri_user_time ri_system_time ri_pkg_idle_wkups ri_interrupt_wkups ri_pageins
ri_wired_size ri_resident_size ri_phys_footprint ri_proc_start_abstime ri_proc_exit_abstime
ri_child_user_time ri_child_system_time ri_child_pkg_idle_wkups ri_child_interrupt_wkups
ri_child_pageins ri_child_elapsed_abstime ri_diskio_bytesread ri_diskio_byteswritten
ri_cpu_time_qos_default ri_cpu_time_qos_maintenance ri_cpu_time_qos_background
ri_cpu_time_qos_utility ri_cpu_time_qos_legacy ri_cpu_time_qos_user_initiated
ri_cpu_time_qos_user_interactive ri_billed_system_time ri_serviced_system_time
ri_logical_writes ri_lifetime_max_phys_footprint ri_instructions ri_cycles
ri_billed_energy ri_serviced_energy ri_interval_max_phys_footprint ri_runnable_time
""".split()


class RUsageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in _USAGE_FIELDS]


class MacProcessUsage:
    def __init__(self):
        self.library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self.library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.library.proc_pid_rusage.restype = ctypes.c_int

    def read(self, pid):
        usage = RUsageInfoV4()
        if self.library.proc_pid_rusage(pid, 4, ctypes.byref(usage)) != 0:
            raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
        return {"rss_bytes": usage.ri_resident_size, "physical_footprint_bytes": usage.ri_phys_footprint,
                "peak_physical_footprint_bytes": usage.ri_lifetime_max_phys_footprint,
                "process_pageins": usage.ri_pageins,
                "disk_read_bytes": usage.ri_diskio_bytesread, "disk_written_bytes": usage.ri_diskio_byteswritten}


def parse_swap_usage(value):
    match = re.search(r"used\s*=\s*([\d.]+)\s*([KMGT]?)", value, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse swap usage: {value}")
    return int(float(match.group(1)) * 1024 ** ("KMGT".index(match.group(2).upper()) + 1 if match.group(2) else 0))


def parse_vm_stat(value):
    page_size = re.search(r"page size of (\d+) bytes", value)
    if not page_size:
        raise ValueError("vm_stat did not report a page size")
    result = {"page_size_bytes": int(page_size.group(1))}
    for line in value.splitlines()[1:]:
        name, separator, count = line.partition(":")
        if separator:
            result[name.strip('" ')] = int(count.strip().rstrip(".").replace(",", ""))
    return result


def system_usage():
    values = subprocess.check_output(["/usr/sbin/sysctl", "-n", "vm.swapusage",
                                       "kern.memorystatus_vm_pressure_level"], text=True, timeout=3).splitlines()
    vm = parse_vm_stat(subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=3))
    return {"swap_used_bytes": parse_swap_usage(values[0]), "swap_raw": values[0],
            "pressure_level": int(values[1]), "vm_stat": vm}


class SafetyGuard:
    def __init__(self, baseline_swap, max_footprint_bytes=14_000_000_000):
        self.baseline_swap = baseline_swap
        self.max_footprint_bytes = max_footprint_bytes
        self.critical_since = None
        self.warning_with_swap_since = None

    def check(self, sample, elapsed):
        process = sample.get("process", {})
        if process.get("physical_footprint_bytes", 0) > self.max_footprint_bytes:
            return "process_physical_footprint_limit"
        system = sample["system"]
        if system["pressure_level"] >= 4:
            if self.critical_since is None:
                self.critical_since = elapsed
            if elapsed - self.critical_since >= 10:
                return "critical_memory_pressure_for_10_seconds"
        else:
            self.critical_since = None
        if system["pressure_level"] >= 2 and system["swap_used_bytes"] - self.baseline_swap >= 2_000_000_000:
            if self.warning_with_swap_since is None:
                self.warning_with_swap_since = elapsed
            if elapsed - self.warning_with_swap_since >= 30:
                return "warning_pressure_and_2GB_swap_growth_for_30_seconds"
        else:
            self.warning_with_swap_since = None
        return None


def stop_process(process):
    for action in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        if process.poll() is not None:
            return
        os.killpg(process.pid, action)
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            pass


def measure(command, output_dir, interval):
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "resources.json").exists():
        raise ValueError("Use a new measurement directory to preserve prior evidence")
    root = Path(__file__).resolve().parents[1]
    baseline = system_usage()
    usage_reader = MacProcessUsage()
    guard = SafetyGuard(baseline["swap_used_bytes"])
    report = {"schema": "pixal3d.process_measurement.v1", "status": "running", "command": command,
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "physical_memory_bytes": int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True)),
              "sampling_interval_seconds": interval, "before": baseline,
              "safety": {"max_process_footprint_bytes": guard.max_footprint_bytes,
                         "critical_pressure_seconds": 10, "warning_with_2GB_swap_growth_seconds": 30}}
    report_path = output_dir / "resources.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    samples = []
    reason = None
    started = time.monotonic()
    with (output_dir / "generation.log").open("w") as log, (output_dir / "resources.jsonl").open("w") as stream:
        process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        report["pid"] = process.pid
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                sample = {"elapsed_seconds": elapsed, "system": system_usage()}
                try:
                    sample["process"] = usage_reader.read(process.pid)
                except OSError as error:
                    sample["process_read_error"] = str(error)
                samples.append(sample)
                stream.write(json.dumps(sample) + "\n")
                stream.flush()
                reason = guard.check(sample, elapsed)
                if reason:
                    stop_process(process)
                    break
                time.sleep(interval)
        except BaseException:
            reason = "measurement_interrupted"
            raise
        finally:
            if process.poll() is None:
                stop_process(process)
            report["returncode"] = process.wait()
            report["elapsed_seconds"] = time.monotonic() - started
            report["after"] = system_usage()
            report["status"] = "stopped_by_guard" if reason else ("complete" if report["returncode"] == 0 else "failed")
            report["stop_reason"] = reason
            processes = [sample["process"] for sample in samples if "process" in sample]
            report["peak_sampled_rss_bytes"] = max((sample["rss_bytes"] for sample in processes), default=None)
            report["peak_process_physical_footprint_bytes"] = max((sample["peak_physical_footprint_bytes"] for sample in processes), default=None)
            report["peak_child_rss_bytes"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            report["peak_system_swap_used_bytes"] = max([baseline["swap_used_bytes"], report["after"]["swap_used_bytes"]] +
                                                         [sample["system"]["swap_used_bytes"] for sample in samples])
            report["system_swap_growth_bytes"] = report["peak_system_swap_used_bytes"] - baseline["swap_used_bytes"]
            report["peak_pressure_level"] = max([baseline["pressure_level"], report["after"]["pressure_level"]] +
                                                 [sample["system"]["pressure_level"] for sample in samples])
            report["sample_count"] = len(samples)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: value for name, value in report.items() if name not in ("before", "after")}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    if sys.platform != "darwin" or arguments.interval <= 0 or not command:
        parser.error("Requires macOS, a positive sampling interval, and a command after --")
    report = measure(command, arguments.output_dir, arguments.interval)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
