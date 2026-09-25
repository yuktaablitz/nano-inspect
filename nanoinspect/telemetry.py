"""GPU telemetry (nvidia-smi), latency statistics, and energy measurement."""
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import psutil

QUERY = "temperature.gpu,clocks.sm,power.draw,utilization.gpu,clocks_event_reasons.active"


def _num(v):
    try:
        return float(v)
    except ValueError:
        return np.nan


def gpu_sample():
    out = subprocess.run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=5).stdout.strip().split(", ")
    return {"temp_c": _num(out[0]), "sm_clock_mhz": _num(out[1]), "power_w": _num(out[2]),
            "gpu_util_pct": _num(out[3]),
            "throttle_active": out[4].strip() not in ("0x0000000000000000", "[N/A]", "N/A"),
            "system_ram_used_gb": psutil.virtual_memory().used / 1e9}


def other_gpu_processes():
    import os
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return [a for a in out if a and not a.startswith(str(os.getpid()))]


class GpuTelemetry:
    """Background sampler. Use as a context manager; results in .df after exit."""

    def __init__(self, interval_s=1.0):
        self.interval_s = interval_s; self.rows = []

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.rows.append({"t_s": time.perf_counter() - self.t0, **gpu_sample()})
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self.t0 = time.perf_counter(); self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True); self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set(); self._thread.join(); self.df = pd.DataFrame(self.rows)

    def energy_j(self):
        """Integrated GPU power over the sampled window (trapezoid rule)."""
        d = self.df.dropna(subset=["power_w"])
        return float(np.trapezoid(d.power_w, d.t_s)) if len(d) > 1 else np.nan


def latency_stats(ms):
    ms = np.asarray(ms, dtype=float)
    return {"n": int(ms.size), "p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)),
            "p99": float(np.percentile(ms, 99)), "mean": float(ms.mean()), "max": float(ms.max())}


def idle_power(seconds=5):
    with GpuTelemetry(0.5) as t:
        time.sleep(seconds)
    return float(t.df.power_w.mean())


def measure_energy(fn, n_items, idle_w):
    """Run fn() under telemetry. Returns energy per item: total GPU energy and energy above idle."""
    with GpuTelemetry(0.25) as t:
        t0 = time.perf_counter(); fn(); secs = time.perf_counter() - t0
    mean_w = float(t.df.power_w.mean())
    total_j = mean_w * secs
    return {"items": n_items, "seconds": round(secs, 2), "items_per_s": round(n_items / secs, 2),
            "mean_gpu_power_w": round(mean_w, 1), "idle_gpu_power_w": round(idle_w, 1),
            "joules_per_item": total_j / n_items, "joules_per_item_above_idle": max(mean_w - idle_w, 0) * secs / n_items,
            "items_per_wh": 3600 * n_items / total_j}
