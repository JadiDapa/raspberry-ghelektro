"""
System info — reads Raspberry Pi hardware stats.
All reads are synchronous but fast (< 5ms each).
Uses only stdlib — no extra dependencies needed.
"""

import os
import time
import platform
import subprocess


_boot_time = time.time()


def cpu_percent() -> float:
    """CPU usage % averaged over a short interval via /proc/stat."""
    try:

        def _read_stat():
            with open("/proc/stat") as f:
                line = f.readline()
            vals = list(map(int, line.split()[1:]))
            idle = vals[3]
            total = sum(vals)
            return idle, total

        idle1, total1 = _read_stat()
        time.sleep(0.1)
        idle2, total2 = _read_stat()

        idle_delta = idle2 - idle1
        total_delta = total2 - total1
        if total_delta == 0:
            return 0.0
        return round((1 - idle_delta / total_delta) * 100, 1)
    except Exception:
        return 0.0


def memory_info() -> dict:
    """RAM usage from /proc/meminfo. Returns MB values."""
    try:
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, val = line.split(":")
                mem[key.strip()] = int(val.split()[0])  # kB

        total = mem.get("MemTotal", 0)
        available = mem.get("MemAvailable", 0)
        used = total - available

        return {
            "total_mb": round(total / 1024, 1),
            "used_mb": round(used / 1024, 1),
            "available_mb": round(available / 1024, 1),
            "percent": round(used / total * 100, 1) if total else 0.0,
        }
    except Exception:
        return {"total_mb": 0, "used_mb": 0, "available_mb": 0, "percent": 0.0}


def disk_info(path: str = "/") -> dict:
    """Disk usage for the given path. Returns GB values."""
    try:
        st = os.statvfs(path)  # type: ignore
        total = st.f_blocks * st.f_frsize
        free = st.f_bfree * st.f_frsize
        used = total - free

        return {
            "total_gb": round(total / 1024**3, 2),
            "used_gb": round(used / 1024**3, 2),
            "free_gb": round(free / 1024**3, 2),
            "percent": round(used / total * 100, 1) if total else 0.0,
        }
    except Exception:
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "percent": 0.0}


def cpu_temperature() -> float | None:
    """
    CPU temperature in °C.
    Reads from thermal zone (Linux) — works on Raspberry Pi.
    Returns None if not available (e.g. on a dev machine).
    """
    try:
        # Primary: Raspberry Pi thermal zone
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read().strip()) / 1000, 1)
    except Exception:
        pass

    try:
        # Fallback: vcgencmd (Pi-specific)
        result = subprocess.run(
            ["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=2
        )
        temp_str = result.stdout.strip()  # e.g. "temp=45.6'C"
        return float(temp_str.replace("temp=", "").replace("'C", ""))
    except Exception:
        return None


def uptime_seconds() -> float:
    """Seconds since this FastAPI process started."""
    return round(time.time() - _boot_time, 1)


def system_uptime_seconds() -> float:
    """Seconds since the Pi itself booted (from /proc/uptime)."""
    try:
        with open("/proc/uptime") as f:
            return float(f.read().split()[0])
    except Exception:
        return 0.0


def get_all() -> dict:
    """Collect all system stats in one call."""
    temp = cpu_temperature()
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "cpu_percent": cpu_percent(),
        "cpu_temp_c": temp,
        "memory": memory_info(),
        "disk": disk_info("/"),
        "process_uptime_sec": uptime_seconds(),
        "system_uptime_sec": system_uptime_seconds(),
    }
