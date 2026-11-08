"""Compute devices: the CPU always; a GPU through CUDA or Metal (PyTorch) or Vulkan (tinygrad + wgpu-native).

GPU drivers can crash or hang while being probed, so the probe runs in a short-lived
subprocess (`python -m model_builder probe-gpu --json`) and its answer is cached.
"""

from __future__ import annotations

import functools
import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import paths, system

PROBE_TIMEOUT = 90
CACHE_SECONDS = 7 * 24 * 3600
_lock = threading.Lock()
_gpus: list[dict] | None = None
_probing = False


def _cache_path() -> Path:
    return paths.cache_dir() / "devices.json"


ARM_VENDORS = {0x41: "Arm", 0x42: "Broadcom", 0x46: "Fujitsu", 0x48: "HiSilicon", 0x4E: "NVIDIA", 0x51: "Qualcomm",
               0x53: "Samsung", 0x61: "Apple", 0x6D: "Microsoft", 0xC0: "Ampere"}
ARM_CORES = {
    (0x41, 0xD03): "Cortex-A53", (0x41, 0xD04): "Cortex-A35", (0x41, 0xD05): "Cortex-A55", (0x41, 0xD07): "Cortex-A57",
    (0x41, 0xD08): "Cortex-A72", (0x41, 0xD09): "Cortex-A73", (0x41, 0xD0A): "Cortex-A75", (0x41, 0xD0B): "Cortex-A76",
    (0x41, 0xD0C): "Neoverse N1", (0x41, 0xD0D): "Cortex-A77", (0x41, 0xD40): "Neoverse V1", (0x41, 0xD41): "Cortex-A78",
    (0x41, 0xD44): "Cortex-X1", (0x41, 0xD46): "Cortex-A510", (0x41, 0xD47): "Cortex-A710", (0x41, 0xD48): "Cortex-X2",
    (0x41, 0xD49): "Neoverse N2", (0x41, 0xD4D): "Cortex-A715", (0x41, 0xD4E): "Cortex-X3", (0x41, 0xD4F): "Neoverse V2",
    (0x41, 0xD80): "Cortex-A520", (0x41, 0xD81): "Cortex-A720", (0x41, 0xD82): "Cortex-X4", (0x41, 0xD85): "Cortex-X925",
    (0x41, 0xD87): "Cortex-A725", (0x51, 0x001): "Oryon", (0x51, 0x002): "Oryon",
}


def _arm_cpu_name(text: str) -> str | None:
    """Name ARM cores from the implementer/part ids in /proc/cpuinfo (it has no model name there)."""
    names, implementer = [], None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        try:
            if key.strip() == "CPU implementer":
                implementer = int(value.strip(), 16)
            elif key.strip() == "CPU part" and implementer is not None:
                part = int(value.strip(), 16)
                name = ARM_CORES.get((implementer, part)) or f"{ARM_VENDORS.get(implementer, 'ARM')} core"
                if implementer != 0x41 and not name.endswith("core"):
                    name = f"{ARM_VENDORS.get(implementer, '')} {name}".strip()
                if name not in names:
                    names.append(name)
        except ValueError:
            continue
    return " + ".join(names) or None


@functools.lru_cache(maxsize=None)
def cpu_name() -> str:
    if system.is_macos():
        name = system.sysctl("machdep.cpu.brand_string")  # e.g. "Apple M2 Pro"
        if name:
            return name
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        for key in ("model name", "Hardware", "Model"):
            for line in text.splitlines():
                if line.startswith(key) and ":" in line:
                    value = line.split(":", 1)[1].strip()
                    if value:
                        return value
        arm = _arm_cpu_name(text)
        if arm:
            return arm
    except OSError:
        pass
    return platform.processor() or platform.machine() or "CPU"


def cpu_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def cpu_device() -> dict:
    return {"id": "cpu", "kind": "cpu", "name": cpu_name(), "label": "CPU",
            "detail": f"{cpu_cores()} cores · PyTorch {_torch_version()}"}


def _torch_version() -> str:
    try:
        from importlib.metadata import version

        return version("torch").split("+")[0]
    except Exception:  # noqa: BLE001
        return ""


def _apple_gpu_name() -> str:
    """The Mac's GPU, e.g. "Apple M2 Pro (19-core GPU)"; a discrete GPU first on Intel Macs."""
    try:
        completed = subprocess.run(["system_profiler", "SPDisplaysDataType", "-json"], capture_output=True,
                                   text=True, timeout=30)
        entries = [entry for entry in json.loads(completed.stdout).get("SPDisplaysDataType", []) if entry.get("sppci_model")]
        entries.sort(key=lambda entry: entry.get("sppci_bus") != "spdisplays_pcie_device")
        if entries:
            cores = entries[0].get("sppci_cores")
            return f"{entries[0]['sppci_model']} ({cores}-core GPU)" if cores else entries[0]["sppci_model"]
    except (OSError, ValueError, AttributeError, subprocess.TimeoutExpired):
        pass
    return f"{cpu_name()} GPU"


def _probe_mps(torch) -> dict | None:
    """A Mac GPU through PyTorch's Metal (MPS) backend, checked with one small training step."""
    backend = getattr(torch.backends, "mps", None)
    if backend is None or not backend.is_available():
        return None
    weights = torch.randn(64, 64, device="mps", requires_grad=True)
    torch.tanh(weights @ weights).square().mean().backward()
    if not bool(torch.isfinite(weights.grad).all()):
        raise RuntimeError("the Metal GPU returned invalid numbers")
    try:
        memory = int(torch.mps.recommended_max_memory())
    except Exception:  # noqa: BLE001 - older PyTorch: report the whole unified memory
        memory = int(system.sysctl("hw.memsize") or 0)
    return {"name": _apple_gpu_name(), "memory": memory}


def probe_here() -> dict:
    """Detect GPUs in this process (used by the probe subprocess)."""
    result: dict = {"cuda": None, "mps": None, "vulkan": None, "errors": {}}
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            result["cuda"] = {"name": properties.name, "memory": int(properties.total_memory)}
    except Exception as error:  # noqa: BLE001
        result["errors"]["cuda"] = str(error)[:200]
    if system.is_macos():
        try:
            import torch

            result["mps"] = _probe_mps(torch)
        except Exception as error:  # noqa: BLE001
            result["errors"]["mps"] = str(error)[:200]
    elif os.environ.get("LMB_DISABLE_VULKAN") != "1":
        try:
            from .gpu_model import gpu_available

            result["vulkan"] = gpu_available()
        except Exception as error:  # noqa: BLE001
            result["errors"]["vulkan"] = str(error)[:200]
    return result


def _gpu_entries(probe: dict) -> list[dict]:
    entries = []
    if probe.get("cuda"):
        cuda = probe["cuda"]
        entries.append({"id": "cuda", "kind": "gpu", "name": cuda["name"], "label": "GPU (CUDA)",
                        "detail": f"{cuda.get('memory', 0) / 2**30:.0f} GB · PyTorch CUDA"})
    if probe.get("mps"):
        mps = probe["mps"]
        memory = f"{mps['memory'] / 2**30:.0f} GB shared · " if mps.get("memory") else ""
        entries.append({"id": "mps", "kind": "gpu", "name": mps.get("name") or "Apple GPU", "label": "GPU (Metal)",
                        "detail": f"{memory}PyTorch MPS"})
    if probe.get("vulkan"):
        vulkan = probe["vulkan"]
        entries.append({"id": "vulkan", "kind": "gpu", "name": vulkan.get("name") or "GPU", "label": "GPU (Vulkan)",
                        "detail": f"{vulkan.get('backend', 'Vulkan')} · tinygrad + wgpu"})
    return entries


def _python_for_engine() -> str:
    return sys.executable


def in_proot() -> bool:
    """True inside PRoot (e.g. a Linux distribution running on an Android phone)."""
    if os.environ.get("PROOT_TMP_DIR") or os.environ.get("PROOT_LOADER"):
        return True
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("TracerPid:"):
                tracer = int(line.split()[1])
                return bool(tracer) and "proot" in Path(f"/proc/{tracer}/comm").read_text(errors="replace")
    except (OSError, ValueError):
        pass
    return False


def engine_env() -> dict:
    """Environment for engine subprocesses.

    Under PRoot, LD_PRELOAD usually carries host shims that can break GPU drivers and browsers in
    child processes, so it is dropped there (LMB_KEEP_LD_PRELOAD=1 keeps it; LMB_DROP_LD_PRELOAD=1
    drops it anywhere).
    """
    env = dict(os.environ)
    keep = os.environ.get("LMB_KEEP_LD_PRELOAD") == "1"
    if not keep and (os.environ.get("LMB_DROP_LD_PRELOAD") == "1" or in_proot()):
        env.pop("LD_PRELOAD", None)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if system.is_macos():
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # the rare op Metal lacks runs on the CPU instead
    return env


def run_probe() -> list[dict]:
    """Probe in a subprocess; returns GPU device entries (possibly empty)."""
    try:
        completed = subprocess.run(
            [_python_for_engine(), "-m", "model_builder", "probe-gpu", "--json"],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT, env=engine_env(),
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        line = next((line for line in reversed(completed.stdout.splitlines()) if line.startswith("{")), "{}")
        probe = json.loads(line)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        probe = {}
    entries = _gpu_entries(probe)
    try:
        _cache_path().parent.mkdir(parents=True, exist_ok=True)
        _cache_path().write_text(json.dumps({"time": time.time(), "python": sys.executable, "gpus": entries}), encoding="utf-8")
    except OSError:
        pass
    return entries


def _cached() -> list[dict] | None:
    try:
        cached = json.loads(_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if cached.get("python") != sys.executable or time.time() - cached.get("time", 0) > CACHE_SECONDS:
        return None
    return cached.get("gpus")


def gpus(wait: bool = False) -> list[dict]:
    global _gpus
    with _lock:
        if _gpus is None:
            _gpus = _cached()
    if _gpus is None and wait:
        refresh()
    return list(_gpus or [])


def refresh(on_change: Callable[[], None] | None = None, background: bool = False) -> None:
    """Re-probe GPUs (optionally in a thread) and call on_change if the list changed."""
    global _probing

    def work() -> None:
        global _gpus, _probing
        try:
            entries = run_probe()
            with _lock:
                changed = entries != _gpus
                _gpus = entries
            if changed and on_change:
                on_change()
        finally:
            _probing = False

    with _lock:
        if _probing:
            return
        _probing = True
    if background:
        threading.Thread(target=work, name="gpu-probe", daemon=True).start()
    else:
        work()


def probing() -> bool:
    return _probing


def available() -> list[dict]:
    return [cpu_device(), *gpus()]


def available_ids() -> list[str]:
    return [device["id"] for device in available()]


def resolve(setting: str, estimates: dict | None = None) -> str:
    """The device a job should use for a device setting ("auto", "cpu", "vulkan", "cuda", "mps")."""
    ids = available_ids()
    if setting in ids:
        return setting
    if estimates and estimates.get("calibrated") and estimates.get("auto_device") in ids:
        return estimates["auto_device"]
    if "cuda" in ids:
        return "cuda"
    if "mps" in ids and platform.machine() == "arm64":
        return "mps"  # Apple silicon; on an Intel Mac, calibrate first: its GPU can be slower than the CPU
    return "cpu"
