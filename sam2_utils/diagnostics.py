"""Memory / disk / file-handle diagnostics for SAM2 sessions.

Use `snapshot("after model load")` between cells to track VRAM/RAM/disk drift,
and `cleanup_vram()` whenever you finish a prediction block before switching
images or starting a video session.

The Windows pagefile readout uses ctypes against kernel32; on Linux/macOS we
fall back to psutil.virtual_memory() and psutil.swap_memory() so the same
function works everywhere.
"""

from __future__ import annotations

import ctypes
import gc
import os
import platform
import sys
import tempfile
from pathlib import Path


# =============================================================================
# Windows-specific RAM/pagefile probe
# =============================================================================

class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength",                 ctypes.c_ulong),
        ("dwMemoryLoad",             ctypes.c_ulong),
        ("ullTotalPhys",             ctypes.c_ulonglong),
        ("ullAvailPhys",             ctypes.c_ulonglong),
        ("ullTotalPageFile",         ctypes.c_ulonglong),
        ("ullAvailPageFile",         ctypes.c_ulonglong),
        ("ullTotalVirtual",          ctypes.c_ulonglong),
        ("ullAvailVirtual",          ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _print_ram():
    """Print RAM + pagefile/swap usage. Windows uses kernel32; else psutil."""
    if platform.system() == "Windows":
        try:
            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            ram_total = stat.ullTotalPhys / 1e9
            ram_used = (stat.ullTotalPhys - stat.ullAvailPhys) / 1e9
            pf_total = stat.ullTotalPageFile / 1e9
            pf_used = (stat.ullTotalPageFile - stat.ullAvailPageFile) / 1e9
            print(f"  [RAM]       used={ram_used:.2f} GB / total={ram_total:.2f} GB")
            print(f"  [Pagefile]  used={pf_used:.2f} GB / total={pf_total:.2f} GB")
            return
        except Exception as e:
            print(f"  [RAM/Pagefile] Windows probe failed ({e})")

    # Non-Windows or Windows-probe-failed fallback
    try:
        import psutil
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        print(f"  [RAM]   used={vm.used / 1e9:.2f} GB / total={vm.total / 1e9:.2f} GB")
        print(f"  [Swap]  used={sw.used / 1e9:.2f} GB / total={sw.total / 1e9:.2f} GB")
    except Exception as e:
        print(f"  [RAM/Swap] unavailable ({e})")


def _print_vram():
    try:
        import torch
    except ImportError:
        print("  [VRAM]      torch not installed")
        return
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserv = torch.cuda.memory_reserved() / 1024**3
        print(f"  [VRAM]      allocated={alloc:.2f} GB  reserved={reserv:.2f} GB")
    else:
        print("  [VRAM]      CUDA not available")


def _dir_size_mb(p: Path) -> float:
    if not p.exists():
        return 0.0
    try:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
    except (PermissionError, OSError):
        return -1.0  # signal "couldn't fully traverse"


def _print_disk():
    nb_dir = Path(".").resolve()
    checkpoint_dir = nb_dir / ".ipynb_checkpoints"
    jupyter_runtime = Path(os.environ.get("JUPYTER_RUNTIME_DIR", tempfile.gettempdir()))
    temp_dir = Path(tempfile.gettempdir())

    print(f"  [Disk] notebook dir:       {_dir_size_mb(nb_dir):.2f} MB  ({nb_dir})")
    print(f"  [Disk] .ipynb_checkpoints: {_dir_size_mb(checkpoint_dir):.2f} MB")
    print(f"  [Disk] Jupyter runtime:    {_dir_size_mb(jupyter_runtime):.2f} MB")
    print(f"  [Disk] system temp:        {_dir_size_mb(temp_dir):.2f} MB")


def _print_open_files():
    try:
        import psutil
        open_files = psutil.Process(os.getpid()).open_files()
        print(f"  [Files] open handles: {len(open_files)}")
        for f in open_files:
            try:
                size_mb = os.path.getsize(f.path) / 1e6
                print(f"    {size_mb:7.2f} MB  {f.path}")
            except OSError:
                print(f"           ?  {f.path}")
    except Exception as e:
        print(f"  [Files] unavailable ({e})")


# =============================================================================
# Public API
# =============================================================================

def snapshot(label: str = "") -> None:
    """Print a unified diagnostics snapshot: RAM, VRAM, disk, file handles."""
    print(f"\n{'=' * 55}")
    print(f"  Snapshot: {label}")
    print(f"{'=' * 55}")
    _print_ram()
    _print_vram()
    _print_disk()
    _print_open_files()
    print()


def cleanup_vram(label: str = "after cleanup") -> None:
    """Run gc + torch.cuda.empty_cache, then print a snapshot.

    Call after finishing a prediction block, before switching to a new image
    or starting a video session. Pair with `predictor.reset_predictor()` if
    you're done with the current image.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    snapshot(label)
