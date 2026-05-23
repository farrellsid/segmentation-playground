"""
diag_utils.py
-------------
Memory and resource diagnostics for SAM2 segmentation pipeline.

Functions:
    snapshot(label)   - Print RAM, pagefile, VRAM, disk, and open file handles.
    cleanup_vram()    - Force VRAM release and print a post-cleanup snapshot.

Usage:
    from diag_utils import snapshot, cleanup_vram
"""

import os
import gc
import ctypes
import psutil
import tempfile
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Windows MEMORYSTATUSEX structure (used by snapshot)
# ---------------------------------------------------------------------------

class MEMORYSTATUSEX(ctypes.Structure):
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def snapshot(label: str = "") -> None:
    """
    Unified diagnostics snapshot. Reports:
      - RAM usage and pagefile (Windows only)
      - VRAM allocated / reserved (CUDA only)
      - Disk usage: notebook dir, checkpoints, Jupyter runtime, system temp
      - Open file handles for this process
    """
    print(f"\n{'='*55}")
    print(f"  Snapshot: {label}")
    print(f"{'='*55}")

    # --- RAM + Pagefile (Windows) ---
    try:
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        ram_total = stat.ullTotalPhys / 1e9
        ram_used  = (stat.ullTotalPhys - stat.ullAvailPhys) / 1e9
        pf_total  = stat.ullTotalPageFile / 1e9
        pf_used   = (stat.ullTotalPageFile - stat.ullAvailPageFile) / 1e9
        print(f"  [RAM]       used={ram_used:.2f} GB / total={ram_total:.2f} GB")
        print(f"  [Pagefile]  used={pf_used:.2f} GB / total={pf_total:.2f} GB")
    except Exception as e:
        print(f"  [RAM/Pagefile] unavailable ({e})")

    # --- VRAM (CUDA) ---
    if torch.cuda.is_available():
        alloc  = torch.cuda.memory_allocated() / 1024**3
        reserv = torch.cuda.memory_reserved()  / 1024**3
        print(f"  [VRAM]      allocated={alloc:.2f} GB  reserved={reserv:.2f} GB")
    else:
        print("  [VRAM]      CUDA not available")

    # --- Disk locations ---
    nb_dir          = Path(".").resolve()
    checkpoint_dir  = nb_dir / ".ipynb_checkpoints"
    jupyter_runtime = Path(os.environ.get("JUPYTER_RUNTIME_DIR", tempfile.gettempdir()))
    temp_dir        = Path(tempfile.gettempdir())

    def dir_size(p: Path) -> float:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6 if p.exists() else 0.0

    print(f"  [Disk] notebook dir:       {dir_size(nb_dir):.2f} MB  ({nb_dir})")
    print(f"  [Disk] .ipynb_checkpoints: {dir_size(checkpoint_dir):.2f} MB")
    print(f"  [Disk] Jupyter runtime:    {dir_size(jupyter_runtime):.2f} MB")
    print(f"  [Disk] system temp:        {dir_size(temp_dir):.2f} MB")

    # --- Open file handles ---
    try:
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

    print()


def cleanup_vram() -> None:
    """Force VRAM release via gc + empty_cache, then print a snapshot."""
    gc.collect()
    torch.cuda.empty_cache()
    snapshot("after cleanup")