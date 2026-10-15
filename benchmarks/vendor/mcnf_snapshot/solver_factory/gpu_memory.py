from __future__ import annotations

import os
import subprocess
import threading
from types import TracebackType
from typing import Any

_INTERNAL_RECORD_GPU_MEMORY = "_mcnf_record_gpu_memory"


class PeakGpuMemoryMonitor:
    def __init__(self, enabled: bool = True, interval_sec: float = 1.0, pid: int | None = None) -> None:
        self.enabled = enabled
        self.interval_sec = interval_sec
        self.pid = os.getpid() if pid is None else int(pid)
        self.peak_mb = 0.0
        self.sample_count = 0
        self.available = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> PeakGpuMemoryMonitor:
        if self.enabled:
            self._thread = threading.Thread(target=self._run, name="mcnf-gpu-memory", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, _type: type[BaseException] | None, _value: BaseException | None, _traceback: TracebackType | None) -> None:
        if self.enabled:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=3.0)
            self._sample()

    def stats(self) -> dict[str, Any]:
        return {
            "gpu_memory_mb_peak": self.peak_mb if self.available else None,
            "gpu_memory_sample_count": self.sample_count,
            "gpu_memory_monitor_enabled": self.enabled,
        }

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self._sample()

    def _sample(self) -> None:
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.SubprocessError):
            return
        used_mb = 0.0
        matched = False
        for line in completed.stdout.splitlines():
            parts = [item.strip() for item in line.split(",", 1)]
            if len(parts) != 2 or parts[0] != str(self.pid):
                continue
            try:
                used_mb += float(parts[1])
            except ValueError:
                continue
            matched = True
        if not matched:
            return
        self.available = True
        self.sample_count += 1
        self.peak_mb = max(self.peak_mb, used_mb)
