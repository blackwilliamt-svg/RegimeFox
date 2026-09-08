"""Array backend selection and the utilization throttle.

Everything numeric in the optimizer goes through the module-level ``xp`` handle
returned by :func:`get_backend`, which is CuPy when a GPU is usable and NumPy
otherwise. The rest of the package never imports either one directly, so the
same code runs on the RTX 3060 and on a CI box with no CUDA at all.

The throttle is a *fixed* duty-cycle cap, not a temperature-reactive governor.
The point is to let a multi-hour search run in the background while the PC is
being used for something else: at 60% the engine works for 60ms out of every
100ms and sleeps the rest, so the machine stays responsive in a way the user can
predict. A thermal controller would give up that predictability in exchange for
throughput the search does not need.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

log = logging.getLogger(__name__)

# Sleeping in slices smaller than this is dominated by scheduler jitter, so the
# throttle accumulates debt until it is worth paying.
MIN_SLEEP_SECONDS = 0.002


@dataclass(slots=True)
class Backend:
    """The array module in use, plus the transfer helpers that differ per device."""

    name: str
    xp: Any
    device: str = "cpu"

    @property
    def on_gpu(self) -> bool:
        return self.name == "cupy"

    def asarray(self, data: Any, dtype: Any = None) -> Any:
        return self.xp.asarray(data, dtype=dtype)

    def to_numpy(self, arr: Any) -> np.ndarray:
        """Bring an array back to host memory, whichever device it lives on."""
        if self.on_gpu:
            return self.xp.asnumpy(arr)
        return np.asarray(arr)

    def sync(self) -> None:
        """Block until queued device work has finished.

        Timing a GPU section without this measures how fast kernels were
        *queued*, which would make the throttle cap the wrong thing entirely.
        """
        if self.on_gpu:
            try:
                self.xp.cuda.runtime.deviceSynchronize()
            except Exception:  # a driver hiccup must not abort a long search
                pass

    def free_pool(self) -> None:
        """Release cached device blocks between window batches."""
        if not self.on_gpu:
            return
        try:
            self.xp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    def free_bytes(self) -> int | None:
        """Free device memory, or None on CPU where the OS handles paging."""
        if not self.on_gpu:
            return None
        try:
            free, _total = self.xp.cuda.runtime.memGetInfo()
            return int(free)
        except Exception:
            return None


_backend: Backend | None = None


def get_backend(prefer_gpu: bool = True) -> Backend:
    """Process-wide backend. ``SOLOPT_FORCE_CPU=1`` pins it to NumPy."""
    global _backend
    if _backend is not None:
        return _backend

    if prefer_gpu and os.getenv("SOLOPT_FORCE_CPU") != "1":
        try:
            import cupy  # type: ignore

            # Importing CuPy succeeds on a box with no CUDA runtime; allocating
            # is what actually proves the device is usable.
            cupy.zeros(1, dtype=cupy.float32) + 1
            name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
            _backend = Backend("cupy", cupy, device=name)
            log.info("optimizer running on GPU: %s", name)
            return _backend
        except Exception as exc:
            log.info("GPU unavailable (%s); falling back to NumPy", exc)

    _backend = Backend("numpy", np, device="cpu")
    return _backend


def reset_backend_for_tests(name: str = "numpy") -> Backend:
    global _backend
    _backend = Backend("numpy", np, device="cpu") if name == "numpy" else None
    return get_backend(prefer_gpu=name != "numpy")


# --------------------------------------------------------------------------
# Utilization throttle
# --------------------------------------------------------------------------
class Throttle:
    """Cap average utilization at a fixed percentage of wall-clock time.

    Wrap each unit of work in :meth:`segment`; the throttle measures how long it
    took and sleeps proportionally afterwards. At ``percent=100`` it is a no-op
    and costs nothing but a monotonic clock read.
    """

    __slots__ = ("_percent", "_debt", "_busy", "_slept", "_backend")

    def __init__(self, percent: float = 100.0, backend: Backend | None = None) -> None:
        self._percent = self._clamp(percent)
        self._debt = 0.0
        self._busy = 0.0
        self._slept = 0.0
        self._backend = backend

    @staticmethod
    def _clamp(percent: float) -> float:
        try:
            value = float(percent)
        except (TypeError, ValueError):
            return 100.0
        # Below 5% a long search would never finish; above 100 is meaningless.
        return max(5.0, min(100.0, value))

    @property
    def percent(self) -> float:
        return self._percent

    def set_percent(self, percent: float) -> None:
        """Move the slider mid-run; the next segment picks up the new cap."""
        self._percent = self._clamp(percent)

    def segment(self) -> "_Segment":
        return _Segment(self)

    def _record(self, elapsed: float) -> None:
        self._busy += elapsed
        if self._percent >= 100.0:
            return
        # busy/(busy+idle) = p  =>  idle = busy * (100 - p) / p
        self._debt += elapsed * (100.0 - self._percent) / self._percent
        if self._debt >= MIN_SLEEP_SECONDS:
            time.sleep(self._debt)
            self._slept += self._debt
            self._debt = 0.0

    def stats(self) -> dict[str, float]:
        total = self._busy + self._slept
        return {
            "percent_cap": self._percent,
            "busy_seconds": round(self._busy, 3),
            "slept_seconds": round(self._slept, 3),
            "actual_percent": round(100.0 * self._busy / total, 1) if total else 100.0,
        }


class _Segment:
    """Context manager returned by :meth:`Throttle.segment`."""

    __slots__ = ("_throttle", "_started")

    def __init__(self, throttle: Throttle) -> None:
        self._throttle = throttle
        self._started = 0.0

    def __enter__(self) -> "_Segment":
        self._started = time.monotonic()
        return self

    def __exit__(self, *_exc: Any) -> None:
        backend = self._throttle._backend
        if backend is not None:
            backend.sync()
        self._throttle._record(time.monotonic() - self._started)


# --------------------------------------------------------------------------
# Sliding windows - the primitive the bounded forward scan is built on
# --------------------------------------------------------------------------
def sliding_window(backend: Backend, arr: Any, size: int) -> Any:
    """A ``(..., T, size)`` view of ``arr``'s trailing axis, forward-looking.

    Element ``[..., t, k]`` is ``arr[..., t + k]``. The last ``size - 1``
    positions would read past the end, so the caller must mask them; this
    returns a strided *view* rather than a copy, which is what makes the
    forward exit scan affordable at all.
    """
    xp = backend.xp
    arr = xp.ascontiguousarray(arr)
    size = max(1, int(size))
    tail = arr.shape[-1]
    if size > tail:
        raise ValueError(f"window {size} is longer than the {tail}-bar series")
    shape = arr.shape[:-1] + (tail - size + 1, size)
    strides = arr.strides + (arr.strides[-1],)
    return xp.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def chunks(total: int, size: int) -> Iterator[tuple[int, int]]:
    """Half-open ``(start, stop)`` spans covering ``range(total)``."""
    size = max(1, int(size))
    for start in range(0, total, size):
        yield start, min(total, start + size)
