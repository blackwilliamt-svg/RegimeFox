"""Single-instance lock for the worker.

Two engines running against one wallet would double every position, so the
worker refuses to start if another one already holds the lock. systemd's
``Restart=on-failure`` makes this a real risk: a slow shutdown overlapping a
restart is exactly the window where two loops could briefly coexist.

Uses ``fcntl.flock`` on Linux (the deployment target) and ``msvcrt.locking`` on
Windows for local development. Both release automatically if the process dies,
which a bare PID file does not.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import TracebackType


class AlreadyRunning(RuntimeError):
    pass


class ProcessLock:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+")
        try:
            if sys.platform == "win32":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._fh.close()
            self._fh = None
            existing = ""
            try:
                existing = self.path.read_text(encoding="utf-8").strip()
            except OSError:
                pass
            raise AlreadyRunning(
                f"another worker already holds {self.path}"
                + (f" (pid {existing})" if existing else "")
                + ". Two engines on one wallet would double every position."
            ) from exc

        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(str(os.getpid()))
        self._fh.flush()

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "ProcessLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
