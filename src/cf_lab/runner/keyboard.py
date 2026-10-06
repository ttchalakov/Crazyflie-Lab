from __future__ import annotations

import os
import select
import sys
import termios
import tty


class KeyboardMonitor:
    """Non-blocking single-key reader; inert when stdin is not a TTY."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._saved: list | None = None

    def __enter__(self) -> "KeyboardMonitor":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        if self._fd is not None and self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def pressed(self) -> str:
        if self._fd is None:
            return ""
        # The raw descriptor: sys.stdin.read(1) buffers every key of a burst, and
        # select() then no longer sees the rest, a SPACE among them.
        keys = b""
        while select.select([self._fd], [], [], 0)[0]:
            chunk = os.read(self._fd, 1024)
            if not chunk:  # stdin closed
                break
            keys += chunk
        return keys.decode(errors="replace")
