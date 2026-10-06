"""Flash progress: one bar that redraws in place, instead of a line per percent."""

from __future__ import annotations

import re
import sys

# cflib's transfer messages: deck-memory writes ("Writing to bcAI:gap8 deck
# memory") and STM32/nRF51 page writes ("Firmware (1/1) Uploading buffer to
# stm32..."). Everything else a flasher reports is a one-off line.
TRANSFER = re.compile(r"Writing to |(Firmware|Bootloader\+softdevice) \(\d+/\d+\) ")
BAR_WIDTH = 30


class FlashProgress:
    """A flasher ``progress_cb`` that draws the transfer as a single bar.

    On a terminal the bar redraws one line, with cflib's current message after
    it. Written to a file or pipe, it prints one bar line per 10% instead, since
    a carriage return there would just pile up.
    """

    def __init__(self, stream=None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._tty = getattr(self._stream, "isatty", lambda: False)()
        self._bar_open = False
        self._last_step = -1

    def __call__(self, message: str, percent: int) -> None:
        if TRANSFER.match(message):
            self._bar(message, percent)
        else:
            self.line(message)

    def line(self, message: str) -> None:
        self.end()
        self._stream.write(f"[flash] {message}\n")
        self._stream.flush()

    def end(self) -> None:
        """Finish a bar that is still being drawn, so the next output starts clean."""
        if self._bar_open:
            self._stream.write("\n")
            self._bar_open = False
        self._last_step = -1

    def _bar(self, message: str, percent: int) -> None:
        filled = BAR_WIDTH * max(0, min(percent, 100)) // 100
        text = f"[flash] [{'#' * filled}{'-' * (BAR_WIDTH - filled)}] {percent:3d}% {message}"
        if self._tty:
            # \x1b[K clears what a longer previous message left behind.
            self._stream.write(f"\r{text}\x1b[K")
            self._bar_open = True
        else:
            step = percent // 10
            if step == self._last_step:
                return
            self._last_step = step
            self._stream.write(text + "\n")
        self._stream.flush()
