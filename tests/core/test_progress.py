"""cf_lab.progress.FlashProgress: one redrawn bar on a terminal, one line per 10% in a log."""
import io
import sys

import pytest

from cf_lab.progress import FlashProgress

DECK = "Writing to bcAI:gap8 deck memory"
STM32 = "Firmware (1/1) Uploading buffer to stm32..."


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def redirect(monkeypatch, stream):
    # Called in the test body: pytest's capture resets sys.stdout after fixture setup.
    monkeypatch.setattr(sys, "stdout", stream)
    return stream


@pytest.mark.parametrize("message", [DECK, STM32, "Bootloader+softdevice (1/2) Writing buffer to nrf51..."])
def test_piped_transfer_prints_one_line_per_ten_percent(monkeypatch, message):
    out = redirect(monkeypatch, io.StringIO())
    progress = FlashProgress()
    for percent in range(101):
        progress(message, percent)

    lines = out.getvalue().splitlines()
    assert len(lines) == 11
    assert lines[0].startswith("[flash] [------------------------------]   0% ")
    assert lines[-1] == f"[flash] [{'#' * 30}] 100% {message}"


def test_piped_one_off_messages_are_never_dropped(monkeypatch):
    out = redirect(monkeypatch, io.StringIO())
    progress = FlashProgress()
    progress(DECK, 41)
    progress("Error during flash operation (code 5)", 41)
    progress(DECK, 42)

    lines = out.getvalue().splitlines()
    assert lines[1] == "[flash] Error during flash operation (code 5)"
    # A bar after a one-off line starts again, even within the same 10% step.
    assert len(lines) == 3 and "42%" in lines[2]


def test_piped_second_transfer_after_end_prints_again(monkeypatch):
    out = redirect(monkeypatch, io.StringIO())
    progress = FlashProgress()
    for percent in (0, 50, 100):
        progress(DECK, percent)
    progress.end()
    progress(DECK, 100)

    assert out.getvalue().count("100%") == 2


def test_terminal_bar_redraws_one_line_and_end_closes_it(monkeypatch):
    out = redirect(monkeypatch, Terminal())
    progress = FlashProgress()
    progress(DECK, 10)
    progress(DECK, 55)
    progress.end()
    progress.end()

    text = out.getvalue()
    assert "\n" not in text[:-1] and text.endswith("\n")
    assert text.count("\r") == 2
    assert f"[{'#' * 16}{'-' * 14}]  55% {DECK}\x1b[K" in text


def test_terminal_one_off_message_starts_on_its_own_line(monkeypatch):
    out = redirect(monkeypatch, Terminal())
    progress = FlashProgress()
    progress(DECK, 30)
    progress("Deck updated! Restarting...", 100)

    assert out.getvalue().endswith("\x1b[K\n[flash] Deck updated! Restarting...\n")
