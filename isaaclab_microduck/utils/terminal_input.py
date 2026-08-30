"""Single-keypress reader on stdin, for driving a sim from the terminal.

Ported from `scripts/infer_policy.py` in the mjlab stack, unchanged in behaviour so
the two stacks keep the same feel. Reading the TERMINAL rather than the viewer
window matters here for two reasons:

* Isaac Lab's own keyboard devices (`Se2Keyboard`, `Se3Keyboard`) go through
  `carb` / `omni.appwindow`, so they only receive events under the **Kit**
  visualizer. This works under `--visualizer newton` (and headless, and over SSH).
* Keypresses in a viewer window also fire that viewer's built-in shortcuts.

Arrow keys arrive as ESC [ A/B/C/D escape sequences and are translated to symbolic
names ("up"/"down"/"left"/"right"); letters are lowercased. cbreak (not raw) mode
keeps ISIG enabled, so Ctrl+C still works.
"""

from __future__ import annotations

import os
import queue
import select
import sys
import termios
import threading
import tty


class TerminalInput:
    """Non-blocking keypress reader. Use as a context manager."""

    _ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self.enabled = sys.stdin.isatty()
        self._fd = sys.stdin.fileno() if self.enabled else -1
        self._old_attrs = None
        self._stop = threading.Event()

    def __enter__(self) -> "TerminalInput":
        if not self.enabled:
            print(
                "WARNING: stdin is not a TTY — keyboard control disabled. "
                "Run this in a terminal (not backgrounded, not piped)."
            )
            return self
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        threading.Thread(target=self._reader, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)

    def _read1(self, timeout: float) -> str | None:
        """One byte from stdin, or None on timeout.

        `os.read` (unbuffered): a buffered `sys.stdin.read` would swallow
        escape-sequence bytes past what `select` reported ready.
        """
        ready, _, _ = select.select([self._fd], [], [], timeout)
        if not ready:
            return None
        data = os.read(self._fd, 1)
        return data.decode(errors="ignore") if data else None

    def _reader(self) -> None:
        while not self._stop.is_set():
            ch = self._read1(0.1)
            if not ch:
                continue
            if ch == "\x1b":  # possible arrow-key escape sequence
                if self._read1(0.05) == "[":
                    final = self._read1(0.05)
                    name = self._ARROWS.get(final) if final else None
                    if name:
                        self._queue.put(name)
                continue  # bare ESC / unknown sequence: ignore
            self._queue.put(ch.lower() if ch.isalpha() else ch)

    def get_keys(self) -> list[str]:
        """Drain and return all pending keys (symbolic names / characters)."""
        keys: list[str] = []
        while True:
            try:
                keys.append(self._queue.get_nowait())
            except queue.Empty:
                return keys
