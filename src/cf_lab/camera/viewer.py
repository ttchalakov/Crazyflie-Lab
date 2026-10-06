"""Live view of an AI-deck image stream (PyQt6).

``Viewer`` is a frame sink on a ``CameraFeed``: raw grayscale or JPEG frames of
any size, as ``lib/gap8/imgstream`` sends them. The receive thread only drops
each frame into a one-frame slot; a decode thread turns the newest one into
pixels, and the window paints whatever is newest when it gets to it. A slow
repaint therefore skips frames and never backs up the feed.

Qt owns the main thread while the window is open (``Viewer.run``), so anything
else the caller does -- recording, watching for errors -- runs in threads or in
the ``should_stop`` callback, which the window polls.

Keys: F fullscreen, Space pause, S snapshot, 1 fit or actual size, M smooth or
sharp pixels, P side panel, Q or Esc quit.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from cf_lab.camera.ext import DeckLine
from cf_lab.camera.feed import CapturedFrame
from cf_lab.camera.stats import FeedStats, StatsSnapshot

# A frame older than this reads as a stalled stream in the window.
STALL_S = 1.0

_STYLE = """
QMainWindow { background: #16181c; }
QWidget#panel { background: #16181c; border-left: 1px solid #262a31; }
QLabel { color: #c9ced6; font-size: 12px; }
QLabel[role="section"] { color: #7d8590; font-size: 10px; font-weight: 600;
                         letter-spacing: 1px; padding-top: 10px; }
QLabel[role="value"] { color: #eef1f5; font-family: "DejaVu Sans Mono", monospace; }
QLabel[role="alert"] { color: #f0b429; font-family: "DejaVu Sans Mono", monospace; }
QStatusBar { background: #101215; color: #9aa3ad; font-size: 11px; }
QStatusBar QLabel { color: #9aa3ad; font-size: 11px; }
"""


class Viewer:
    """Show the newest frame from a feed, with its metrics beside it."""

    def __init__(
        self,
        *,
        title: str = "AI-deck camera",
        snapshot_dir: str | Path = ".",
        status: Callable[[], str] | None = None,
    ) -> None:
        """``status`` supplies extra status-bar text, polled a few times a second."""
        self.title = title
        self.snapshot_dir = Path(snapshot_dir)
        self.status = status
        self.stats = FeedStats()
        self.deck_config: dict[str, int | str] = {}
        self.deck_status: dict[str, int | str] = {}
        self.message = ""
        self._slot_lock = threading.Lock()
        self._pending: CapturedFrame | None = None
        self._frame_ready = threading.Event()
        self._image_lock = threading.Lock()
        self._image = None  # (ndarray, CapturedFrame), newest decoded
        self._image_seq = 0
        self._stop = threading.Event()

    # -- sink side: the feed's receive thread ---------------------------------

    def offer(self, frame: CapturedFrame) -> None:
        self.stats.offer(frame)
        with self._slot_lock:
            self._pending = frame
        self._frame_ready.set()

    def offer_console(self, line: DeckLine) -> None:
        if line.kind == "config":
            self.deck_config = line.fields
        elif line.kind == "status":
            self.deck_status = line.fields

    def show_message(self, text: str) -> None:
        """Text shown over the image area, e.g. a connection error."""
        self.message = text

    # -- decode thread ----------------------------------------------------------

    def _decode_loop(self) -> None:
        while not self._stop.is_set():
            if not self._frame_ready.wait(0.2):
                continue
            self._frame_ready.clear()
            with self._slot_lock:
                frame, self._pending = self._pending, None
            if frame is None:
                continue
            try:
                image = frame.decode()
            except Exception:
                image = None
            if image is None:
                continue
            with self._image_lock:
                self._image = (image, frame)
                self._image_seq += 1

    def latest_image(self):
        """``(seq, ndarray, frame)`` for the newest decoded frame, or None."""
        with self._image_lock:
            if self._image is None:
                return None
            return self._image_seq, *self._image

    # -- window ---------------------------------------------------------------

    def run(self, should_stop: Callable[[], bool] = lambda: False) -> None:
        """Open the window and block until it is closed or ``should_stop()`` is true."""
        app = _qt_app()
        decoder = threading.Thread(target=self._decode_loop, name="viewer-decode", daemon=True)
        decoder.start()
        window = _Window(self, should_stop)
        window.show()
        try:
            app.exec()
        finally:
            self._stop.set()
            decoder.join(timeout=1.0)


def _qt_app():
    # opencv-python points Qt at its own bundled Qt5 plugins when cv2 is imported;
    # PyQt6 must load its own.
    for name in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        if "cv2" in os.environ.get(name, ""):
            del os.environ[name]
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(_STYLE)
    return app


def _Window(viewer: Viewer, should_stop: Callable[[], bool]):
    """Build the main window. Qt classes are defined here so importing this
    module never needs PyQt6 (headless runs and the recorder do not)."""
    from PyQt6.QtCore import QRectF, Qt, QTimer
    from PyQt6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QShortcut
    from PyQt6.QtWidgets import (
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QVBoxLayout,
        QWidget,
    )

    class ImageView(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setMinimumSize(320, 240)
            self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
            self.image: QImage | None = None
            self.pixels = None  # keeps the ndarray behind self.image alive
            self.fit = True
            self.smooth = True
            self.paused = False
            self.chip = ""
            self.alert = ""

        def set_frame(self, pixels) -> None:
            height, width = pixels.shape
            self.pixels = pixels
            self.image = QImage(pixels.data, width, height, pixels.strides[0],
                                QImage.Format.Format_Grayscale8)
            self.update()

        def paintEvent(self, _event) -> None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("#0b0c0e"))
            if self.image is not None:
                w, h = self.image.width(), self.image.height()
                if self.fit:
                    scale = min(self.width() / w, self.height() / h)
                else:
                    scale = 1.0
                target = QRectF((self.width() - w * scale) / 2,
                                (self.height() - h * scale) / 2, w * scale, h * scale)
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.smooth)
                painter.drawImage(target, self.image)
            self._draw_chip(painter, self.chip, 12, 12, "#e6e9ee")
            if self.paused:
                self._draw_chip(painter, "PAUSED", 12, 42, "#f0b429")
            if self.alert:
                painter.setFont(QFont("DejaVu Sans", 13))
                painter.setPen(QColor("#f0b429"))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter
                                 | Qt.TextFlag.TextWordWrap, self.alert)
            painter.end()

        @staticmethod
        def _draw_chip(painter: QPainter, text: str, x: int, y: int, color: str) -> None:
            if not text:
                return
            painter.setFont(QFont("DejaVu Sans Mono", 10))
            metrics = painter.fontMetrics()
            box = QRectF(x, y, metrics.horizontalAdvance(text) + 16, metrics.height() + 8)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 150))
            painter.drawRoundedRect(box, 6, 6)
            painter.setPen(QColor(color))
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    class Panel(QWidget):
        """Label/value rows in titled sections, rebuilt only when rows change."""

        def __init__(self) -> None:
            super().__init__()
            self.setObjectName("panel")
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
            self.setFixedWidth(270)
            self.grid = QGridLayout()
            self.grid.setContentsMargins(14, 6, 14, 14)
            self.grid.setVerticalSpacing(3)
            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.addLayout(self.grid)
            outer.addStretch(1)
            self.rows: dict[tuple[str, str], QLabel] = {}
            self.layout_key: tuple = ()

        def show_sections(self, sections: list[tuple[str, list[tuple[str, str, bool]]]]) -> None:
            key = tuple((title, tuple(label for label, _, _ in rows)) for title, rows in sections)
            if key != self.layout_key:
                self._rebuild(sections)
                self.layout_key = key
            for title, rows in sections:
                for label, value, alert in rows:
                    cell = self.rows[(title, label)]
                    cell.setText(value)
                    role = "alert" if alert else "value"
                    if cell.property("role") != role:
                        cell.setProperty("role", role)
                        cell.style().unpolish(cell)
                        cell.style().polish(cell)

        def _rebuild(self, sections) -> None:
            while self.grid.count():
                item = self.grid.takeAt(0)
                if item.widget() is not None:
                    item.widget().deleteLater()
            self.rows.clear()
            row = 0
            for title, rows in sections:
                heading = QLabel(title.upper())
                heading.setProperty("role", "section")
                self.grid.addWidget(heading, row, 0, 1, 2)
                row += 1
                for label, _, _ in rows:
                    self.grid.addWidget(QLabel(label), row, 0)
                    value = QLabel("")
                    value.setProperty("role", "value")
                    value.setAlignment(Qt.AlignmentFlag.AlignRight)
                    self.grid.addWidget(value, row, 1)
                    self.rows[(title, label)] = value
                    row += 1

    class Window(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(viewer.title)
            self.resize(980, 600)
            self.view = ImageView()
            self.panel = Panel()
            body = QWidget()
            layout = QHBoxLayout(body)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            layout.addWidget(self.view, 1)
            layout.addWidget(self.panel)
            self.setCentralWidget(body)
            self.state = QLabel("")
            self.statusBar().addWidget(self.state)
            self.left = QLabel("")
            self.statusBar().addWidget(self.left, 1)
            self.statusBar().addPermanentWidget(QLabel(
                "F fullscreen · Space pause · S snapshot · 1 size · M smoothing · P panel · Q quit"))
            self.shown_seq = 0
            self.last_snapshot = ""

            for keys, action in (
                (("F", "F11"), self.toggle_fullscreen),
                (("Space",), self.toggle_pause),
                (("S",), self.snapshot),
                (("1",), self.toggle_fit),
                (("M",), self.toggle_smooth),
                (("P",), lambda: self.panel.setVisible(not self.panel.isVisible())),
                (("Q", "Esc"), self.close),
            ):
                for key in keys:
                    QShortcut(QKeySequence(key), self, activated=action)

            self.frame_timer = QTimer(self, interval=5, timeout=self.poll_frame)
            self.frame_timer.start()
            self.info_timer = QTimer(self, interval=250, timeout=self.refresh_info)
            self.info_timer.start()
            self.refresh_info()

        def poll_frame(self) -> None:
            if should_stop():
                self.close()
                return
            latest = viewer.latest_image()
            if latest is None or self.view.paused:
                return
            seq, pixels, _frame = latest
            if seq != self.shown_seq:
                self.shown_seq = seq
                self.view.set_frame(pixels)

        def refresh_info(self) -> None:
            stats = viewer.stats.snapshot()
            stalled = stats.frames == 0 or time.monotonic() - stats.last_arrival_s > STALL_S
            if stats.frames == 0:
                state, color = "WAITING", "#9aa3ad"
            elif stalled:
                state, color = "STALLED", "#f0b429"
            else:
                state, color = "LIVE", "#3fb950"
            self.state.setText(f'<span style="color:{color}">&#9679; {state}</span>')
            if not stalled:
                self.view.alert = ""
            elif viewer.message:
                self.view.alert = viewer.message
            elif stats.frames == 0:
                self.view.alert = "Waiting for frames..."
            else:
                self.view.alert = f"No frames for {time.monotonic() - stats.last_arrival_s:.1f} s"
            if stats.frames:
                kind = "JPEG" if stats.is_jpeg else "raw"
                self.view.chip = f"{stats.fps:5.1f} fps  {stats.width}x{stats.height} {kind}"
            self.view.update()
            self.panel.show_sections(_sections(stats, viewer.deck_config, viewer.deck_status))
            parts = [viewer.status() if viewer.status else "", self.last_snapshot]
            self.left.setText("   ".join(part for part in parts if part))

        def toggle_fullscreen(self) -> None:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

        def toggle_pause(self) -> None:
            self.view.paused = not self.view.paused
            self.view.update()

        def toggle_fit(self) -> None:
            self.view.fit = not self.view.fit
            self.view.update()

        def toggle_smooth(self) -> None:
            self.view.smooth = not self.view.smooth
            self.view.update()

        def snapshot(self) -> None:
            if self.view.image is None:
                return
            viewer.snapshot_dir.mkdir(parents=True, exist_ok=True)
            path = viewer.snapshot_dir / f"snapshot_{time.strftime('%Y%m%dT%H%M%S')}.png"
            self.view.image.save(str(path))
            self.last_snapshot = f"saved {path}"

    return Window()


def _sections(stats: StatsSnapshot, config: dict, status: dict):
    """The side panel's rows: (section, [(label, value, alert)])."""
    sections = [(
        "Stream",
        [
            ("Received", f"{stats.fps:.1f} fps", False),
            ("Bitrate", f"{stats.bitrate_kbps / 1000:.2f} Mbit/s", False),
            ("Frame", f"{stats.width}x{stats.height} {'JPEG' if stats.is_jpeg else 'raw'}", False),
            ("Size", f"{stats.last_size / 1024:.1f} KiB", False),
            ("Frames", f"{stats.frames}", False),
        ],
    )]
    if stats.has_ext:
        sections += [
            ("Deck", [
                ("Captured", f"{stats.camera_fps:.1f} fps", False),
                ("Capture", f"{stats.capture_ms} ms", False),
                ("Resize", f"{stats.preprocess_ms} ms", False),
                ("JPEG", f"{stats.jpeg_ms} ms", False),
                ("Link jitter", f"+{stats.link_jitter_ms:.1f} ms", stats.link_jitter_ms > 100),
            ]),
            ("Dropped on deck", [
                ("Capture", f"{stats.dropped_capture}", False),
                ("Waiting", f"{stats.dropped_ready}", False),
                ("While sending", f"{stats.dropped_streaming}", False),
            ]),
        ]
    if config:
        sections.append(("Build", [
            ("Sensor", str(config.get("sensor", "?")), False),
            ("Output", f"{config.get('out', '?')} {config.get('fit', '')}".strip(), False),
            ("Format", _format_name(config), False),
            ("Policy", _policy_name(config), False),
            ("Target", f"{config.get('fps', '?')} fps", False),
            ("Transport", str(config.get("transport", "?")).upper(), False),
        ]))
    if status:
        sections.append(("Deck status", [
            ("Camera faulted", "yes" if status.get("faulted") else "no", bool(status.get("faulted"))),
            ("Watchdog restarts", f"{status.get('watchdog_restarts', 0)}",
             bool(status.get("watchdog_restarts"))),
            ("Encode failures", f"{status.get('encode_failures', 0)}",
             bool(status.get("encode_failures"))),
            ("TX queue free", f"{status.get('tx_free', '?')}", False),
            ("SPI recoveries", f"{status.get('spi_soft', 0)}/{status.get('spi_hard', 0)}",
             bool(status.get("spi_hard"))),
        ]))
    return sections


def _policy_name(config: dict) -> str:
    if config.get("policy") != "exact":
        return str(config.get("policy", "?"))
    buffers = config.get("buffers")
    return f"exact, {buffers} buffers" if buffers else "exact"


def _format_name(config: dict) -> str:
    if config.get("format") != "jpeg":
        return "raw"
    return f"JPEG {config.get('jpeg', '')} q{config.get('quality', '?')}"


def add_arguments(parser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/drone.yaml"),
                        help="its radio.uri names the deck; its camera: section gives the transport")
    parser.add_argument("--ip", help="the deck's address (default: looked up by radio.uri)")
    parser.add_argument("--port", type=int, help="default: camera.deck_port")
    parser.add_argument("--transport", choices=("tcp", "udp"), help="default: camera.transport")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("."),
                        help="where S saves snapshots")


def run(args) -> int:
    """``cf-lab view``: open the deck's stream in a window."""
    from functools import partial

    from cf_lab.camera.feed import CameraFeed
    from cf_lab.camera.find_deck import resolve_deck
    from cf_lab.config import load_camera_config, radio_uri

    config = args.config if args.config.is_file() else None
    camera = load_camera_config(config, transport=args.transport)
    ip = args.ip
    look_up = None  # an --ip is used as given
    if ip is None:
        if config is None:
            print(f"error: {args.config} not found; pass --ip or --config", file=sys.stderr)
            return 1
        uri = radio_uri(args.config)
        print(f"Looking up the deck of {uri} ...", flush=True)
        try:
            ip = resolve_deck(uri, timeout_s=60.0)
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        look_up = partial(resolve_deck, uri)
    port = args.port or camera.deck_port
    address = f"{ip}:{port} {camera.transport.upper()}"
    feed = CameraFeed(ip=ip, port=port, transport=camera.transport, connect_timeout_s=60.0,
                      look_up=look_up)
    viewer = Viewer(
        title=f"AI-deck camera - {ip}:{port}",
        snapshot_dir=args.snapshot_dir,
        status=lambda: address if feed.error is None else f"{address}: {feed.error}",
    )
    feed.subscribe(viewer)
    viewer.show_message(f"Connecting to {address}...")

    def connect() -> None:
        try:
            feed.start()
        except Exception as exc:
            viewer.show_message(f"Could not connect to {address}: {exc}")
            return
        viewer.show_message("")

    threading.Thread(target=connect, name="viewer-connect", daemon=True).start()
    try:
        viewer.run()
    finally:
        feed.stop()
    return 0
