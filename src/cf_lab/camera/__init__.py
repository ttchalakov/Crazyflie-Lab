"""The host side of the AI-deck image stream.

``CameraFeed`` is the single owner of the deck's one connection; a viewer and a
telemetry recorder both attach to it as sinks rather than connecting themselves.
See ``cf_lab.camera.feed`` for why that is forced rather than merely tidy.

Importing this package pulls in no OpenCV: recording frames does not need it,
and only the viewer and the video renderer do.
"""

from cf_lab.camera.feed import CameraFeed, CapturedFrame, FrameSink

__all__ = [
    "CameraFeed",
    "CapturedFrame",
    "FrameSink",
]
