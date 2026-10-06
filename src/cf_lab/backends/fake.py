from __future__ import annotations

from dataclasses import dataclass, field

from cf_lab.mocap.types import PoseSample


@dataclass
class FakeCrazyflieBackend:
    """In-memory backend that records safety-critical calls for dry runs."""

    armed: bool = False
    emergency_stopped: bool = False
    extpose_count: int = 0
    commands: list[str] = field(default_factory=list)

    def arm(self) -> None:
        self.armed = True
        self.commands.append("arm")

    def disarm(self) -> None:
        self.armed = False
        self.commands.append("disarm")

    def emergency_stop(self) -> None:
        self.armed = False
        self.emergency_stopped = True
        self.commands.append("emergency_stop")

    def send_extpose(self, pose: PoseSample) -> None:
        self.extpose_count += 1
        self.commands.append(f"extpose:{pose.rigid_body}")
