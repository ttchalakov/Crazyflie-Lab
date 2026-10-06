"""Finite GAP8 flash matrix; sources make every transition reproducible."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Fixture:
    name: str
    identity: int
    payload_bytes: int = 4096
    interval_ms: int = 1000
    silent: bool = False


FIXTURES = {
    fixture.name: fixture for fixture in (
        Fixture("a", 1), Fixture("b", 2),
        Fixture("busy", 3, interval_ms=50),
        Fixture("silent", 4, payload_bytes=0, silent=True),
        Fixture("large", 5, payload_bytes=1024 * 1024),
    )
}


@dataclass(frozen=True)
class Case:
    name: str
    source: str
    target: str
    cold_boot: bool = False
    attempts: int | None = None  # None preserves the production default.
    repair: bool = False
    power_cycle_after: bool = False


CASES = {case.name: case for case in (
    Case("warm-a-a", "a", "a"),
    Case("warm-a-b", "a", "b"),
    Case("warm-b-b", "b", "b"),
    Case("warm-b-a", "b", "a"),
    Case("single-attempt-a-b", "a", "b", attempts=1),
    Case("single-attempt-b-a", "b", "a", attempts=1),
    Case("a-to-silent", "a", "silent"),
    Case("silent-to-a", "silent", "a"),
    Case("a-to-busy", "a", "busy"),
    Case("busy-to-b", "busy", "b"),
    Case("small-to-large", "b", "large"),
    Case("large-to-small", "large", "a"),
    Case("repair-empty-area", "a", "a", repair=True),
    Case("cold-a-b", "a", "b", cold_boot=True),
    Case("cold-b-a", "b", "a", cold_boot=True),
    Case("persist-small", "a", "b", power_cycle_after=True),
    Case("persist-large", "b", "large", power_cycle_after=True),
)}
SETS = {
    "common": ["warm-a-b", "warm-b-a", "a-to-silent", "silent-to-a",
               "repair-empty-area"],
    "exhaustive": list(CASES),
}


def select(set_name: str, names: list[str] | None) -> list[Case]:
    if names is None:
        if set_name not in SETS:
            raise ValueError(f"unknown set {set_name!r}; choose {', '.join(SETS)}")
        names = SETS[set_name]
    unknown = set(names) - CASES.keys()
    if unknown:
        raise ValueError(f"unknown cases: {', '.join(sorted(unknown))}")
    return [CASES[name] for name in names]
