from __future__ import annotations


def parse_ego_speed(value: str) -> float | None:
    """Parse an explicit speed or the per-frame automatic mode."""
    if value.strip().lower() == "auto":
        return None
    return float(value)
