import uuid
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


SCHEDULE_TIME = "time_of_day"
SCHEDULE_INTERVAL = "interval"


@dataclass
class Routine:
    id: str
    cart_id: str
    name: str
    schedule_type: str          # SCHEDULE_TIME or SCHEDULE_INTERVAL
    waypoints: List[Tuple[int, int]] = field(default_factory=list)
    time_of_day: Optional[str] = None       # "HH:MM"
    interval_seconds: Optional[int] = None
    repeat: bool = False
    enabled: bool = True
    last_triggered: Optional[str] = None    # ISO datetime string

    @staticmethod
    def make(cart_id: str, name: str, schedule_type: str) -> "Routine":
        return Routine(id=str(uuid.uuid4())[:12], cart_id=cart_id,
                       name=name, schedule_type=schedule_type)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "cart_id": self.cart_id,
            "name": self.name,
            "schedule_type": self.schedule_type,
            "waypoints": [list(wp) for wp in self.waypoints],
            "time_of_day": self.time_of_day,
            "interval_seconds": self.interval_seconds,
            "repeat": self.repeat,
            "enabled": self.enabled,
            "last_triggered": self.last_triggered,
        }

    @staticmethod
    def from_dict(d: dict) -> "Routine":
        return Routine(
            id=d["id"],
            cart_id=d["cart_id"],
            name=d["name"],
            schedule_type=d["schedule_type"],
            waypoints=[tuple(wp) for wp in d.get("waypoints", [])],
            time_of_day=d.get("time_of_day"),
            interval_seconds=d.get("interval_seconds"),
            repeat=d.get("repeat", False),
            enabled=d.get("enabled", True),
            last_triggered=d.get("last_triggered"),
        )
