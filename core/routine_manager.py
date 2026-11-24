import json
import os
from datetime import datetime, time as dtime
from typing import Dict, List, Optional

from core.routine import Routine, SCHEDULE_TIME, SCHEDULE_INTERVAL
from core.cart import Cart


class RoutineManager:
    def __init__(self, filepath: Optional[str]):
        """filepath=None keeps routines in memory only (not saved)."""
        self.filepath = filepath
        self.routines: Dict[str, Routine] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.filepath or not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath) as f:
                data = json.load(f)
            for d in data:
                r = Routine.from_dict(d)
                self.routines[r.id] = r
        except (json.JSONDecodeError, KeyError):
            pass

    def save(self) -> None:
        if not self.filepath:
            return
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump([r.to_dict() for r in self.routines.values()], f, indent=2)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, routine: Routine) -> None:
        self.routines[routine.id] = routine
        self.save()

    def remove(self, routine_id: str) -> None:
        self.routines.pop(routine_id, None)
        self.save()

    def update(self, routine: Routine) -> None:
        self.routines[routine.id] = routine
        self.save()

    def get(self, routine_id: str) -> Optional[Routine]:
        return self.routines.get(routine_id)

    def for_cart(self, cart_id: str) -> List[Routine]:
        return [r for r in self.routines.values() if r.cart_id == cart_id]

    def all(self) -> List[Routine]:
        return list(self.routines.values())

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def check_due(self, now: datetime, busy_cart_ids: set) -> List[Routine]:
        """Return routines that should fire right now."""
        due: List[Routine] = []
        for r in self.routines.values():
            if not r.enabled or not r.waypoints:
                continue
            if r.cart_id in busy_cart_ids:
                continue
            if r.schedule_type == SCHEDULE_TIME:
                if r.time_of_day:
                    try:
                        target = dtime.fromisoformat(r.time_of_day)
                    except ValueError:
                        continue
                    stamp = now.strftime("%Y-%m-%d %H:%M")
                    if (now.hour == target.hour and now.minute == target.minute
                            and now.second < 2 and r.last_triggered != stamp):
                        due.append(r)
            elif r.schedule_type == SCHEDULE_INTERVAL:
                if r.interval_seconds and r.interval_seconds > 0:
                    if r.last_triggered is None:
                        due.append(r)
                    else:
                        try:
                            last = datetime.fromisoformat(r.last_triggered)
                            if (now - last).total_seconds() >= r.interval_seconds:
                                due.append(r)
                        except ValueError:
                            due.append(r)
        return due

    def mark_triggered(self, routine: Routine, now: datetime) -> None:
        if routine.schedule_type == SCHEDULE_TIME:
            routine.last_triggered = now.strftime("%Y-%m-%d %H:%M")
        else:
            routine.last_triggered = now.isoformat()
        self.save()
