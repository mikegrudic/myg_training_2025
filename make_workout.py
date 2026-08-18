"""Build a Garmin Connect running workout JSON and optionally upload it.

Usage:
    python make_workout.py                      # write workout.json
    python make_workout.py --upload             # upload to Garmin Connect
    python make_workout.py --upload --date 2026-08-20   # upload and schedule
"""

import argparse
import json

METERS_PER_MILE = 1609.344

SPORT_RUNNING = {"sportTypeId": 1, "sportTypeKey": "running", "displayOrder": 1}

STEP_TYPES = {
    "warmup": {"stepTypeId": 1, "stepTypeKey": "warmup", "displayOrder": 1},
    "cooldown": {"stepTypeId": 2, "stepTypeKey": "cooldown", "displayOrder": 2},
    "interval": {"stepTypeId": 3, "stepTypeKey": "interval", "displayOrder": 3},
    "recovery": {"stepTypeId": 4, "stepTypeKey": "recovery", "displayOrder": 4},
    "repeat": {"stepTypeId": 6, "stepTypeKey": "repeat", "displayOrder": 6},
}

END_DISTANCE = {
    "conditionTypeId": 1,
    "conditionTypeKey": "distance",
    "displayOrder": 1,
    "displayable": True,
}
END_TIME = {
    "conditionTypeId": 2,
    "conditionTypeKey": "time",
    "displayOrder": 2,
    "displayable": True,
}
END_ITERATIONS = {
    "conditionTypeId": 7,
    "conditionTypeKey": "iterations",
    "displayOrder": 7,
    "displayable": False,
}

TARGET_NONE = {
    "workoutTargetTypeId": 1,
    "workoutTargetTypeKey": "no.target",
    "displayOrder": 1,
}
TARGET_PACE = {
    "workoutTargetTypeId": 6,
    "workoutTargetTypeKey": "pace.zone",
    "displayOrder": 6,
}

EASY = (630, 570)  # 10:30-9:30 min/mi


def pace_to_speed(pace_seconds_per_mile: float) -> float:
    return METERS_PER_MILE / pace_seconds_per_mile


def speed_to_pace(speed_m_per_s: float) -> str:
    minutes, seconds = divmod(round(METERS_PER_MILE / speed_m_per_s), 60)
    return f"{minutes}:{seconds:02d}"


def pace_band(center: str, tol_seconds: float = 15.0) -> tuple[float, float]:
    """'8:30' +/- tol -> (slow_pace_secs, fast_pace_secs)."""
    minutes, seconds = center.split(":")
    center_secs = int(minutes) * 60 + int(seconds)
    return center_secs + tol_seconds, center_secs - tol_seconds


class Builder:
    """Emits steps with the global, monotonic stepOrder that Connect expects."""

    def __init__(self):
        self.order = 0
        self.child_id = 0

    def _step(self, kind, end_condition, end_value, pace, child):
        self.order += 1
        s = {
            "type": "ExecutableStepDTO",
            "stepOrder": self.order,
            "stepType": STEP_TYPES[kind],
            "endCondition": end_condition,
            "endConditionValue": float(end_value),
            "targetType": TARGET_NONE,
        }
        if child is not None:
            s["childStepId"] = child
        if pace is not None:
            slow, fast = pace
            # pace.zone targets are m/s; one = slower bound, two = faster bound
            s["targetType"] = TARGET_PACE
            s["targetValueOne"] = round(pace_to_speed(slow), 4)
            s["targetValueTwo"] = round(pace_to_speed(fast), 4)
        return s

    def miles(self, kind, n, pace=None, child=None):
        return self._step(kind, END_DISTANCE, n * METERS_PER_MILE, pace, child)

    def seconds(self, kind, n, pace=None, child=None):
        return self._step(kind, END_TIME, n, pace, child)

    def repeat(self, iterations, make_steps):
        self.order += 1
        group_order = self.order
        self.child_id += 1
        child = self.child_id
        return {
            "type": "RepeatGroupDTO",
            "stepOrder": group_order,
            "stepType": STEP_TYPES["repeat"],
            "numberOfIterations": iterations,
            "endCondition": END_ITERATIONS,
            "endConditionValue": float(iterations),
            "smartRepeat": False,
            "childStepId": child,
            "workoutSteps": make_steps(child),
        }


def build():
    b = Builder()
    steps = [b.miles("warmup", 2.0, EASY)]

    # 5 x (90s / 60s / 30s) descending, 1 min easy within, 3 min between sets
    def interval_set(child):
        out = []
        for dur, center in [(90, "8:30"), (60, "8:00"), (30, "7:15")]:
            out.append(b.seconds("interval", dur, pace_band(center), child))
            out.append(b.seconds("recovery", 60, EASY, child))
        # last within-set recovery is the 3 min between-set jog
        out[-1]["endConditionValue"] = 180.0
        return out

    steps.append(b.repeat(5, interval_set))
    steps.append(b.miles("interval", 2.0, pace_band("9:15")))
    steps.append(b.miles("cooldown", 1.0, EASY))

    duration = round(
        2 * 600  # 2 mi warmup at 10:00
        + 5 * (90 + 60 + 60 + 60 + 30 + 180)
        + 2 * 555  # 2 mi at 9:15
        + 600  # 1 mi cooldown at 10:00
    )

    return {
        "workoutName": "Descending 90/60/30 + 2mi tempo",
        "description": "2mi easy / 5x(90s 8:30, 60s 8:00, 30s 7:15) / 2mi 9:15 / 1mi easy",
        "sportType": SPORT_RUNNING,
        "estimatedDurationInSecs": duration,
        "author": {},
        "workoutSegments": [
            {"segmentOrder": 1, "sportType": SPORT_RUNNING, "workoutSteps": steps}
        ],
    }


def flatten(steps):
    """Expand repeat groups into the steps actually executed."""
    for s in steps:
        if s.get("type") == "RepeatGroupDTO":
            for _ in range(s["numberOfIterations"]):
                yield from flatten(s["workoutSteps"])
        else:
            yield s


def totals(steps, bound="center"):
    """Distance (mi) and time (s) with every band run at its slow/center/fast edge.

    Distance-based steps fix the mileage and vary the time; time-based steps do
    the reverse, so the two totals move in opposite directions.
    """
    dist = time = 0.0
    for s in flatten(steps):
        if s.get("targetValueOne"):
            slow, fast = sorted((s["targetValueOne"], s["targetValueTwo"]))
            speed = {"slow": slow, "fast": fast, "center": (slow + fast) / 2}[bound]
            pace = METERS_PER_MILE / speed
        else:
            pace = 600.0  # untargeted steps: assume 10:00/mi
        if s["endCondition"]["conditionTypeKey"] == "distance":
            d = s["endConditionValue"] / METERS_PER_MILE
            dist, time = dist + d, time + d * pace
        else:
            time += s["endConditionValue"]
            dist += s["endConditionValue"] / pace
    return dist, time


def show(steps, depth=0):
    """Print steps as Garmin stored them, converting pace targets to min/mi."""
    for s in steps:
        pad = "  " * depth
        if s.get("type") == "RepeatGroupDTO":
            print(f"{pad}repeat x{s['numberOfIterations']}")
            show(s["workoutSteps"], depth + 1)
            continue
        end = s["endCondition"]["conditionTypeKey"]
        value = s["endConditionValue"]
        amount = f"{value / METERS_PER_MILE:.2f} mi" if end == "distance" else f"{value:.0f}s"
        target = s["targetType"]["workoutTargetTypeKey"]
        if s.get("targetValueOne"):
            lo, hi = sorted((s["targetValueOne"], s["targetValueTwo"]))
            target += f" {speed_to_pace(lo)}-{speed_to_pace(hi)}/mi"
        print(f"{pad}{s['stepType']['stepTypeKey']:9} {amount:9} [{target}]")

    if depth == 0:
        for bound in ("slow", "center", "fast"):
            dist, time = totals(steps, bound)
            minutes, seconds = divmod(round(time), 60)
            pace = time / dist
            print(
                f"{bound:6} {dist:5.2f} mi  {minutes}:{seconds:02d}  "
                f"avg {int(pace // 60)}:{round(pace % 60):02d}/mi"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="workout.json")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--date", help="schedule on YYYY-MM-DD after upload")
    p.add_argument("--fit", help="write the FIT Garmin generates to this path")
    args = p.parse_args()

    workout = build()
    with open(args.out, "w") as f:
        json.dump(workout, f, indent=2)
    print(f"wrote {args.out} ({workout['estimatedDurationInSecs'] / 60:.0f} min)")
    show(workout["workoutSegments"][0]["workoutSteps"])

    if not args.upload:
        return

    from garmin_sync import login

    client = login()
    resp = client.upload_workout(workout)
    workout_id = resp["workoutId"]
    print(f"uploaded workoutId={workout_id}")

    print("as stored by Garmin:")
    saved = client.get_workout_by_id(workout_id)
    show(saved["workoutSegments"][0]["workoutSteps"], depth=1)

    if args.date:
        client.schedule_workout(workout_id, args.date)
        print(f"scheduled for {args.date}")

    if args.fit:
        with open(args.fit, "wb") as f:
            f.write(client.download_workout(workout_id))
        print(f"wrote {args.fit}")


if __name__ == "__main__":
    main()
