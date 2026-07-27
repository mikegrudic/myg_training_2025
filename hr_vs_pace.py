from fitfile_parsing import fitfile_to_data, smooth
from garmin_sync import sync_runs
from glob import glob
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from astropy import units as u
from datetime import datetime


cmap = plt.get_cmap("rainbow_r")
CMAP_MONTHS = 18
USE_GAP = False  # set True to use grade-adjusted pace, False for plain pace


def grade_adjustment(grade_percent):
    grade, fac = np.loadtxt("strava_GAP_table.dat").T
    fac = np.interp(grade_percent, grade, fac)
    fac[np.isnan(fac)] = 1.0
    return fac


def make_hr_vs_pace_plot(dist_km: int):
    runs = sorted(
        glob("runs_block1/*.fit") + glob("runs_block2/*.fit") + glob("runs/*.fit"),
        key=os.path.basename,
    )
    fig, ax = plt.subplots()
    today = datetime.today().date()

    # First pass: collect everything that will plot, tagged by date.
    entries = []
    for i, f in enumerate(reversed(runs)):
        values, _ = fitfile_to_data(f, smoothing_seconds=60.0, seconds_tocut=0)

        distances = values["distance"]
        altitude = values["enhanced_altitude"]
        if distances.max() < 1e3 * dist_km * 0.97:  # account for garmin tax
            continue
        heartrates = values["heart_rate"]
        timestamps = values["timestamp"]

        seconds_to_cut = 300
        if np.any(np.diff(distances) < 0):
            continue

        SPEED_SMOOTHING_SECONDS = 10
        speed = np.gradient(smooth(distances, SPEED_SMOOTHING_SECONDS))
        vertical_speed = np.gradient(smooth(altitude, SPEED_SMOOTHING_SECONDS))
        grade_percent = vertical_speed / speed * 100
        gap_factor = grade_adjustment(grade_percent)
        if not len(speed):
            continue

        pace = 1 / (speed * u.m.to(u.imperial.mile) * 60)
        if USE_GAP:
            pace = pace / gap_factor
        if np.any(speed == 0):
            pace[speed == 0] = np.inf

        pace = pace[distances < 1e3 * dist_km]
        heartrates = heartrates[distances < 1e3 * dist_km]
        sigma_pace = np.diff(np.percentile(pace[seconds_to_cut:], [16, 50, 84]))[:, None]
        sigma_hr = np.diff(np.percentile(heartrates[seconds_to_cut:], [16, 50, 84]))[:, None]

        entries.append({
            "date": timestamps[-1].date(),
            "is_today": timestamps[-1].date() == today and i == 0,
            "hr": float(np.median(heartrates[seconds_to_cut:])),
            "pace": float(np.median(pace[seconds_to_cut:])),
            "sigma_hr": sigma_hr,
            "sigma_pace": sigma_pace,
        })

    if not entries:
        plt.close()
        return

    t0 = min(e["date"] for e in entries)
    today_found = False
    ts = []

    for e in entries:
        dt_days = (e["date"] - t0).days
        if e["is_today"]:
            marker, markersize, edgec = "*", 12, "black"
            today_found = True
        else:
            marker, markersize, edgec = "o", 4, None
        ax.errorbar(
            e["hr"], e["pace"],
            yerr=e["sigma_pace"], xerr=e["sigma_hr"],
            lw=0.5,
            marker=marker, markersize=markersize, markeredgecolor=edgec,
            color=cmap(dt_days / 30 / CMAP_MONTHS),
            zorder=dt_days,
            alpha=1, capsize=1,
        )
        ts.append(dt_days / 30)

    s = ax.scatter(np.zeros_like(ts), np.zeros_like(ts), c=ts, vmin=0, vmax=CMAP_MONTHS, cmap=cmap)
    plt.colorbar(s, label="Time (months)", pad=0, ticks=range(CMAP_MONTHS + 1))
    ax.set_title(f"{dist_km}k Runs")
    ax.set(xlim=[135, 180], ylim=[8, 11])
    plt.tick_params(axis="y", right=False)
    plt.ylabel(f"{'Grade-adjusted pace' if USE_GAP else 'Pace'} (min/mi)")
    plt.xlabel("Heart Rate (bpm)")

    if today_found:
        ax.legend(
            handles=[
                Line2D(
                    [0], [0],
                    marker="*", linestyle="", markersize=12,
                    markerfacecolor="lightgray", markeredgecolor="black", label="Today",
                )
            ],
            loc="upper left",
        )

    plt.tight_layout()
    plt.savefig(f"heartrate_vs_speed_{dist_km}.pdf", bbox_inches="tight")
    plt.close()


def main():
    try:
        sync_runs()
    except Exception as e:
        print(f"Skipping Garmin sync: {e}")
    for dist in (5, 10, 21):
        make_hr_vs_pace_plot(dist)


if __name__ == "__main__":
    main()
