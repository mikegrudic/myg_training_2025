from fitfile_parsing import fitfile_to_data, smooth
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from datetime import datetime, timedelta
from palettable.cmocean.sequential import Haline_16_r

cmap = plt.get_cmap("rainbow_r")
CMAP_WEEKS = 16

race_day = datetime(2025, 9, 6)

runs = glob("runs_block2/*.fit")
num_runs = len(runs)


def grade_adjustment(grade_percent):
    grade, fac = np.loadtxt("strava_GAP_table.dat").T
    return np.interp(grade_percent, grade, fac)


for dist in 5, 10, 21:
    fig, ax = plt.subplots()
    # ax.set_prop_cycle("color", Dark2_7.mpl_colors)
    t0 = datetime.today().date()
    ts = []
    for i, f in enumerate(reversed(sorted(runs))):
        values, units = fitfile_to_data(f, smoothing_seconds=60.0, seconds_tocut=0)

        distances = values["distance"]
        altitude = values["enhanced_altitude"]
        if distances.max() < 1e3 * dist * 0.97:
            continue
        heartrates = values["heart_rate"]
        timestamps = values["timestamp"]
        if timestamps[-1].date() < t0:
            t0 = timestamps[-1].date()
        dt = race_day.date() - datetime.today().date()  # - timestamps[-1].date()
        dt_race = datetime(2025, 5, 3).date() - timestamps[-1].date()

        seconds_to_cut = 300
        if np.any(np.diff(distances) < 0):
            continue

        SPEED_SMOOTHING_SECONDS = 10
        speed = np.gradient(smooth(distances, SPEED_SMOOTHING_SECONDS))
        vertical_speed = np.gradient(smooth(altitude, SPEED_SMOOTHING_SECONDS))
        grade_percent = vertical_speed / speed * 100
        gap_factor = grade_adjustment(grade_percent)

        # print(f)
        # plt.plot(grade_percent)
        # plt.show()
        # continue
        # # exit()
        if not len(speed):
            continue

        pace = 1 / (speed * u.m.to(u.imperial.mile) * 60) / gap_factor
        if np.any(speed == 0):
            pace[speed == 0] = np.inf

        pace = pace[distances < 1e3 * dist]
        heartrates = heartrates[distances < 1e3 * dist]
        sigma_pace = np.diff(np.percentile(pace[seconds_to_cut:], [16, 50, 84]))[:, None]
        sigma_hr = np.diff(np.percentile(heartrates[seconds_to_cut:], [16, 50, 84]))[:, None]

        ax.errorbar(
            np.median(heartrates[seconds_to_cut:]),
            np.median(pace[seconds_to_cut:]),
            yerr=sigma_pace,
            xerr=sigma_hr,
            lw=0.8,
            marker="o",
            markersize=2,
            color=cmap(dt.days / 7 / CMAP_WEEKS),  # cmap(dt_race.days / 7 / cmap_weeks)
            zorder=-dt.days,  # -dt_race.days,  #
            alpha=1,  # (0.5 if dt.days > 7 * CMAP_WEEKS else 1.0),
            capsize=1,
        )
    #    ts.append(dt.days / 7)

    s = ax.scatter(np.zeros_like(ts), np.zeros_like(ts), c=ts, vmin=0, vmax=CMAP_WEEKS, cmap=cmap)
    plt.colorbar(s, label="Weeks til Race Day", pad=0)
    # plt.colorbar(s, label="Weeks Until Race Day", pad=0)
    ax.set_title(f"{dist}k Runs")
    ax.set(xlim=[145, 170], ylim=[7, 12])
    plt.ylabel("Grade-adjusted pace (min/mi)")
    plt.xlabel("Heart Rate (bpm)")
    # plt.xlim(120,180)
    # plt.legend(labelspacing=0, frameon=True, fontsize=8)
    plt.savefig(f"heartrate_vs_speed_{dist}.pdf", bbox_inches="tight")
