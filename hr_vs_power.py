from fitfile_parsing import fitfile_to_data, smooth
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from datetime import datetime, timedelta
from palettable.cmocean.sequential import Haline_16_r

cmap = plt.get_cmap("rainbow_r")
CMAP_WEEKS = 4

rides = glob("rides/*.fit")
num_rides = len(rides)


# def grade_adjustment(grade_percent):
#     grade, fac = np.loadtxt("strava_GAP_table.dat").T
#     #    print(fac.min())
#     fac = np.interp(grade_percent, grade, fac)
#     fac[np.isnan(fac)] = 1.0
#     return fac


# for dist in 5, 10, 21:
fig, ax = plt.subplots()
# ax.set_prop_cycle("color", Dark2_7.mpl_colors)
t0 = datetime.today().date()
ts = []
for i, f in enumerate(reversed(sorted(rides))):
    print(f)
    values, units = fitfile_to_data(f, smoothing_seconds=3.0, seconds_tocut=0)

    distances = values["distance"]
    altitude = values["enhanced_altitude"]
    heartrates = values["heart_rate"]
    power = values["power"]
    timestamps = values["timestamp"]
    dt = t0 - timestamps[-1].date()

    minutes_start = 10
    minutes_end = 40
    avg_label = f"(Average {minutes_start}-{minutes_end} min)"
    power, heartrates = power[minutes_start * 60 : minutes_end * 60], heartrates[minutes_start * 60 : minutes_end * 60]
    if len(power) == 0:
        continue
    sigma_power = np.diff(np.percentile(power, [16, 50, 84]))[:, None]
    sigma_hr = np.diff(np.percentile(heartrates, [16, 50, 84]))[:, None]
    # ax.errorbar(
    #     np.median(heartrates[seconds_to_cut:]),
    #     np.median(power[seconds_to_cut:]),
    #     yerr=sigma_power,
    #     xerr=sigma_hr,
    #     lw=0.5,
    #     marker="o",
    #     markersize=2,
    #     color=cmap(dt.days / 7 / CMAP_WEEKS),  # cmap(dt_race.days / 7 / cmap_weeks)
    #     zorder=-dt.days,  # -dt_race.days,  #
    #     alpha=1,  # (0.5 if dt.days > 7 * CMAP_WEEKS else 1.0),
    #     capsize=1,
    # )
    ax.scatter(
        [np.mean(heartrates)],
        [np.mean(power)],
        lw=0.5,
        marker="s",
        s=20,
        color=cmap(dt.days / 7 / CMAP_WEEKS),  # cmap(dt_race.days / 7 / cmap_weeks)
        zorder=-dt.days,  # -dt_race.days,  #
        alpha=1,  # (0.5 if dt.days > 7 * CMAP_WEEKS else 1.0),
    )
    ts.append(dt.days / 7)
print(ts)
s = ax.scatter(np.zeros_like(ts), np.zeros_like(ts), c=ts, vmin=0, vmax=CMAP_WEEKS, cmap=cmap)
plt.colorbar(s, label="Lookback Time (weeks)", pad=0)
ax.set(xlim=[145, 175], ylim=[150, 250])
plt.ylabel(f"Power {avg_label} (W)")
plt.xlabel(f"Heart Rate {avg_label} (bpm)")
plt.tight_layout()
plt.savefig(f"heartrate_vs_power.pdf", bbox_inches="tight")
