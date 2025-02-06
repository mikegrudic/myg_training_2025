from fitfile_parsing import fitfile_to_data, smooth
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u

runs = glob("runs/*.fit")
num_runs = len(runs)

for i, f in enumerate(sorted(runs)):
    values, units = fitfile_to_data(f, smoothing_seconds=60.0, seconds_tocut=0)

    distances = values["distance"]
    heartrates = values["heart_rate"]
    timestamps = values["timestamp"]
    seconds_to_cut = 300
    if np.any(np.diff(distances) < 0):
        continue
    speed = np.gradient(smooth(distances, 60.0))[seconds_to_cut:]
    if not len(speed):
        continue

    pace = 1 / (speed * u.m.to(u.imperial.mile) * 60)
    if np.any(speed == 0):
        pace[speed == 0] = np.inf

    sigma_pace = 0.5 * np.diff(np.percentile(pace, [16, 84]))[0]
    if sigma_pace > 4:
        continue
    sigma_hr = 0.5 * np.diff(np.percentile(heartrates[seconds_to_cut:], [16, 84]))[0]

    plt.errorbar(
        np.median(heartrates[seconds_to_cut:]),
        np.median(pace[seconds_to_cut:]),
        yerr=sigma_pace,
        xerr=sigma_hr,
        label=timestamps[-1].date(),
        lw=1,
        marker="s",
    )
plt.ylabel("Pace (min/mi)")
plt.xlabel("Heart Rate (bpm)")
# plt.xlim(120,180)
plt.legend(labelspacing=0, frameon=True, fontsize=8)
plt.savefig("heartrate_vs_speed.pdf", bbox_inches="tight")
