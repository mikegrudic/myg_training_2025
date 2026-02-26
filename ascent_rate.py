from fitfile_parsing import fitfile_to_data, smooth
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u

runs = glob("hikes/*.fit")
num_runs = len(runs)

for i, f in enumerate(sorted(runs)):
    values, units = fitfile_to_data(f, smoothing_seconds=0.0, seconds_tocut=0)
    distances = values["distance"]
    elevation = values["enhanced_altitude"]
    heartrates = values["heart_rate"]
    timestamps = values["timestamp"]
    seconds = np.array([(t - timestamps[0]).seconds for t in timestamps])
    hours = seconds / 3600
    if not np.any(hours > 1):
        continue
    for tau in (180.0,):
        climb_rate = smooth(np.gradient(elevation, hours), tau) * u.m.to(u.imperial.ft)
        if np.any(np.diff(distances) < 0):
            continue
        speed = smooth(np.gradient(distances, seconds), tau)
        if not len(speed):
            continue
        pace = 1 / (speed * u.m.to(u.imperial.mile) * 60)
        if np.any(speed == 0):
            pace[speed == 0] = np.inf

        if timestamps[-1].year == 2024:
            color = "black"
        else:
            color = "blue"
        plt.scatter(
            smooth(heartrates, tau / 2)[::6], climb_rate[::6], label=timestamps[0].date(), lw=1, color=color, s=0.4
        )
        # plt.scatter(
        #     (climb_rate / (speed * u.m.to(u.imperial.ft) * 3600))[::60] * 100,
        #     pace[::60],
        #     label=timestamps[0].date(),
        #     lw=1,
        #     color=color,
        #     s=0.4,
        # )
        # plt.xlim(-50, 50)
        # plt.ylim(0, 60)
    # plt.plot(hours, climb_rate, label=timestamps[0].date(), lw=1, color=color)  # , s=0.4)
    plt.xlim(100, 180)
    plt.ylim(0, 2500.0)
    #    plt.legend()
    # plt.xlabel(r"Grade ($\%$)")
    # plt.xlabel("Ascent Rate (ft/h)")
    plt.ylabel("Pace (min/mi)")
    plt.ylabel("Ascent Rate (ft/h)")
    plt.xlabel("Heart Rate (bpm)")
# plt.xlabel("Elapsed Time (h)")

plt.savefig("climb_rate.pdf", bbox_inches="tight")
