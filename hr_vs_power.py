from fitfile_parsing import fitfile_to_data
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from palettable.cmocean.sequential import Haline_4_r
from scipy.optimize import least_squares

cmap = plt.get_cmap("rainbow_r")
CMAP_WEEKS = 4

rides = glob("rides/*.fit")
num_rides = len(rides)


fig, ax = plt.subplots()
t0 = datetime(2025, 11, 3).date()
ts = []
mean_hrs = []
mean_powers = []
for i, f in enumerate(reversed(sorted(rides))):
    print(f)
    values, units = fitfile_to_data(f, smoothing_seconds=3.0, seconds_tocut=0)

    distances = values["distance"]
    altitude = values["enhanced_altitude"]
    heartrates = values["heart_rate"]
    power = values["power"]
    timestamps = values["timestamp"]
    dt = timestamps[-1].date() - t0

    minutes_start = 10
    minutes_end = 40
    avg_label = f"({minutes_start}-{minutes_end} min)"
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
        color=cmap(dt.days / 7 / CMAP_WEEKS),
        zorder=-dt.days,
        alpha=1,
    )
    ts.append(dt.days / 7)
    mean_hrs.append(np.mean(heartrates))
    mean_powers.append(np.mean(power))


s = ax.scatter(np.zeros_like(ts), np.zeros_like(ts), c=ts, vmin=0, vmax=CMAP_WEEKS, cmap=cmap)
# plt.colorbar(s, label="Lookback Time (weeks)", pad=0)
plt.colorbar(s, label="Time (weeks)", pad=0, ticks=range(CMAP_WEEKS + 1))
ax.set(xlim=[135, 170], ylim=[100, 250])
plt.tick_params(axis="y", right=False)
plt.ylabel(f"Avg. Power {avg_label} (W)")
plt.xlabel(f"Avg. Heart Rate {avg_label} (bpm)")


mean_hrs = np.array(mean_hrs)
mean_powers = np.array(mean_powers)
ts = np.array(ts)
# cut = ts > 2  # mean_hrs > 140
cut = ts >= 1
# logistic_func = lambda x: 1 / (1 / x + 1)  #
logistic_func = lambda x: 1 / (1 + np.exp(-x))
# logistic_func = lambda x: 0.5 * (1 + np.tanh(x))
model = lambda x, hr, t: x[0] + x[1] * hr + x[2] * logistic_func(np.polyval(x[3:], t))
model_residuals = lambda x: (model(x, mean_hrs, ts) - mean_powers)[cut]
for fit_order in 0, 1:
    fit = least_squares(model_residuals, np.ones(fit_order + 4))
    print(fit.fun.std())
fit = fit.x
print(fit[2], fit[2] * logistic_func(np.polyval(fit[3:], 4)))

for t in range(1, 5):
    hr_plot = np.array([0, 200])
    plt.plot(
        hr_plot,
        model(fit, hr_plot, t),
        color=cmap(t / CMAP_WEEKS),
        ls="dotted",
    )

plt.tight_layout()
plt.savefig("heartrate_vs_power.pdf", bbox_inches="tight")

plt.close()
