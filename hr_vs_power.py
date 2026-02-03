from fitfile_parsing import fitfile_to_data
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from palettable.cmocean.sequential import Haline_4_r
from scipy.optimize import least_squares

cmap = plt.get_cmap("rainbow_r")
CMAP_WEEKS = 16

rides = glob("rides/*.fit")
num_rides = len(rides)


def make_hr_vs_power_plot(time_minutes: float):
    fig, ax = plt.subplots()
    t0 = datetime(2025, 11, 3).date()
    ts = []
    mean_hrs = []
    mean_powers = []
    for i, f in enumerate(reversed(sorted(rides))):
        print(f)
        values, _ = fitfile_to_data(f, smoothing_seconds=3.0, seconds_tocut=0)

        distances = values["distance"]
        altitude = values["enhanced_altitude"]
        heartrates = values["heart_rate"]
        power = values["power"]
        timestamps = values["timestamp"]
        dt = timestamps[-1].date() - t0

        # minutes_start = 10
        # minutes_end = 40
        # avg_label = f"({minutes_start}-{minutes_end} min)"

        # find the most powerful 30min
        dt_window = time_minutes * 60
        psum = power.cumsum()
        if len(power) < dt_window:
            continue
        power_avg = (psum[dt_window:] - psum[:-dt_window]) / dt_window
        hrsum = heartrates.cumsum()
        hr_avg = (hrsum[dt_window:] - hrsum[:-dt_window]) / dt_window
        power_mean = power_avg.max()
        hr_mean = hr_avg[power_avg.argmax()]

        ax.scatter(
            [hr_mean],
            [power_mean],
            lw=0.5,
            marker="s",
            s=20,
            color=cmap(dt.days / 7 / CMAP_WEEKS),
            zorder=-dt.days,
            alpha=1,
        )
        ts.append(dt.days / 7)
        mean_hrs.append(hr_mean)
        mean_powers.append(power_mean)

    s = ax.scatter(np.zeros_like(ts), np.zeros_like(ts), c=ts, vmin=0, vmax=CMAP_WEEKS, cmap=cmap)
    # plt.colorbar(s, label="Lookback Time (weeks)", pad=0)
    plt.colorbar(s, label="Time (weeks)", pad=0, ticks=range(CMAP_WEEKS + 1))
    ax.set(xlim=[130, 170], ylim=[170, 320])
    plt.tick_params(axis="y", right=False)
    plt.ylabel(f"{time_minutes}min Power (W)")
    plt.xlabel(f"{time_minutes}min Heart Rate (bpm)")
    mean_hrs = np.array(mean_hrs)
    mean_powers = np.array(mean_powers)
    cut = mean_powers > 150
    ts = np.array(ts)
    # mean_hrs = mean_hrs[cut]
    # mean_powers = mean_powers[cut]

    # cut = ts > 2  # mean_hrs > 140
    cut = mean_powers > 150
    # logistic_func = lambda x: 1 / (1 / x + 1)  #
    logistic_func = lambda x: 0.5 * (1 + np.tanh(x / 2))
    model = lambda x, hr, t: x[0] + x[1] * hr + x[2] * logistic_func(np.polyval(x[3:], t))
    model_residuals = lambda x: (model(x, mean_hrs, ts) - mean_powers)[cut]
    for fit_order in (0, 1):
        fit = least_squares(model_residuals, np.ones(fit_order + 4))
        print(fit.fun.std())
    fit = fit.x
    print(fit[2], fit[2] * logistic_func(np.polyval(fit[3:], 4)))

    # for t in range(1, 7):
    #     hr_plot = np.array([0, 200])
    #     plt.plot(
    #         hr_plot,
    #         model(fit, hr_plot, t),
    #         color=cmap(t / CMAP_WEEKS),
    #         ls="dotted",
    #     )

    plt.tight_layout()
    plt.savefig(f"heartrate_vs_power_{time_minutes}.pdf", bbox_inches="tight")

    plt.close()


def main():
    for m in 5, 20, 30, 60:
        make_hr_vs_power_plot(m)


if __name__ == "__main__":
    main()
