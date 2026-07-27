from fitfile_parsing import fitfile_to_data
from garmin_sync import sync_rides
from glob import glob
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime
from palettable.cmocean.sequential import Haline_4_r
from scipy.optimize import least_squares

cmap = plt.get_cmap("rainbow_r")
CMAP_MONTHS = 12


def make_hr_vs_power_plot(time_minutes: float, most_efficient=False):
    rides = sorted(glob("rides/*.fit") + glob("rides/heat/*.fit"), key=os.path.basename)
    fig, ax = plt.subplots()
    t0 = datetime(2025, 11, 3).date()
    ts = []
    mean_hrs = []
    mean_powers = []
    today_found = False
    heat_found = False
    cadence_found = False
    for i, f in enumerate(reversed(rides)):
        is_heat = "/heat/" in f
        values, _ = fitfile_to_data(f, smoothing_seconds=3.0, seconds_tocut=300)

        distances = values["distance"]
        altitude = values["enhanced_altitude"]
        heartrates = values["heart_rate"]
        power = values["power"]
        cadence = values.get("cadence")
        timestamps = values["timestamp"]
        dt = timestamps[-1].date() - t0
        ride_is_today = timestamps[-1].date() == datetime.today().date() and i == 0

        # find the most powerful 30min
        dt_window = time_minutes * 60
        psum = power.cumsum()
        if len(power) < dt_window:
            continue
        power_avg = (psum[dt_window:] - psum[:-dt_window]) / dt_window
        hrsum = heartrates.cumsum()
        hr_avg = (hrsum[dt_window:] - hrsum[:-dt_window]) / dt_window
        efficiency_avg = power_avg / hr_avg
        if most_efficient:
            sel_idx = int(efficiency_avg.argmax())
        else:
            sel_idx = int(power_avg.argmax())
        power_mean = power_avg[sel_idx]
        hr_mean = hr_avg[sel_idx]

        high_cadence = False
        if cadence is not None and len(cadence) >= sel_idx + dt_window:
            cad_window = np.asarray(cadence[sel_idx : sel_idx + dt_window], dtype=float)
            valid = np.isfinite(cad_window)
            if np.any(valid):
                high_cadence = float(np.mean(cad_window[valid])) >= 80.0

        if ride_is_today:
            marker = "*"
            size = 60
            edgec = "black"
            today_found = True
        elif is_heat:
            marker = "v"
            size = 30
            edgec = "red"
            heat_found = True
        elif high_cadence:
            marker = "D"
            size = 25
            edgec = "blue"
            cadence_found = True
        else:
            marker = "s"
            size = 20
            edgec = None

        ax.scatter(
            [hr_mean],
            [power_mean],
            lw=0.5,
            marker=marker,
            s=size,
            edgecolor=edgec,
            color=cmap(dt.days / 30 / CMAP_MONTHS),
            zorder=dt.days,
            alpha=1,
        )
        ts.append(dt.days / 7 / 30)
        mean_hrs.append(hr_mean)
        mean_powers.append(power_mean)

    s = ax.scatter(np.zeros_like(ts), np.zeros_like(ts), c=ts, vmin=0, vmax=CMAP_MONTHS, cmap=cmap)
    # plt.colorbar(s, label="Lookback Time (weeks)", pad=0)
    plt.colorbar(s, label="Time (months)", pad=0, ticks=range(CMAP_MONTHS + 1))
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
    #         color=cmap(t / CMAP_MONTHS),
    #         ls="dotted",
    #     )

    legend_handles = []
    if today_found:
        legend_handles.append(
            Line2D([0], [0], marker="*", linestyle="", markersize=10,
                   markerfacecolor="lightgray", markeredgecolor="black", label="Today")
        )
    if heat_found:
        legend_handles.append(
            Line2D([0], [0], marker="v", linestyle="", markersize=8,
                   markerfacecolor="lightgray", markeredgecolor="red", label="Heat")
        )
    if cadence_found:
        legend_handles.append(
            Line2D([0], [0], marker="D", linestyle="", markersize=7,
                   markerfacecolor="lightgray", markeredgecolor="blue", label="High Cadence")
        )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left")

    plt.tight_layout()
    plt.savefig(f"heartrate_vs_power_{time_minutes}.pdf", bbox_inches="tight")

    plt.close()


def main():
    try:
        sync_rides()
    except Exception as e:
        print(f"Skipping Garmin sync: {e}")
    for m in 3, 5, 10, 20, 30, 45, 60:
        make_hr_vs_power_plot(m, most_efficient=True)


if __name__ == "__main__":
    main()
