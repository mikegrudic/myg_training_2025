import datetime
from pandas import DataFrame
from matplotlib import pyplot as plt
import numpy as np


def get_num_runs_from_volume(volume):
    if volume > 6:
        if volume > 9:
            return 4
        else:
            return 3
    return 2


def get_miles_today(row):
    match row["Date"].weekday():
        case 3:
            return row["Short Run 1"]
        case 6:
            return row["Short Run 2"]
        case 4:
            return row["Short Run 3"]
        case 0:
            return row["Long Run"]
        case _:
            return 0


def get_lowimpact_minutes_today(row):
    match row["Date"].weekday():
        case 3:
            return row["Short Low-Impact 1"]
        case 4:
            return row["Short Low-Impact 2"]
        case 6:
            return row["Short Low-Impact 3"]
        case 0:
            return row["Long Low-Impact"]
        case _:
            return 0


def weekly_volume_percentage(volume, max_volume=25, sharpness=1.0):
    return max(30.0 * (1 - (volume / max_volume) ** sharpness), 5)
    # if volume < 9:
    #     return 40.
    # elif volume < 15:
    #     return 30.
    # elif volume < 25:
    #     return 20
    # return 10.


def make_program_csv():
    program = DataFrame()
    t0 = datetime.date(2025, 1, 20)
    race_day = datetime.date(2025, 5, 3)
    dt = race_day - t0
    print(f"{dt} until race day.")
    program["Date"] = [t0]
    program["Day"] = [1]
    program["Short Run 1"] = [1.5]
    program["Short Run 2"] = [1.5]
    program["Short Run 3"] = [0.0]
    program["Long Run"] = [0.0]
    program["Daily Miles"] = [1.5]
    program["Short Low-Impact 1"] = program["Short Low-Impact 2"] = program["Short Low-Impact 3"] = [0]
    program["Long Low-Impact"] = program["Daily Low-Impact Minutes"] = program["Weekly Low-Impact Minutes"] = [0]

    weeks_between_deloads = 4
    days_since_last_deload = 0
    weekly_taper_percentage = 50.0
    peak_weeks = 2
    taper_weeks = 2
    initial_short_run = 1.5
    initial_runs_per_week = 2
    initial_miles_per_week = initial_runs_per_week * initial_short_run
    max_volume = 40.0

    volume = initial_miles_per_week
    program["Miles Per Week"] = volume

    num_runs = 2
    t = t0
    while t < race_day:
        t += datetime.timedelta(days=1)
        row = program.loc[len(program) - 1].copy()
        row["Date"] = t
        # print(t.weekday())

        row["Day"] += 1
        time_until_race_day = race_day - t
        if time_until_race_day.days > 7 * (peak_weeks + taper_weeks):
            volume *= (1 + weekly_volume_percentage(volume, max_volume) / 100.0) ** (1.0 / 7)
            if days_since_last_deload > 7 * weeks_between_deloads and get_num_runs_from_volume(volume) >= 4:
                volume *= 1 - weekly_volume_percentage(volume, max_volume) / 100.0
                days_since_last_deload = 0
            low_impact_minutes = (max_volume - volume) * 10
        elif time_until_race_day.days < 7 * taper_weeks:
            taper_fac = (1 - weekly_taper_percentage / 100.0) ** (1.0 / 7)
            volume *= taper_fac
            low_impact_minutes *= taper_fac

        print(low_impact_minutes)
        days_since_last_deload += 1

        row["Miles Per Week"] = volume
        num_runs = get_num_runs_from_volume(volume)
        if num_runs == 2:
            row["Short Run 1"] = row["Short Run 2"] = volume / 2
            row["Short Run 3"] = row["Long Run"] = 0.0
        elif num_runs == 3:
            row["Short Run 1"] = row["Short Run 2"] = row["Short Run 3"] = volume / 3
            row["Long Run"] = 0.0
        else:
            row["Short Run 1"] = row["Short Run 2"] = row["Short Run 3"] = volume / 5
            row["Long Run"] = volume * 2 / 5

        row["Short Low-Impact 1"] = row["Short Low-Impact 2"] = row["Short Low-Impact 3"] = low_impact_minutes * 0.2
        row["Long Low-Impact"] = low_impact_minutes * 0.4

        row["Daily Miles"] = get_miles_today(row)
        row["Daily Low-Impact Minutes"] = get_lowimpact_minutes_today(row)
        row["Weekly Low-Impact Minutes"] = low_impact_minutes
        program.loc[len(program)] = row

    #    plt.figure(figsize=(8,3))
    fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    run_day = program["Daily Miles"] > 0
    #    lowimpact_day = # (program["Daily Low-Impact Minutes"] > 0)
    plotargs = {"marker": ".", "ls": "", "color": "black"}
    ax[0].plot((program["Day"][1::7] / 7) + 1, program["Weekly Low-Impact Minutes"][1::7], **plotargs)
    #    ax[0].plot(program["Day"][lowimpact_day]/7 , program["Weekly Low-Impact Minutes"][lowimpact_day],**plotargs)
    ax[1].plot((program["Day"][run_day] / 7) + 1, program["Daily Miles"][run_day], **plotargs)
    ax[0].set_yticks([100, 200, 300, 400])
    ax[1].set_yticks(range(1, 8))
    plt.xticks(range(1, 17))  # ,rotation=90)#,range(11

    plt.subplots_adjust(hspace=0)
    plt.xlabel("Week")
    ax[1].set_ylabel("Daily Running Miles")
    ax[0].set_ylabel("Weekly Low-Impact Minutes")
    plt.savefig("Program.pdf", bbox_inches="tight")
    program.to_csv("program.csv")


if __name__ == "__main__":
    make_program_csv()
