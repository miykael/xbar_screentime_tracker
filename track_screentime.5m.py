#!/Users/mnotter/miniconda3/bin/python
# <xbar.title>Screen time logger</xbar.title>
# <xbar.version>v0.3</xbar.version>
# <xbar.author>Michael Notter</xbar.author>
# <xbar.author.github>miykael</xbar.author.github>
# <xbar.desc>Tracks time spent in front of screen today and creates overview figures with a 5h shift</xbar.desc>
# <xbar.dependencies>python</xbar.dependencies>

import os
import pathlib
import sys
import subprocess
from glob import glob
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("darkgrid")
sns.set_context("talk")

# -----------------------------------------------------------------------------------
# Global configuration and constants
# -----------------------------------------------------------------------------------
# The list of categories to be plotted. "Active" is computed from all three conditions.
CATEGORIES = ["Active", "LidOpen", "ActiveDisplay", "ScreenLock"]
DEFAULT_DAILY_TIMEWINDOW = "1d"
DEFAULT_SCREENSAVER_OFFSET = "10m"
DEFAULT_SAMPLE_RATE = "60s"  # lowercase 's' avoids deprecation warnings
DEFAULT_DAYSHIFT = "5h"


# -----------------------------------------------------------------------------------
# Log extraction and DataFrame creation functions
# -----------------------------------------------------------------------------------
def collect_log_information(timewindow, predicate):
    """
    Executes the macOS log command and returns a list of log lines.
    The predicate is enclosed in single quotes so that internal double quotes are preserved.
    """
    command = f"log show --last {timewindow} --style compact --predicate '{predicate}'"
    try:
        output = subprocess.check_output(command, shell=True, text=True)
    except subprocess.CalledProcessError:
        return []
    lines = output.splitlines()
    if lines and lines[0].startswith("Timestamp"):
        return lines[1:]
    return lines


def create_log_dataframe(log_lines, key, transform_func):
    """
    Converts log lines into a DataFrame.
    Assumes each log line starts with a timestamp (first two tokens)
    and applies transform_func to extract a numeric value.
    """
    data = []
    timestamps = []
    for line in log_lines:
        parts = line.split()
        timestamp_str = parts[0] + " " + parts[1]
        timestamp = pd.to_datetime(timestamp_str, errors="coerce")
        if pd.isnull(timestamp):
            continue
        try:
            value = transform_func(line)
        except Exception:
            value = 0
        timestamps.append(timestamp)
        data.append(value)
    if not timestamps:
        return pd.DataFrame({key: []})
    return pd.DataFrame({key: data}, index=pd.DatetimeIndex(timestamps))


# --- Transformation functions (returning 0 by default if not found) ---
def extract_lid_value(line):
    # Expect lines like: "... IOPMScheduleUserActiveChangedNotification received:1"
    if "received:" in line:
        try:
            return int(line.split("received:")[-1].strip())
        except Exception:
            return 0
    return 0


def extract_active_display_value(line):
    # Expect lines like: "... ActiveDisplayList count = 1"
    if "=" in line:
        try:
            return 1 if int(line.split("=")[-1].strip()) > 0 else 0
        except Exception:
            return 0
    return 0


def extract_screenlock_value(line):
    # Expect lines like: "... _setStatusViewHidden:] | Enter, hidden: 0" (0 means locked, so 1 means unlocked)
    if "hidden:" in line:
        try:
            # Return 1 for unlocked, 0 for locked.
            return 1 - int(line.split("hidden:")[-1].strip())
        except Exception:
            return 0
    return 0


def read_log_files(timewindow, screensaver_offset=DEFAULT_SCREENSAVER_OFFSET):
    """
    Reads log entries for various events concurrently based on preset predicates
    and converts them into a dictionary mapping event names to DataFrames.
    """
    predicates = {
        "LidOpen": 'subsystem == "com.apple.loginwindow.logging" AND composedMessage CONTAINS[c] "IOPMScheduleUserActiveChangedNotification"',
        "ActiveDisplay": 'subsystem == "com.apple.loginwindow.logging" AND composedMessage CONTAINS[c] "ActiveDisplayList count"',
        "ScreenLock": 'subsystem == "com.apple.loginwindow.logging" AND composedMessage CONTAINS[c] "_setStatusViewHidden:] | Enter, hidden"',
    }

    transforms = {
        "LidOpen": extract_lid_value,
        "ActiveDisplay": extract_active_display_value,
        "ScreenLock": extract_screenlock_value,
    }
    key_map = {
        "LidOpen": "lid",
        "ActiveDisplay": "active_display",
        "ScreenLock": "screen_lock",
    }

    def fetch_data(orig_key):
        pred = predicates[orig_key]
        tf = transforms[orig_key]
        log_lines = collect_log_information(timewindow, pred)
        df = create_log_dataframe(log_lines, orig_key, tf)
        return key_map[orig_key], df

    results = {}
    with ThreadPoolExecutor(max_workers=len(predicates)) as executor:
        futures = {executor.submit(fetch_data, key): key for key in predicates}
        for future in as_completed(futures):
            k, df = future.result()
            results[k] = df

    time_start = (pd.Timestamp.now() - pd.Timedelta(timewindow)).floor("D")
    time_now = pd.Timestamp.now().round("s")
    time_end = pd.Timestamp.now().ceil("D")
    new_borders = pd.to_datetime([time_start, time_now, time_end])
    for key, df in results.items():
        if df.empty:
            continue
        borders_df = pd.DataFrame({df.columns[0]: 0}, index=new_borders)
        results[key] = pd.concat([df, borders_df]).sort_index()
    return results


def unite_information(
    loginfos, sample_rate=DEFAULT_SAMPLE_RATE, dayshift=DEFAULT_DAYSHIFT
):
    """
    Combines separate DataFrames of log events into a unified DataFrame.
    It also subtracts the day shift from timestamps so that each event is labeled
    with a date corresponding to (timestamp - day_shift).
    """
    df_lid = loginfos["lid"].resample(sample_rate).max().ffill().bfill()
    df_active_display = (
        loginfos["active_display"].resample(sample_rate).max().ffill().bfill()
    )
    df_screen_lock = loginfos["screen_lock"].resample(sample_rate).max().ffill().bfill()

    combined = pd.DataFrame(
        {
            "LidOpen": df_lid.LidOpen,
            "ActiveDisplay": df_active_display.ActiveDisplay,
            "ScreenLock": df_screen_lock.ScreenLock,
        }
    )

    # Compute "Active" as the product of all three conditions.
    combined["Active"] = (
        combined[["LidOpen", "ActiveDisplay", "ScreenLock"]]
        .fillna(0)
        .prod(axis=1)
        .replace(0, np.nan)
    )

    # Add additional time info.
    # Subtract the day shift to determine which workday the log entry belongs to.
    combined["weeknumber"] = combined.index.dayofweek
    combined["dayname"] = combined.index.day_name()
    combined["daynumber"] = combined.index.day
    combined["date"] = (combined.index - pd.Timedelta(dayshift)).strftime("%Y-%m-%d")
    combined["timestamp"] = combined.index.copy()
    # For plotting, we keep the time-of-day as HH:MM.
    combined.index = combined.index.strftime("%H:%M")
    combined = combined.replace(0, np.nan)
    return combined


# -----------------------------------------------------------------------------------
# Helper to compute active intervals
# -----------------------------------------------------------------------------------
def calculate_active_intervals(df, active_col="Active", day_shift=DEFAULT_DAYSHIFT):
    """
    Determines active time intervals using shifted timestamps to group events,
    then re-adds the shift so that reported times are in original time.

    For example, if a session spans from 09:30 on Feb 15 to 02:30 on Feb 16 (original time),
    subtracting a 5h shift turns these into 04:30 and 21:30 on the shifted Feb 15.
    After computing the active interval (and padding boundaries if necessary), the 5h is re-added
    so that the reported times remain in the original time.
    """
    # Convert original timestamps and compute shifted timestamps.
    original_ts = pd.to_datetime(df.timestamp)
    shifted_ts = original_ts - pd.Timedelta(day_shift)

    # Work on a copy sorted by shifted time.
    df_sorted = df.copy()
    df_sorted["shifted_timestamp"] = shifted_ts
    df_sorted = df_sorted.sort_values("shifted_timestamp")

    # The shifted day (as a date string) is already in df["date"].
    # We assume that df has been filtered to only contain the target day.
    shifted_day_str = df_sorted["date"].iloc[0]
    shifted_day = pd.to_datetime(shifted_day_str)  # represents midnight in shifted time
    day_start = shifted_day
    day_end = shifted_day + pd.Timedelta(days=1)

    # Use the current time in shifted space.
    current_shifted = pd.Timestamp.now() - pd.Timedelta(day_shift)

    active_series = df_sorted[active_col].fillna(0)
    diff = active_series.diff()

    # Identify state transitions.
    start_events = df_sorted["shifted_timestamp"][diff > 0]
    stop_events = df_sorted["shifted_timestamp"][diff < 0]

    # If the first event is a stop (i.e. the session started before logging began), add the shifted day start.
    if len(stop_events) > len(start_events):
        start_events = pd.concat(
            [pd.Series([day_start], index=["start_boundary"]), start_events]
        )

    # If a session is still active (more start than stop events), use the current shifted time (but not beyond day_end).
    if len(start_events) > len(stop_events):
        stop_candidate = current_shifted if current_shifted < day_end else day_end
        stop_events = pd.concat(
            [stop_events, pd.Series([stop_candidate], index=["stop_boundary"])]
        )

    # Ensure proper pairing.
    start_events = start_events.sort_values()
    stop_events = stop_events.sort_values()

    durations = stop_events.values - start_events.values
    total_duration = sum(durations, pd.Timedelta(0))

    # Re-add the day shift so that the intervals are reported in original time.
    start_events_orig = start_events + pd.Timedelta(day_shift)
    stop_events_orig = stop_events + pd.Timedelta(day_shift)

    return start_events_orig, stop_events_orig, total_duration


# -----------------------------------------------------------------------------------
# Reporting and plotting functions
# -----------------------------------------------------------------------------------
def get_worktime_today(timewindow=DEFAULT_DAILY_TIMEWINDOW, day_shift=DEFAULT_DAYSHIFT):
    """
    Computes the total active work time for the target day.
    The target day is determined as (current time - day_shift).
    Logs from the last 48h are collected, then only the entries for the target day are used.
    """
    loginfos = read_log_files(timewindow)
    df = unite_information(loginfos, dayshift=day_shift)
    # Determine the target day (shifted current time)
    target_date = (pd.Timestamp.now() - pd.Timedelta(day_shift)).strftime("%Y-%m-%d")
    df_target = df[df.date == target_date]
    if df_target.empty or "date" not in df_target.columns:
        return timedelta(0)
    # Compute active intervals on the target day.
    _, _, total_duration = calculate_active_intervals(
        df_target, active_col="Active", day_shift=day_shift
    )
    return total_duration


def report_worktime():
    """
    Calculates work time for the target day and prints the output.
    Only this output (and the final menu options) will be printed.
    """
    time_worked = get_worktime_today()
    txt_out = str(time_worked)[-8:-3]
    if time_worked.total_seconds() / 60 < 4.1 * 60:
        color, icon = (
            "1;36",
            "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAAZ0lEQVQ4jeXQsQmAQBQD0CdYW7iAG7qNm7iGK1gIdjrD2dzBFSpnJ5rm88PPTwhvQ4MRXamgjrPDgBYTlsiHG22VP1gwY0WfCavSJOn4keAMf+9gj87bU9eEEPeQ8cUd5An2C4Ov4gDIJRmeUPVagQAAAABJRU5ErkJggg==",
        )
    elif time_worked.total_seconds() / 60 < 8.2 * 60:
        color, icon = (
            "1;32",
            "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAAcUlEQVQ4je3SPQqDQBDF8R8ewV7BgwYtxJt4u0A6cStt/Co2mCkCKfLgMTA7839TLL+kAQnLjRP6HCCh/CCoxJx7WALXHrNFYCmrP+AE1Fu9+wO7obqCRnSB4G7bOfRCEwA0eF4bLabMqe884REI/KJWwGUqilB/fg8AAAAASUVORK5CYII=",
        )
    elif time_worked.total_seconds() / 60 < 9 * 60:
        color, icon = (
            "1;33",
            "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAApElEQVQ4jcXSIQoCQRTG8d8ubhZEjAbTZovRG3gBmwewegWL0RN4GYvdImgRjaaNaxlhWVYdFPGDgcf73v+bNzD8WONwPtIEF1xDDWUsPA3wCEOcMYsNmOOIvNLLcYoJWOCAQYPXxx7LJjDBCjv0XlzQwRZrpI9mhk1Yr/1uxTBTBiZroYsimLfKYP29SW2mCOxToIzxUl/q/wFJpY7+pg3sH3UHCfMenVq4Fs0AAAAASUVORK5CYII=",
        )
    else:
        color, icon = (
            "1;31",
            "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAAnUlEQVQ4jb3TSwrCMBgE4O8yXeveF3oZwWsoihvPWV0VrWBXdWGFKE1bnwMh+TP/TBKY8ANMkKF8GhnGbeIh9phFuBTTJnEaEXc2eRs7nLGu6vDtqv0cq4AvglqO/pNpGawLJLgEXILjvWGOAxYRg7JmLrAMT+zh9MINHrCtxJuIwarilzXc9zDQnoORW9CiOWgyGekYorEP/sL/cQWw8joTqoKrxAAAAABJRU5ErkJggg==",
        )
    output = f"\033[{color}m{txt_out}\033[0m | templateImage='{icon}'"
    print(output)


def plot_daily_stats(df, date, filename, plot_restrictions=["08:00", "18:00"]):
    """
    Plots daily statistics for the given date.
    Saves a CSV of the data and marks opening/closing times.
    Uses a step-post style for clear state transitions.
    """
    categories = CATEGORIES  # This now includes our new columns.
    fig, ax = plt.subplots(figsize=(12, 4))
    group = df[df.date == date]
    for j, category in enumerate(categories):
        line_style = "-" if j == 0 else "--"
        line_width = 3 if j == 0 else 1
        ds = -group[category] - 0.125 * j
        ds.plot(
            ax=ax,
            legend=True,
            linestyle=line_style,
            linewidth=line_width,
            drawstyle="steps-post",
        )

        # Add gap duration labels for the Active category only
        if category == "Active":

            # Find gaps (NaN values)
            is_nan = ds.isna()
            gap_starts = np.where(is_nan & ~is_nan.shift(1).fillna(False))[0]
            gap_ends = np.where(is_nan & ~is_nan.shift(-1).fillna(False))[0]

            # Process each gap
            for start, end in zip(gap_starts, gap_ends):
                # Skip if this is at the start or end of the day
                if start == 0 or end == len(ds) - 1:
                    continue

                # Calculate gap duration in minutes
                start_time = pd.to_datetime(group.timestamp.iloc[start])
                end_time = pd.to_datetime(group.timestamp.iloc[end])
                gap_mins = (end_time - start_time).total_seconds() / 60

                # Only label gaps > 5 minutes
                if gap_mins >= 5:

                    # Position label in middle of gap
                    x_pos = (start + end) / 2
                    y_pos = ds.iloc[start - 1] + 0.01  # Slightly above the line

                    # Add text label
                    ax.text(x_pos, y_pos, f"{int(gap_mins)}m",
                           horizontalalignment='center',
                           verticalalignment='bottom',
                           fontsize=10,
                           color='grey')

    csv_filename = filename.replace(".png", ".csv.zip").replace(
        "_log_plots", "_log_data"
    )
    group.to_csv(csv_filename)
    ax.set_ylim(np.array(ax.get_ylim()) + np.array((-0.1, 0.1)))
    try:
        open_idx = np.argwhere(group.index == "09:00")[0][0]
        close_idx = np.argwhere(group.index == "17:00")[0][0]
        y_lim = ax.get_ylim()
        ax.vlines(open_idx, *y_lim, colors="k", linestyle=":", linewidth=2)
        ax.vlines(close_idx, *y_lim, colors="k", linestyle=":", linewidth=2)
    except IndexError:
        pass
    tick_indices = np.argwhere(
        group.index.str.contains(":00") | group.index.str.contains(":30")
    ).ravel()
    tick_labels = group.index[tick_indices]
    ax.set_xticks(tick_indices[::2])
    ax.set_xticklabels(tick_labels[::2], rotation=90)
    ax.set_yticks([])
    if plot_restrictions and group["Active"].notna().sum():
        valid_entries = group["Active"].notna()
        idx = np.argwhere(valid_entries.to_numpy())
        idx_min = idx.min() - 30
        idx_max = idx.max() + 30
        start_idx = np.argmax(group.index == plot_restrictions[0])
        end_idx = np.argmax(group.index == plot_restrictions[1])
        idx_min = min(idx_min, start_idx)
        idx_max = max(idx_max, end_idx)
        ax.set_xlim(idx_min, idx_max)
    box = ax.get_position()
    ax.set_position(
        [box.x0, box.y0 + box.height * 0.1, box.width * 0.8, box.height * 0.9]
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.55, -0.43), ncol=5, fontsize=15.5)
    delta_start, delta_stop, total_active = calculate_active_intervals(
        group, active_col="Active", day_shift=DEFAULT_DAYSHIFT
    )
    total_str = str(total_active)[-8:]
    text_box = "Delta\n"
    for start, delta in zip(
        delta_start.index, pd.to_timedelta(delta_stop.values - delta_start.values)
    ):
        text_box += f"{start} - {str(delta)[-8:]}\n"
    text_box += f"\nTotal: {total_str:>9}"
    props = dict(boxstyle="round", facecolor="gray", alpha=0.1)
    ax.text(
        1.03,
        0.97,
        text_box,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
        family="monospace",
    )
    week_day = (
        pd.to_datetime(group.timestamp.iloc[0]).day_name()[:3]
        if not group.empty
        else ""
    )
    plt.title(f"Date: {date} ({week_day}) - Total: {total_str}")
    plt.tight_layout(rect=(-0.05, -0.03, 0.85, 1.03))
    plt.savefig(filename)
    plt.close()


def create_daily_stats(out_path, days_back=5, dayshift=DEFAULT_DAYSHIFT):
    """
    Creates daily plots for any missing day within the last 'days_back' days.
    """
    recorded_stats = sorted(glob(os.path.join(out_path, "day_*")))
    recorded_dates = [
        d.split("day_")[1].split(".")[0] for d in recorded_stats if "day_20" in d
    ]
    missing_dates = []
    today = pd.Timestamp.now() - pd.Timedelta(dayshift)
    for i in range(days_back):
        check_date = (today - pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
        if check_date not in recorded_dates:
            missing_dates.append(check_date)
    if missing_dates:
        loginfos = read_log_files(DEFAULT_DAILY_TIMEWINDOW)
        df = unite_information(loginfos, dayshift=dayshift)
        for date in missing_dates:
            filename = os.path.join(out_path, f"day_{date}.png")
            plot_daily_stats(df, date, filename)


def plot_stats_today(
    out_path,
    timewindow=DEFAULT_DAILY_TIMEWINDOW,
    show_plot=True,
    day_shift=DEFAULT_DAYSHIFT,
):
    """
    Plots statistics for the target day.
    The target day is determined as (current time - day_shift).
    The logs from the last 48h are collected, then only the entries for the target day are used.
    The result is saved as 'current_day.png'.
    """
    loginfos = read_log_files(timewindow)
    df = unite_information(loginfos, dayshift=day_shift)
    if df.empty:
        return
    target_date = (pd.Timestamp.now() - pd.Timedelta(day_shift)).strftime("%Y-%m-%d")
    filename = os.path.join(out_path, "current_day.png")
    plot_daily_stats(df, target_date, filename)
    if show_plot:
        os.system(f"open {filename}")


# -----------------------------------------------------------------------------------
# Main script entry point (for command line and xbar integration)
# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    home_path = os.path.expanduser("~")
    out_path = os.path.join(home_path, "Documents", "screentime_log_plots")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_path_data = os.path.join(home_path, "Documents", "screentime_log_data")
    if not os.path.exists(out_path_data):
        os.makedirs(out_path_data)

    arguments = sys.argv
    if len(arguments) > 1:
        if arguments[1] == "stats":
            plot_stats_today(out_path)
    else:
        create_daily_stats(out_path)
        report_worktime()

        # Final menu command options printed for xbar.
        xbar_dir = pathlib.Path(__file__).parent.absolute()
        file_path = os.path.join(xbar_dir, __file__)
        cmd_template = "shell=/Users/mnotter/miniconda3/bin/python param1='{0}' param2='{1}' terminal=false"
        print("---")
        print("Stats Today | {0}".format(cmd_template.format(file_path, "stats")))
