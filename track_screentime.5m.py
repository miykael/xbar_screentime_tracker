#!/Users/mnotter/.venvs/screentime/bin/python3
# <xbar.title>Screen time logger</xbar.title>
# <xbar.author>Michael Notter</xbar.author>
# <xbar.author.github>miykael</xbar.author.github>
# <xbar.desc>Tracks time spent in front of screen today and creates overview figures with a 5h shift</xbar.desc>
# <xbar.dependencies>python</xbar.dependencies>
# <swiftbar.title>Screen time logger</swiftbar.title>
# <swiftbar.refreshOnOpen>false</swiftbar.refreshOnOpen>

import os
import sys
import subprocess
from glob import glob
from datetime import timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("darkgrid")
sns.set_context("talk")

# -----------------------------------------------------------------------------------
# Global configuration and constants
# -----------------------------------------------------------------------------------
CATEGORIES = ["Active", "LidOpen", "ActiveDisplay", "ScreenLock"]
DEFAULT_DAILY_TIMEWINDOW = "24h"
DEFAULT_SAMPLE_RATE = "60s"
DEFAULT_DAYSHIFT = "5h"

# Visuals
OPEN_HOUR = "09:00"
CLOSE_HOUR = "17:00"

# Idle overlay
ENABLE_IDLE_OVERLAY = True
DEFAULT_IDLE_THRESHOLD_MINUTES = 5  # label gaps >= this many minutes

# Unified predicate for a single pass over the log
COMBINED_PREDICATE = (
    "("
    'process == "loginwindow" OR process == "powerd"'
    ") AND ("
    'eventMessage CONTAINS[c] "IOPMScheduleUserActiveChangedNotification" OR '
    'eventMessage CONTAINS[c] "ActiveDisplayList count" OR '
    'eventMessage CONTAINS[c] "Display is turned" OR '
    '(eventMessage CONTAINS[c] "SessionAgentNotificationCenter" AND eventMessage CONTAINS[c] "screenIsLocked")'
    ")"
)


# -----------------------------------------------------------------------------------
# Log extraction
# -----------------------------------------------------------------------------------
def _run_log_show(timewindow: str, predicate: str) -> list[str]:
    """
    Execute `log show` with a predicate and return timestamped lines.
    Uses compact style and filters out headers. No shell invocation.
    """
    cmd = [
        "log",
        "show",
        "--last",
        timewindow,
        "--style",
        "compact",
        "--predicate",
        predicate,
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0 or not cp.stdout:
        return []
    return [ln for ln in cp.stdout.splitlines() if ln[:4].isdigit()]


def _to_ts(two_tokens: list[str]) -> pd.Timestamp | None:
    """Convert first two whitespace tokens into pandas Timestamp."""
    try:
        return pd.to_datetime(f"{two_tokens[0]} {two_tokens[1]}", errors="coerce")
    except Exception:
        return None


# --- Line parsers (return 1/0) -----------------------------------------------------
def _parse_lid(line: str) -> int:
    """IOPMScheduleUserActiveChangedNotification received:N  -> N in {0,1}."""
    if "IOPMScheduleUserActiveChangedNotification" in line and "received:" in line:
        try:
            return int(line.split("received:")[-1].strip())
        except Exception:
            return 0
    return 0


def _parse_display(line: str) -> int:
    """
    Active display state:
      - loginwindow: ActiveDisplayList count = N  -> 1 if N>0
      - powerd: Display is turned on/off         -> 1/0
    """
    if "ActiveDisplayList" in line and "=" in line:
        try:
            return 1 if int(line.split("=")[-1].strip()) > 0 else 0
        except Exception:
            return 0
    if "powerd" in line:
        if "Display is turned on" in line:
            return 1
        if "Display is turned off" in line:
            return 0
    return 0


def _parse_lock(line: str) -> int:
    """
    Session lock:
      loginwindow/SessionAgentNotificationCenter: ... screenIsLocked = 1/0
      Return 1 for *unlocked*, 0 for locked.
    """
    if "SessionAgentNotificationCenter" in line and "screenIsLocked" in line:
        tail = line.split("screenIsLocked")[-1]
        digits = "".join(ch for ch in tail if ch.isdigit())
        if digits:
            return 0 if digits[-1] == "1" else 1
    return 0


# -----------------------------------------------------------------------------------
# Unified read (one log pass) and DataFrame construction
# -----------------------------------------------------------------------------------
def read_log_files(timewindow: str) -> dict[str, pd.DataFrame]:
    """
    Read all relevant events with a single `log show` call, then split into streams.
    Returns dict with keys: lid, active_display, screen_lock.
    """
    lines = _run_log_show(timewindow, COMBINED_PREDICATE)
    if not lines:

        def empty(col):
            return pd.DataFrame({col: []})

        return {
            "lid": empty("LidOpen"),
            "active_display": empty("ActiveDisplay"),
            "screen_lock": empty("ScreenLock"),
        }

    lid_ts, lid_vals = [], []
    disp_ts, disp_vals = [], []
    lock_ts, lock_vals = [], []

    for ln in lines:
        parts = ln.split()
        if len(parts) < 2:
            continue
        ts = _to_ts(parts[:2])
        if ts is None or pd.isna(ts):
            continue

        if "IOPMScheduleUserActiveChangedNotification" in ln:
            lid_ts.append(ts)
            lid_vals.append(_parse_lid(ln))
            continue
        if "ActiveDisplayList" in ln or "Display is turned" in ln:
            disp_ts.append(ts)
            disp_vals.append(_parse_display(ln))
            continue
        if "screenIsLocked" in ln:
            lock_ts.append(ts)
            lock_vals.append(_parse_lock(ln))
            continue

    def _mkdf(ts_list, vals_list, col):
        if not ts_list:
            return pd.DataFrame({col: []})
        return pd.DataFrame(
            {col: vals_list}, index=pd.DatetimeIndex(ts_list, name="ts")
        )

    results = {
        "lid": _mkdf(lid_ts, lid_vals, "LidOpen"),
        "active_display": _mkdf(disp_ts, disp_vals, "ActiveDisplay"),
        "screen_lock": _mkdf(lock_ts, lock_vals, "ScreenLock"),
    }

    # Pad each stream with [day start, now, day end] zeros for stable resampling
    time_start = (pd.Timestamp.now() - pd.Timedelta(timewindow)).floor("D")
    time_now = pd.Timestamp.now().round("s")
    time_end = pd.Timestamp.now().ceil("D")
    borders = pd.to_datetime([time_start, time_now, time_end])

    for k, df in results.items():
        if df.empty:
            continue
        pad = pd.DataFrame({df.columns[0]: [0, 0, 0]}, index=borders)
        results[k] = pd.concat([df, pad]).sort_index()

    return results


# -----------------------------------------------------------------------------------
# Unification and derived columns
# -----------------------------------------------------------------------------------
def unite_information(
    loginfos: dict,
    sample_rate: str = DEFAULT_SAMPLE_RATE,
    dayshift: str = DEFAULT_DAYSHIFT,
) -> pd.DataFrame:
    """
    Resample the three signals and compute:
      Active = LidOpen × ActiveDisplay × ScreenLock  (1/0 product, NaN for 0)
    Attach shifted-date fields for grouping and plotting.
    """
    df_lid = loginfos["lid"].resample(sample_rate).max().ffill().bfill()
    df_disp = loginfos["active_display"].resample(sample_rate).max().ffill().bfill()
    df_lock = loginfos["screen_lock"].resample(sample_rate).max().ffill().bfill()

    combined = pd.DataFrame(
        {
            "LidOpen": df_lid["LidOpen"],
            "ActiveDisplay": df_disp["ActiveDisplay"],
            "ScreenLock": df_lock["ScreenLock"],  # 1 == unlocked
        }
    )

    combined["Active"] = (
        combined[["LidOpen", "ActiveDisplay", "ScreenLock"]]
        .fillna(0)
        .prod(axis=1)
        .replace(0, np.nan)
    )

    combined["weeknumber"] = combined.index.dayofweek
    combined["dayname"] = combined.index.day_name()
    combined["daynumber"] = combined.index.day
    combined["date"] = (combined.index - pd.Timedelta(dayshift)).strftime("%Y-%m-%d")
    combined["timestamp"] = combined.index.copy()

    # HH:MM index for plotting
    combined.index = combined.index.strftime("%H:%M")
    return combined.replace(0, np.nan)


# -----------------------------------------------------------------------------------
# Interval computation
# -----------------------------------------------------------------------------------
def calculate_active_intervals(
    df: pd.DataFrame, active_col: str = "Active", day_shift: str = DEFAULT_DAYSHIFT
):
    """
    Find contiguous active intervals on the shifted day and return
    start/stop times in original time plus total active duration.
    """
    if df.empty:
        return (
            pd.Series(dtype="datetime64[ns]"),
            pd.Series(dtype="datetime64[ns]"),
            pd.Timedelta(0),
        )

    original_ts = pd.to_datetime(df.timestamp)
    shifted_ts = original_ts - pd.Timedelta(day_shift)

    df_sorted = df.copy()
    df_sorted["shifted_timestamp"] = shifted_ts
    df_sorted = df_sorted.sort_values("shifted_timestamp")

    shifted_day = pd.to_datetime(df_sorted["date"].iloc[0])  # midnight shifted
    day_start = shifted_day
    day_end = shifted_day + pd.Timedelta(days=1)
    now_shifted = pd.Timestamp.now() - pd.Timedelta(day_shift)

    series = df_sorted[active_col].fillna(0)
    diff = series.diff()
    starts = df_sorted["shifted_timestamp"][diff > 0]
    stops = df_sorted["shifted_timestamp"][diff < 0]

    if len(stops) > len(starts):
        starts = pd.concat([pd.Series([day_start], index=["start_boundary"]), starts])
    if len(starts) > len(stops):
        stop_candidate = now_shifted if now_shifted < day_end else day_end
        stops = pd.concat([stops, pd.Series([stop_candidate], index=["stop_boundary"])])

    starts = starts.sort_values()
    stops = stops.sort_values()

    durations = stops.values - starts.values
    total = sum(durations, pd.Timedelta(0))

    starts_orig = starts + pd.Timedelta(day_shift)
    stops_orig = stops + pd.Timedelta(day_shift)
    return starts_orig, stops_orig, total


# -----------------------------------------------------------------------------------
# Reporting and plotting
# -----------------------------------------------------------------------------------
def get_worktime_today(
    timewindow: str = DEFAULT_DAILY_TIMEWINDOW, day_shift: str = DEFAULT_DAYSHIFT
) -> timedelta:
    """Compute total active time for the shifted 'today'."""
    df = unite_information(read_log_files(timewindow), dayshift=day_shift)
    target_date = (pd.Timestamp.now() - pd.Timedelta(day_shift)).strftime("%Y-%m-%d")
    df_target = df[df.date == target_date]
    if df_target.empty or "date" not in df_target.columns:
        return timedelta(0)
    _, _, total = calculate_active_intervals(df_target, "Active", day_shift)
    return total


def report_worktime():
    """Print SwiftBar line with HH:MM and color/icon thresholding."""
    time_worked = get_worktime_today()
    mins = int(time_worked.total_seconds() // 60)
    txt_out = f"{mins // 60:02d}:{mins % 60:02d}"
    if mins < 4.1 * 60:
        color_hex, icon = (
            "#00BFFF",
            "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAAZ0lEQVQ4jeXQsQmAQBQD0CdYW7iAG7qNm7iGK1gIdjrD2dzBFSpnJ5rm88PPTwhvQ4MRXamgjrPDgBYTlsiHG22VP1gwY0WfCavSJOn4keAMf+9gj87bU9eEEPeQ8cUd5An2C4Ov4gDIJRmeUPVagQAAAABJRU5ErkJggg==",
        )
    elif mins < 8.2 * 60:
        color_hex, icon = (
            "#40FF61",
            "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAAcUlEQVQ4je3SPQqDQBDF8R8ewV7BgwYtxJt4u0A6cStt/Co2mCkCKfLgMTA7839TLL+kAQnLjRP6HCCh/CCoxJx7WALXHrNFYCmrP+AE1Fu9+wO7obqCRnSB4G7bOfRCEwA0eF4bLabMqe884REI/KJWwGUqilB/fg8AAAAASUVORK5CYII=",
        )
    elif mins < 9 * 60:
        color_hex, icon = (
            "#FFBF00",
            "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAApElEQVQ4jcXSIQoCQRTG8d8ubhZEjAbTZovRG3gBmwewegWL0RN4GYvdImgRjaaNaxlhWVYdFPGDgcf73v+bNzD8WONwPtIEF1xDDWUsPA3wCEOcMYsNmOOIvNLLcYoJWOCAQYPXxx7LJjDBCjv0XlzQwRZrpI9mhk1Yr/1uxTBTBiZroYsimLfKYP29SW2mCOxToIzxUl/q/wFJpY7+pg3sH3UHCfMenVq4Fs0AAAAASUVORK5CYII=",
        )
    else:
        color_hex, icon = (
            "#FF4040",
            "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAAnUlEQVQ4jb3TSwrCMBgE4O8yXeveF3oZwWsoihvPWV0VrWBXdWGFKE1bnwMh+TP/TBKY8ANMkKF8GhnGbeIh9phFuBTTJnEaEXc2eRs7nLGu6vDtqv0cq4AvglqO/pNpGawLJLgEXILjvWGOAxYRg7JmLrAMT+zh9MINHrCtxJuIwarilzXc9zDQnoORW9CiOWgyGekYorEP/sL/cQWw8joTqoKrxAAAAABJRU5ErkJggg==",
        )
    print(f"{txt_out} | color={color_hex} templateImage='{icon}'")


def plot_daily_stats(
    df: pd.DataFrame,
    date: str,
    filename: str,
    plot_restrictions: tuple[str, str] = ("08:00", "18:00"),
):
    """
    Plot the daily tracks with optional idle gap labels.
    Saves a zipped CSV of the plotted rows next to the PNG.
    """
    categories = CATEGORIES
    fig, ax = plt.subplots(figsize=(12, 4))
    group = df[df.date == date]

    for j, cat in enumerate(categories):
        ds = -group[cat] - 0.125 * j
        ds.plot(
            ax=ax,
            legend=True,
            linestyle="-" if j == 0 else "--",
            linewidth=3 if j == 0 else 1,
            drawstyle="steps-post",
        )

        # Idle gap labels on Active only
        if ENABLE_IDLE_OVERLAY and cat == "Active":
            is_nan = ds.isna()
            gap_starts = np.where(is_nan & ~is_nan.shift(1).fillna(False))[0]
            gap_ends = np.where(is_nan & ~is_nan.shift(-1).fillna(False))[0]
            for start, end in zip(gap_starts, gap_ends):
                if start == 0 or end == len(ds) - 1:
                    continue
                t0 = pd.to_datetime(group.timestamp.iloc[start])
                t1 = pd.to_datetime(group.timestamp.iloc[end])
                gap_mins = (t1 - t0).total_seconds() / 60
                if gap_mins >= DEFAULT_IDLE_THRESHOLD_MINUTES:
                    x_pos = (start + end) / 2
                    y_pos = ds.iloc[max(start - 1, 0)] + 0.01
                    ax.text(
                        x_pos,
                        y_pos,
                        f"{int(gap_mins)}m",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="darkblue",
                        zorder=5,
                    )

    # Zipped CSV next to PNG
    csv_filename = filename.replace(".png", ".csv.zip").replace(
        "_log_plots", "_log_data"
    )
    group.to_csv(
        csv_filename,
        compression={
            "method": "zip",
            "archive_name": os.path.basename(csv_filename).replace(".zip", ""),
        },
    )

    ax.set_ylim(np.array(ax.get_ylim()) + np.array((-0.15, 0.1)))
    try:
        open_idx = np.where(group.index == OPEN_HOUR)[0]
        close_idx = np.where(group.index == CLOSE_HOUR)[0]
        if open_idx.size and close_idx.size:
            y_lim = ax.get_ylim()
            ax.vlines(int(open_idx[0]), *y_lim, colors="k", linestyle=":", linewidth=2)
            ax.vlines(int(close_idx[0]), *y_lim, colors="k", linestyle=":", linewidth=2)
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
        valid = group["Active"].notna().to_numpy()
        idx = np.argwhere(valid)
        if idx.size:
            idx_min = int(idx.min()) - 30
            idx_max = int(idx.max()) + 30
            a = np.where(group.index == plot_restrictions[0])[0]
            b = np.where(group.index == plot_restrictions[1])[0]
            start_idx = int(a[0]) if a.size else 0
            end_idx = int(b[0]) if b.size else len(group) - 1
            ax.set_xlim(min(idx_min, start_idx), max(idx_max, end_idx))

    box = ax.get_position()
    ax.set_position(
        [box.x0, box.y0 + box.height * 0.1, box.width * 0.8, box.height * 0.9]
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.55, -0.43), ncol=5, fontsize=15.5)

    delta_start, delta_stop, total_active = calculate_active_intervals(
        group, "Active", DEFAULT_DAYSHIFT
    )
    total_str = str(total_active)[-8:]
    text_box = "Delta\n"
    for start, delta in zip(
        delta_start.index, pd.to_timedelta(delta_stop.values - delta_start.values)
    ):
        text_box += f"{start} - {str(delta)[-8:]}\n"
    text_box += f"\nTotal: {total_str:>9}"
    ax.text(
        1.03,
        0.97,
        text_box,
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        bbox=dict(boxstyle="round", facecolor="gray", alpha=0.1),
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


def create_daily_stats(
    out_path: str, days_back: int = 5, dayshift: str = DEFAULT_DAYSHIFT
):
    """Backfill plots for the last `days_back` shifted days where images are missing."""
    recorded_stats = sorted(glob(os.path.join(out_path, "day_*")))
    recorded_dates = [
        d.split("day_")[1].split(".")[0] for d in recorded_stats if "day_20" in d
    ]
    missing = []
    today = pd.Timestamp.now() - pd.Timedelta(dayshift)
    for i in range(days_back):
        check_date = (today - pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
        if check_date not in recorded_dates:
            missing.append(check_date)
    if missing:
        df = unite_information(read_log_files(f"{days_back + 1}d"), dayshift=dayshift)
        for date in missing:
            plot_daily_stats(df, date, os.path.join(out_path, f"day_{date}.png"))


def plot_stats_today(
    out_path: str,
    timewindow: str = DEFAULT_DAILY_TIMEWINDOW,
    show_plot: bool = True,
    day_shift: str = DEFAULT_DAYSHIFT,
):
    """Generate today's plot image and optionally open it."""
    df = unite_information(read_log_files(timewindow), dayshift=day_shift)
    if df.empty:
        return
    target_date = (pd.Timestamp.now() - pd.Timedelta(day_shift)).strftime("%Y-%m-%d")
    filename = os.path.join(out_path, "current_day.png")
    plot_daily_stats(df, target_date, filename)
    if show_plot:
        os.system(f"open {filename}")


# -----------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    home_path = os.path.expanduser("~")
    out_path = os.path.join(home_path, "Documents", "screentime_log_plots")
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(
        os.path.join(home_path, "Documents", "screentime_log_data"), exist_ok=True
    )

    script_path = os.path.abspath(__file__)
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        plot_stats_today(out_path)
    else:
        create_daily_stats(out_path)
        report_worktime()
        print("---")
        print(
            f"Stats Today | bash='{script_path}' param1='stats' refresh=true terminal=false"
        )
