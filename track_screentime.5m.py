#!/anaconda3/bin/python

# <bitbar.title>Screen time logger</bitbar.title>
# <bitbar.version>v0.1</bitbar.version>
# <bitbar.author>Michael Notter</bitbar.author>
# <bitbar.author.github>miykael</bitbar.author.github>
# <bitbar.desc>Tracks time spent in front of screen today and creates overview figures</bitbar.desc>
# <bitbar.dependencies>python</bitbar.dependencies>

import os
import sys
import pathlib
import numpy as np
import pandas as pd
from glob import glob

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_context('talk')

def collect_log_information(timewindow, predicate):

    # Define log command
    cmd = 'log show --last %s --style compact ' % timewindow
    cmd += '--predicate \'%s\'' % predicate

    # Collect log information
    stream = os.popen(cmd)
    return stream.read().splitlines()[1:]


def read_log_files(timewindow='7d', screensaver_offset='10m'):

    # Extract Screen lock information
    time_logins = []
    predicate = 'subsystem == "com.apple.corespeech" and '
    predicate += 'eventMessage contains "Screen Lock Status Changed"'
    for o in collect_log_information(timewindow, predicate):
        time_logins.append([o[:23], int(o[o.find('Screen Lock Status Changed')+29:]=='Unlocked')])

    # Extract LidStatus information
    time_lid = []
    predicate = 'subsystem == "com.apple.bluetooth" and '
    predicate += 'eventMessage contains "LidStatus"'
    for o in collect_log_information(timewindow, predicate)[1:]:
        time_lid.append([o[:23], int(o[-3]), int(o[-1])])

    # Extract Sleep State information
    time_sleep = []
    predicate = 'process == "PowerChime" and '
    predicate += 'eventMessage contains "WAKE" and eventMessage contains "kStateAwake"'
    for o in collect_log_information(timewindow, predicate):
        time_sleep.append([o[:23], int(o[o.find(' WAKE:')+6:].split(' -> ')[1]=='kStateAwake')])

    # Extract ScreenSaver information
    time_screensaver = []
    #predicate = 'process == "studentd" and '
    #predicate += 'eventMessage contains "ScreenSaver" and eventMessage contains "isRunning"'
    #for o in collect_log_information(timewindow, predicate):
    #    time_screensaver.append([o[:23], 1 - int(o[-1])])
    predicate = 'subsystem == "com.apple.loginwindow.logging" and '
    predicate += 'eventMessage contains "screenSaverIsRunning"'
    for o in collect_log_information(timewindow, predicate):
        time_screensaver.append([o[:23], 1 - int(o[-1])])

    # Create Login DataFrame
    df_logins = pd.DataFrame(time_logins, columns=['TimeStamp', 'Unlocked'])
    df_logins.TimeStamp = pd.to_datetime(df_logins.TimeStamp)
    df_logins = df_logins.set_index('TimeStamp')
    df_logins = df_logins[df_logins.Unlocked.diff().fillna(1)!=0]

    # Create Lid DataFrame
    df_lid = pd.DataFrame(time_lid, columns=['TimeStamp', 'stat1', 'stat2'])
    df_lid.TimeStamp = pd.to_datetime(df_lid.TimeStamp)
    df_lid = df_lid.set_index('TimeStamp')

    # Compute Lid status
    df_lid['LidOpen'] = np.array(df_lid.stat2 * df_lid.stat1 == 0).astype('int')
    df_lid = df_lid.drop(['stat1', 'stat2'], axis=1)

    # Create Sleep DataFrame
    df_sleep = pd.DataFrame(time_sleep, columns=['TimeStamp', 'WakeState'])
    df_sleep.TimeStamp = pd.to_datetime(df_sleep.TimeStamp)
    df_sleep = df_sleep.set_index('TimeStamp')

    # Create ScreenSaver DataFrame
    df_screensaver = pd.DataFrame(time_screensaver, columns=['TimeStamp', 'ScreenSaverOff'])
    df_screensaver.TimeStamp = pd.to_datetime(df_screensaver.TimeStamp)
    df_screensaver.loc[(df_screensaver['ScreenSaverOff'] == 0), 'TimeStamp'] -= pd.Timedelta(screensaver_offset)
    df_screensaver = df_screensaver.set_index('TimeStamp')
    df_screensaver = df_screensaver[df_screensaver.ScreenSaverOff.diff().fillna(1)!=0]

    # Compute time borders of search
    time_start = pd.Timestamp.now() - pd.Timedelta(timewindow)
    time_start = time_start.to_period("D").to_timestamp(how="start").round('S')
    time_now = pd.Timestamp.now().round('S')
    time_end = pd.Timestamp.now().to_period("D").to_timestamp(how="end").round('S')
    new_borders = pd.to_datetime([time_start, time_now, time_end])

    # Add time borders of search as empty strings
    df_logins = df_logins.append(
        pd.DataFrame(0, columns=['Unlocked'], index=new_borders)).sort_index()
    df_lid = df_lid.append(
        pd.DataFrame(0, columns=['LidOpen'], index=new_borders)).sort_index()
    df_sleep = df_sleep.append(
        pd.DataFrame(0, columns=['WakeState'], index=new_borders)).sort_index()
    df_screensaver = df_screensaver.append(
        pd.DataFrame(0, columns=['ScreenSaverOff'], index=new_borders)).sort_index()

    # Store individual logfiles in dictionary
    loginfos = {'logins': df_logins,
                'lid': df_lid,
                'sleep': df_sleep,
                'screensaver': df_screensaver}

    return loginfos


def unite_information(loginfos, sample_rate='60S', dayshift='5h'):

    # Resample dataframes to requested sampling rate
    df_logins = loginfos['logins'].resample(sample_rate).ffill().bfill()
    df_lid = loginfos['lid'].resample(sample_rate).ffill().bfill()
    df_sleep = loginfos['sleep'].resample(sample_rate).ffill().bfill()
    df_screensaver = loginfos['screensaver'].resample(sample_rate).ffill().bfill()

    # Combine logfile dataframes into one
    df = pd.DataFrame([df_lid.LidOpen,
                       df_logins.Unlocked,
                       df_sleep.WakeState,
                       df_screensaver.ScreenSaverOff]).T

    # Compute 'active' feature
    #df['Active'] = (df.LidOpen + df.Unlocked)>=1
    #df['Active'] = df['Active'] * df.WakeState * df.ScreenSaverOff
    df['Active'] = df.prod(axis=1)

    # Extract date information
    df['weeknumber'] = df.index.weekofyear
    df['dayname'] = df.index.day_name()
    df['daynumber'] = df.index.day
    df['weeknumber'] = df.index.dayofweek
    df['date'] = (df.index - pd.Timedelta(dayshift)).strftime('%Y-%m-%d')
    df['timestamp'] = df.index.copy()
    df.index = df.index.strftime('%H:%M')
    
    # Replace zero values with NaNs
    df = df.replace(0, np.NaN)

    # Remove date without any recordings
    df_records = df.groupby('date').sum()[['LidOpen', 'ScreenSaverOff', 'Active', 'Unlocked', 'WakeState']]
    date_to_drop = df_records[df_records.sum(axis=1)==0].index
    df = df[~df.date.isin(date_to_drop)]

    return df


# Current Work Time
def get_worktime_today(timewindow='3d'):

    # Read logfiles to extract relevant information
    loginfos = read_log_files(timewindow=timewindow)

    # Unite information in a dataframe
    df = unite_information(loginfos)

    # Restrict dataframe to today
    date_today = df.date.unique()[-1]
    df_today = df[df.date==date_today]

    # Compute total work time today
    target = df_today['Active']
    delta_start = df_today.timestamp[target.fillna(0).diff()>0]
    delta_stop = df_today.timestamp[target.fillna(0).diff()<0]

    # Correct if end of day is missing
    if len(delta_stop)>len(delta_start):
        start_of_day = delta_start[0].to_period("D").to_timestamp(how="start")
        delta_start['00:00'] = start_of_day.strftime('%Y-%m-%d %H:%M:%S')
    if len(delta_stop)<len(delta_start):
        end_of_day = delta_stop[0].to_period("D").to_timestamp(how="end")
        delta_stop['23:59'] = end_of_day.strftime('%Y-%m-%d %H:%M:%S')

    deltas = pd.to_timedelta(delta_stop.values - delta_start.values)

    return pd.Series(deltas).sum()


def report_worktime():

    # Extract current work time
    time_worked = get_worktime_today()

    # Save timestamp into string
    txt_out = str(time_worked)[-8:-3]

    # Specify different icons to use
    icon_start = 'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAAZ0lEQVQ4jeXQsQmAQBQD0CdYW7iAG7qNm7iGK1gIdjrD2dzBFSpnJ5rm88PPTwhvQ4MRXamgjrPDgBYTlsiHG22VP1gwY0WfCavSJOn4keAMf+9gj87bU9eEEPeQ8cUd5An2C4Ov4gDIJRmeUPVagQAAAABJRU5ErkJggg=='
    icon_work = 'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAAcUlEQVQ4je3SPQqDQBDF8R8ewV7BgwYtxJt4u0A6cStt/Co2mCkCKfLgMTA7839TLL+kAQnLjRP6HCCh/CCoxJx7WALXHrNFYCmrP+AE1Fu9+wO7obqCRnSB4G7bOfRCEwA0eF4bLabMqe884REI/KJWwGUqilB/fg8AAAAASUVORK5CYII='
    icon_end = 'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAApElEQVQ4jcXSIQoCQRTG8d8ubhZEjAbTZovRG3gBmwewegWL0RN4GYvdImgRjaaNaxlhWVYdFPGDgcf73v+bNzD8WONwPtIEF1xDDWUsPA3wCEOcMYsNmOOIvNLLcYoJWOCAQYPXxx7LJjDBCjv0XlzQwRZrpI9mhk1Yr/1uxTBTBiZroYsimLfKYP29SW2mCOxToIzxUl/q/wFJpY7+pg3sH3UHCfMenVq4Fs0AAAAASUVORK5CYII='
    icon_home = 'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAAnUlEQVQ4jb3TSwrCMBgE4O8yXeveF3oZwWsoihvPWV0VrWBXdWGFKE1bnwMh+TP/TBKY8ANMkKF8GhnGbeIh9phFuBTTJnEaEXc2eRs7nLGu6vDtqv0cq4AvglqO/pNpGawLJLgEXILjvWGOAxYRg7JmLrAMT+zh9MINHrCtxJuIwarilzXc9zDQnoORW9CiOWgyGekYorEP/sL/cQWw8joTqoKrxAAAAABJRU5ErkJggg=='

    # Decide which color scheem to use for the amount of time worked
    cRED='1;31'
    cGREEN='1;32'
    cYELLOW='1;33'
    cBLUE='1;36'

    total_min = time_worked.total_seconds() / 60
    if total_min < 4.1 * 60:
        color = cBLUE
        icon = icon_start
    elif total_min < 8.2 * 60:
        color = cGREEN
        icon = icon_work
    elif total_min < 9 * 60:
        color = cYELLOW
        icon = icon_end
    else:
        color = cRED
        icon = icon_home

    # Adapt color of output accordingly
    output = "\033[%sm%s\033[0m | templateImage='%s'" % (
        color, txt_out, icon)

    print(output)


def plot_daily_stats(df, date, filename, plot_restrictions=['08:00', '18:00']):

    # Categories to plot
    categories = ['Active', 'LidOpen', 'ScreenSaverOff', 'Unlocked', 'WakeState']

    # Plot lines (separated by category and color coded by date)
    fig, ax = plt.subplots(figsize=(12, 4))

    # Select date from dataframe and plot information
    group = df[df.date==date]
    for j, category in enumerate(categories):
        ds = -1 * group[category]  - 0.125 * j
        lw = 3 if j==0 else 1
        ls = '-' if j==0 else '--'
        ds.plot(linewidth=lw, ax=ax, legend=True, linestyle=ls)

    # Adjust y-axis limit
    ax.set_ylim(np.array(ax.get_ylim()) - np.array((0.1, -0.1)))

    # Add opening hour indications
    xid_open = np.argwhere(group.index=='09:00')[0]
    xid_close = np.argwhere(group.index=='17:00')[0]
    y_lim = ax.get_ylim()
    ax.vlines(xid_open, *y_lim, colors='k', linestyle=':', linewidth=2)
    ax.vlines(xid_close, *y_lim, colors='k', linestyle=':', linewidth=2)
    plt.ylim(y_lim)

    # Tweak tick label appearances
    tick_id = np.linspace(0, len(group), num=24, endpoint=False, dtype='int')
    tick_labels = group.index[tick_id]
    ax.set_xticks(tick_id)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticks([])

    # Reset xlimits to 'plot_restrictions' values, if 'None', nothing will be done
    entries_count = ~group['Active'].isna()
    if plot_restrictions and entries_count.sum():
        entries_count = ~group['Active'].isna()
        idx = np.argwhere(entries_count.to_numpy())
        idx_min = idx.min() - 10
        idx_max = idx.max() + 10

        idx_min_start = np.argmax(entries_count.index==plot_restrictions[0])
        idx_max_end = np.argmax(entries_count.index==plot_restrictions[1])

        if idx_min > idx_min_start:
            idx_min = idx_min_start
        if idx_max < idx_max_end:
            idx_max = idx_max_end

        ax.set_xlim(idx_min, idx_max)

    # Plot legend on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width * 0.8, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.55, -0.43), ncol=5, fontsize=15.5)
    
    # Compute total work time
    target = group['Active']
    delta_start = group.timestamp[target.fillna(0).diff()>0]
    delta_stop = group.timestamp[target.fillna(0).diff()<0]

    # Correct if end of day is missing
    if len(delta_stop)>len(delta_start):
        start_of_day = delta_start[0].to_period("D").to_timestamp(how="start")
        delta_start['00:00'] = start_of_day.strftime('%Y-%m-%d %H:%M:%S')
    if len(delta_stop)<len(delta_start):
        end_of_day = delta_stop[0].to_period("D").to_timestamp(how="end")
        delta_stop['23:59'] = end_of_day.strftime('%Y-%m-%d %H:%M:%S')

    deltas = pd.to_timedelta(delta_stop.values - delta_start.values)
    deltas_total = str(pd.Series(deltas).sum())[-8:]
    
    # Add Time Table
    text_box = "Start - Delta\n"
    for start, delta in zip(delta_start.index, deltas):
        text_box += '%s - %s\n' % (start, str(delta)[-8:])
    text_box += '\nTotal: {:>9}'.format(deltas_total)
    props = dict(boxstyle='round', facecolor='gray', alpha=0.1)
    ax.text(1.03, 0.97, text_box, transform=ax.transAxes, fontsize=15,
            verticalalignment='top', bbox=props, family='monospace')

    # Add title
    week_day = group.timestamp[0].day_name()[:3]
    plt.title('Date: %s (%s) - Total: %s' % (date, week_day, deltas_total));

    # Correct figure layout to tight
    plt.tight_layout(rect=(-0.05, -0.03, 0.85, 1.03))

    # Save figure
    plt.savefig(filename)
    
    # Close figure
    plt.close()


def create_datily_stats(out_path, days_back=7, dayshift='5h'):

    # Collect list of recorded days
    recorded_stats = glob(os.path.join(out_path, 'day_*'))
    recorded_stats = [d[d.find('day_20')+4:-4] for d in recorded_stats]

    # Get missing dates of last 7 stats
    days_to_check = []
    for i in range(days_back):
        day_shift = '%sd' % (i + 1)
        today = pd.Timestamp.now() - pd.Timedelta(dayshift)
        check_date = today - pd.Timedelta(day_shift)
        check_date = check_date.strftime('%Y-%m-%d')
        if check_date not in recorded_stats:
            days_to_check.append(check_date)

    # Check if states for missing days need to be created
    if len(days_to_check):

        # Read logfiles of last few days to extract relevant information
        loginfos = read_log_files(timewindow='%dd' % (days_back + 2))

        # Unite information in a dataframe
        df = unite_information(loginfos)

        # Create stats figures
        for date in days_to_check:
            filename = os.path.join(out_path, 'day_%s.png' % date)
            plot_daily_stats(df, date, filename)


def plot_stats_today(out_path, timewindow='3d', show_plot=True):

    # Read logfiles to extract relevant information
    loginfos = read_log_files(timewindow=timewindow)

    # Unite information in a dataframe
    df = unite_information(loginfos)

    # Select today
    date_today = df.date.unique()[-1]
    df_today = df[df.date==date_today]

    # Plot daily stats for last few days
    filename = os.path.join(out_path, 'current_day.png')
    plot_daily_stats(df_today, date_today, filename)

    # Show today overview
    if show_plot:
        res = os.popen('open %s' % filename)


def plot_overview(df, filename, week_id, plot_restrictions=['08:00', '18:00']):

    # Categories to plot
    categories = ['Active', 'LidOpen', 'ScreenSaverOff', 'Unlocked', 'WakeState']

    # Plot lines (separated by category and color coded by date)
    n_rows = df.date.nunique()
    fig, ax = plt.subplots(figsize=(13, 2 + n_rows * (5 / 7)))
    n_date = len(df.groupby('date'))
    color_palette = sns.color_palette("bright", n_date)
    for i, (date, group) in enumerate(df.groupby('date')):

        for j, category in enumerate(categories):
            ds = -1 * group[category] - i  - 0.125 * j
            lw = 3 if j==0 else 1
            ls = '-' if j==0 else '--'
            ds.plot(linewidth=lw, ax=ax, legend=False, c=color_palette[i], linestyle=ls)

    # Add opening hour indications
    xid_open = np.argwhere(df.index=='09:00')[0]
    xid_close = np.argwhere(df.index=='17:00')[0]
    y_lim = ax.get_ylim()
    ax.vlines(xid_open, *y_lim, colors='k', linestyle=':', linewidth=2)
    ax.vlines(xid_close, *y_lim, colors='k', linestyle=':', linewidth=2)
    plt.ylim(y_lim)

    # Tweak tick label appearances
    day_max = df.groupby('date').count()['timestamp'].idxmax()
    df_date = df[df['date']==day_max]
    tick_id = np.linspace(0, len(df_date), num=24, endpoint=False, dtype='int')
    tick_labels = df_date.index[tick_id]
    ax.set_xticks(tick_id)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticks([])
    
    # Reset xlimits to 'plot_restrictions' values, if 'None', nothing will be done
    if plot_restrictions:
        entries_count = (~df['Active'].isna()).groupby(df.index).sum()
        timeticks_order = df.index.drop_duplicates(keep='first')
        idx = np.argwhere(entries_count[timeticks_order].to_numpy())
        if len(idx)!=0:
            idx_min = idx.min()
            idx_max = idx.max()
        else:
            idx_min = 0
            idx_max = len(entries_count)

        idx_min_start = np.argmax(entries_count[timeticks_order].index==plot_restrictions[0])
        idx_max_end = np.argmax(entries_count[timeticks_order].index==plot_restrictions[1])

        if idx_min > idx_min_start:
            idx_min = idx_min_start
        if idx_max < idx_max_end:
            idx_max = idx_max_end

        ax.set_xlim(idx_min, idx_max)
    
    # Plot new legend
    text_hight = np.linspace(y_lim[1] - 0.4, y_lim[0] + 0.2, num=n_date * 2, endpoint=True)
    color_palette = sns.color_palette("bright", n_date)
    for i, (date, group) in enumerate(df.groupby('date')):
        day_name = pd.to_datetime(date).day_name()[:3]
        legend_text = '%s - %s' % (group.date.iloc[0], day_name)
        ax.text(ax.get_xlim()[1] * 1.01, text_hight[::2][i], legend_text,
                fontdict={'c': color_palette[i], 'fontsize': 16})
        
        # Compute time deltas of activity
        target = group['Active']

        delta_start = group.timestamp[target.fillna(0).diff()>0]
        delta_stop = group.timestamp[target.fillna(0).diff()<0]

        # Correct if end of day is missing
        if len(delta_stop)>len(delta_start):
            start_of_day = delta_start[0].to_period("D").to_timestamp(how="start")
            delta_start['00:00'] = start_of_day.strftime('%Y-%m-%d %H:%M:%S')
        if len(delta_stop)<len(delta_start):
            end_of_day = delta_stop[0].to_period("D").to_timestamp(how="end")
            delta_stop['23:59'] = end_of_day.strftime('%Y-%m-%d %H:%M:%S')

        deltas = pd.to_timedelta(delta_stop.values - delta_start.values)
        deltas_total = str(pd.Series(deltas).sum())[-8:]
        ax.text(ax.get_xlim()[1] * 1.03, text_hight[1::2][i] + 0.1, deltas_total,
                fontdict={'c': 'black', 'fontsize': 16})

    # Add title
    category_txt = ['(%d) %s' % (i + 1, c) for i, c in enumerate(categories)]
    plt.title('Week %s:' % week_id + ' '.join(category_txt));
    
    # Correct figure layout to tight
    plt.tight_layout()

    # Save figure
    plt.savefig(filename)

    # Close figure
    plt.close()


def plot_prev_week(out_path, days_back=14, dayshift='5h'):

    # Collect list of recorded weeks
    recorded_weeks = glob(os.path.join(out_path, 'week_*'))
    recorded_weeks = [d[d.find('week_20')+5:-4] for d in recorded_weeks]

    # Get week ID of last week
    today = pd.Timestamp.now() - pd.Timedelta(dayshift)
    last_week = today - pd.Timedelta('5d') # only 5d to account for weekend
    week_id = '%d_%02d' % (last_week.year, last_week.weekofyear)

    # Create overview for last week if not yet recorded
    if week_id not in recorded_weeks:

        # Read logfiles of last few days to extract relevant information
        loginfos = read_log_files(timewindow='%dd' % (days_back + 2))

        # Unite information in a dataframe
        df = unite_information(loginfos)

        # Only keep data from last week (+2d to correct for weekend)
        week_number = df.timestamp - pd.Timedelta(dayshift) + pd.Timedelta('2d')
        df['week'] = week_number.dt.weekofyear 
        df_week = df[df.week==last_week.weekofyear]

        # Plot Overview figure for previous week
        filename = os.path.join(out_path, 'week_%s.png' % week_id)
        plot_overview(df_week, filename, week_id)


def plot_current_week(out_path, days_back=7, dayshift='5h', show_plot=True):

    # Get week ID of current week
    today = pd.Timestamp.now() - pd.Timedelta(dayshift)
    this_week = today + pd.Timedelta('2d') # +2d to account for weekend
    week_id = '%d_%02d' % (this_week.year, this_week.weekofyear)

    # Read logfiles of current week
    loginfos = read_log_files(timewindow='%dd' % (days_back + 2))

    # Unite information in a dataframe
    df = unite_information(loginfos)

    # Only keep data from current week (+2d to correct for weekend)
    week_number = df.timestamp - pd.Timedelta(dayshift) + pd.Timedelta('2d')
    df['week'] = week_number.dt.weekofyear 
    df_week = df[df.week==this_week.weekofyear]

    # Plot Overview figure for current week
    filename = os.path.join(out_path, 'current_week.png')
    plot_overview(df_week, filename, week_id)

    # Show today overview
    if show_plot:
        res = os.popen('open %s' % filename)


if __name__ == '__main__':

    # Create folder 'screentime_log' in home folder if not exist
    home_path = os.path.expanduser("~")
    out_path = os.path.join(home_path, 'screentime_log')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Create current figures if requested
    arguments = sys.argv
    if len(arguments) > 1:

        # Option to create figures for current day or week
        if str(arguments[1])=='stats':
            plot_stats_today(out_path)
        elif str(arguments[1])=='week':
            plot_current_week(out_path)

    else:

        # Create daily stats figure for missing dates
        create_datily_stats(out_path)

        # Create overview figure for previous week
        plot_prev_week(out_path)

        # Report current worktime
        report_worktime()

        # Print options
        bitbar_dir = pathlib.Path(__file__).parent.absolute()
        file_path = os.path.join(bitbar_dir, __file__)
        cmd_template = 'bash=/anaconda3/bin/python param1={0} param2={1} terminal=false'
        print('---')
        print('Stats Today | %s' % cmd_template.format(file_path, 'stats'))
        print('Overview Week  | %s' % cmd_template.format(file_path, 'week'))
