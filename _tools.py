import re
from datetime import datetime, timedelta

import pandas as pd
import xarray as xr


def netcdf_to_dataframe(netcdf_file):
    # Open the NetCDF file
    dataset = xr.open_dataset(netcdf_file, decode_times=False)
    start_date, end_date = get_start_and_end_date_from_netcdf(dataset)
    dataset['date'] = pd.date_range(start_date, end_date, freq='D')
    df = dataset.to_dataframe().reset_index()

    df['date'] = pd.date_range(start_date, start_date + timedelta(days=len(df) - 1))
    df.set_index('date', inplace=True)
    return df


def get_start_and_end_date_from_netcdf(netcdf_file: xr) -> tuple[datetime, datetime]:
    # Extract the 'date' variable
    date_var = netcdf_file['date']

    # Get the 'units' attribute
    units_attr = date_var.units

    # Extract the reference date from the units string
    match = re.match(r'days since (\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?)', units_attr)
    if match:
        ref_date_str = match.group(1)
        ref_date = datetime.strptime(ref_date_str, '%Y-%m-%d %H:%M:%S' if ' ' in ref_date_str else '%Y-%m-%d')
    else:
        print(f"Could not parse date units in file {netcdf_file.filepath()}")
        exit()

    # Get the number of observations
    num_observations = date_var.shape[0]

    # Calculate the start and end dates
    start_date = ref_date
    end_date = ref_date + timedelta(days=int(num_observations - 1))
    return start_date, end_date


def is_year_complete(year, df: pd.DataFrame, verbose=False) -> bool:
    theoretical_range = pd.date_range(f'{year}-01-01', f'{year}-12-31')
    message = ''
    ok = False

    try:
        df.loc[str(year)]
        ok = True
    except KeyError:
        message += f"No data for {year}."

    if ok:
        # Find missing dates
        missing_dates = theoretical_range.difference(df.loc[str(year)].index)
        if missing_dates.size > 0:
            ok = False

            message += f"Missing dates in {year}:"
            for date in missing_dates:
                message += date.strftime('%Y-%m-%d')
        else:
            message += f"No missing dates in {year}."

    if verbose:
        print(message)

    return ok
