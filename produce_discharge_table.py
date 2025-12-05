import glob

import numpy as np
import pandas as pd

from _config import *
from _tools import netcdf_to_dataframe


def interpolate(s: pd.Series) -> pd.Series:
    """
    Interpolate daily discharge with rules:
      - >10-day gaps: do NOT fill.
      - <=3-day gaps: cubic polynomial (spline-like) interpolation.
      - 4–10 day gaps: time-based interpolation + light local smoothing.
    Returns float32 to match your dataframe.
    """
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("Series must have a DatetimeIndex")

    # Work on float64, keep original order
    x = s.sort_index().astype("float64")
    if x.isna().all():
        return x.astype("float32")

    isnan = x.isna()

    # --- helper to compute NaN run lengths (per position) ---
    def _nan_run_lengths(mask: pd.Series) -> pd.Series:
        # label consecutive equal segments
        grp = (mask != mask.shift()).cumsum()
        # lengths by group id
        lengths = mask.groupby(grp).transform("size")
        # keep only for NaN runs, else 0
        lengths = lengths.where(mask, 0)
        return lengths

    run_len = _nan_run_lengths(isnan)

    # 1) SMALL GAPS (<=3): cubic polynomial interpolation, inside only
    #    - limit=3 ensures only small runs are filled
    y = x.interpolate(method="polynomial", order=3, limit=3, limit_area="inside")

    # 2) MID GAPS (4–10): time-based interpolation (captures local trend)
    #    - limit=10 keeps >10-day gaps untouched
    y = y.interpolate(method="time", limit=10, limit_area="inside")

    # 2b) LIGHT LOCAL SMOOTHING on the filled mid-size gaps only
    #     We smooth the values in those runs (plus a 1-day margin on each side)
    mid_gap_mask = (run_len >= 4) & (run_len <= 10)
    if mid_gap_mask.any():
        # Build a mask that includes one-day margins around mid-size gaps
        mid_idx = mid_gap_mask[mid_gap_mask].index
        margin = pd.Index(mid_idx.union(mid_idx - pd.Timedelta(days=1)).union(mid_idx + pd.Timedelta(days=1))).intersection(y.index)
        # Apply rolling mean only over those indices (copy to avoid touching others)
        y_sm = y.copy()
        # Use a centered 3-day window to gently smooth
        y_sm.loc[margin] = y.loc[margin].rolling(window=3, center=True, min_periods=1).mean()
        y = y_sm

    # 3) Ensure large gaps (>10) remain NaN (belt-and-suspenders)
    large_gap_mask = run_len > 10
    y[large_gap_mask] = np.nan

    # Preserve original index order and dtype
    y = y.reindex(s.index).astype("float32")
    return y


# %% Load and interpolate data
output_dir = PATH_PROJECT / 'output/data/discharge_tables'

os.makedirs(output_dir, exist_ok=True)

exlucde_catchments = pd.read_csv(PATH_PROJECT / 'per_catchment_exclude_all.csv', dtype=str)
exlucde_catchments_list = exlucde_catchments['CID'].tolist()
print(f'Excluding {len(exlucde_catchments_list)} catchments')

nc_file_list = glob.glob(str(PATH_DATA))
print(f'Found {len(nc_file_list)} files')
catchment_ids = [Path(x).stem for x in nc_file_list if Path(x).stem not in exlucde_catchments_list]
print(f'Using {len(catchment_ids)} catchments')


date_range = pd.date_range(start='2001-01-01', end='2022-12-31')
discharge_table = pd.DataFrame(index=date_range, columns=catchment_ids)


for nc_file in nc_file_list:
    catchment_id = Path(nc_file).stem
    if catchment_id in exlucde_catchments_list:
        continue
    print(f'Processing catchment {catchment_id}...')
    df = netcdf_to_dataframe(nc_file)
    # reindex to date_range
    df = df.reindex(date_range)
    df['Q[mmday]'] = interpolate(df['Q[mmday]'])
    discharge_table[catchment_id] = df['Q[mmday]']

# %% Exclusion periods
exlucde_periods = pd.read_csv(PATH_PROJECT / "per_catchment_exclude_periods.csv")
print(exlucde_periods)
# Build a mask (same shape as discharge_table)
mask = pd.DataFrame(False, index=discharge_table.index, columns=discharge_table.columns, dtype=bool)

for _, row in exlucde_periods.iterrows():
    if str(row['CID']) not in discharge_table.columns:
        continue
    mask.loc[row['start']:row['end'], str(row['CID'])] = True
    print(f'Will exclude {row["CID"]} from {row["start"]} to {row["end"]}')

# Apply all exclusions at once
print(f'Number of NaN values before exclusion: {discharge_table.isna().sum().sum()}')
discharge_table = discharge_table.mask(mask, np.nan)
print(f'Number of NaN values after exclusion: {discharge_table.isna().sum().sum()}')

# %%
output_file = output_dir / 'discharge_table_2001_2022.csv'
discharge_table.to_csv(output_file)
#%%
summary = pd.DataFrame({
    "missing_ratio": discharge_table.isna().mean(),
    "mean_Q": discharge_table.mean(),
    "min_Q": discharge_table.min(),
    "max_Q": discharge_table.max()
})
summary.head()
#%%

# group by year and keep catchemnts with complete year covereage
complete_years = {}
for year, group in discharge_table.groupby(discharge_table.index.year):
    complete_catchments = group.columns[group.notna().all()]
    complete_years[year] = complete_catchments.tolist()
    print(f"Year {year}: {len(complete_catchments)} complete catchments")

complete_years_df = pd.DataFrame.from_dict(complete_years, orient='index')
complete_years_df.to_csv(output_dir / 'complete_years.csv')

# plot histogram of complete catchments per year
import matplotlib.pyplot as plt
counts = [len(v) for v in complete_years.values()]
plt.bar(complete_years.keys(), counts)
plt.xlabel('Year')
plt.ylabel('Number of complete catchments')
plt.title('Number of complete catchments per year (2001-2022)')
plt.show()