#======================================================
# This program is to extract the subdomain of ERA5 data
#======================================================

##################
# Import package
##################

import xarray as xr

##################
# Load data
##################

with xr.open_dataset("/work/b11209013/2024_Research/ERA5/t/t_Itp.nc") as ds:
    ds = ds.sel(
        lat=slice(-10, 10),
        time=slice("2006-01-01", "2017-12-31")
    ).mean("lat")

    ds.to_netcdf("/work/b11209013/2024_Research/ERA5/t/t_sub.nc")

with xr.open_dataset("/work/b11209013/2024_Research/ERA5/q/q_Itp.nc") as ds:
    ds = ds.sel(
        lat=slice(-10, 10),
        time=slice("2006-01-01", "2017-12-31")
    ).mean("lat")

    ds.to_netcdf("/work/b11209013/2024_Research/ERA5/q/q_sub.nc")


