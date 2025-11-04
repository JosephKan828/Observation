##############################################
# This program is to compute the linear response function between ERA5 and CloudSat
# Import Package
################################################
import h5py
import numpy as np
import xarray as xr

from pandas import date_range
from matplotlib import pyplot as plt

################################################
# MAIN FUNCTION
################################################

def main():
    # ==== 1. load data ==== #
    fpath = "/work/DATA/Satellite/CloudSat/"
    

    time_series = date_range("1979-01-01", "2021-12-31", freq="1D")

    str_idx = np.where(time_series.year==2006)[0][0]

    # load ERA5 Data
    with xr.open_dataset(fpath+"ERA5/q.nc", chunks={}) as ds:
        ds = ds.isel(time=slice(str_idx, str_idx+1000))
        ds = ds.sel(lat=slice(-15,15), lon=slice(160, 260))
    
        coords = ds.coords.to_dataset()
    
        q = ds["q"].values * 1000.0 # (time, lev, lat, lon), convert to g/kg

    with xr.open_dataset(fpath+"ERA5/t.nc", chunks={}) as ds:
        ds = ds.isel(time=slice(str_idx, str_idx+1000))
        ds = ds.sel(lat=slice(-15,15), lon=slice(160, 260))
    
        t = ds["t"].values # (time, lev, lat, lon)

    # load CloudSat Data
    with xr.open_dataset(fpath+"CloudSat_filtered.nc", chunks={}) as ds:
        ds = ds.isel(time=slice(None, 1000))
        ds = ds.sel(lat=slice(-15,15), lon=slice(160, 260))

        qlw = ds["qlw"].values # (time, lev, lat, lon)
        qsw = ds["qsw"].values # (time, lev, lat, lon)

    print(qlw.shape)

    # Load mean-state information
    with h5py.File()

################################################
# EXECUTE MAIN
################################################

if __name__ == "__main__":
    main();
