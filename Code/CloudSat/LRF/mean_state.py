# This program is to compute the mean-state of temperature and moisture
import h5py;
import numpy as np;
import xarray as xr;
from matplotlib import pyplot as plt;

def main():
    # load data 
    fpath = "/work/DATA/Satellite/CloudSat/ERA5/"

    with xr.open_dataset(fpath+"t.nc") as ds:
        ds = ds.sel(lon=slice(160, 260), lat=slice(-15, 15))

        coords = ds.coords;

        t_mean = ds["t"].mean(dim={"time", "lon", "lat"});

    with xr.open_dataset(fpath+"q.nc") as ds:
        ds = ds.sel(lon=slice(160, 260), lat=slice(-15, 15))

        q_mean = ds["q"].mean(dim={"time", "lon", "lat"});

    t_mean.to_netcdf("/home/b11209013/2025_Research/CloudSat/Files/era5_tmean.h5");
    q_mean.to_netcdf("/home/b11209013/2025_Research/CloudSat/Files/era5_qmean.h5");




if __name__ == "__main__":
    main();
