# This program is to apply mergetime on data
import os;
import numpy as np;
import xarray as xr;

def main():
    # load file
    fpath = "/work/b11209013/2024_Research/CloudSat/Stage3/";

    ds = xr.open_mfdataset(
        os.path.join(fpath, "*.nc"),
        combine="by_coords",
        # concat_dim="time"
    );

    ds.to_netcdf(os.path.join(fpath,"CloudSat.nc"));

    

if __name__ == "__main__":
    main();