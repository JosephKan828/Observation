# This program is to check the validity of CloudSat data
# Import data
import numpy as np;
import xarray as xr;

def main():
    # ==== 1. Load data ==== #
    fpath = "/work/DATA/Satellite/CloudSat/CloudSat_gridded.nc";

    with xr.open_dataset(fpath) as f:
        coords = f.coords;

        qlw = f["qlw"]; qsw = f["qsw"];

    

if __name__ == "__main__":
    main();
