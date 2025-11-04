# This program is to merge all Daily JSON file into one nc file
# Import package
import os;
import json;
import numpy as np;
import pandas as pd;
import netCDF4 as nc;

from glob import glob;
from matplotlib import pyplot as plt;

def main():
    # file path
    fpath = "/work/b11209013/2024_Research/CloudSat/";

    # collect yearly data
    for y in range(2006, 2018):
        year_collect = glob(os.path.join(fpath, f"Stage2/{y:04d}_*.json"));
        
        # load coord information
        with open(year_collect[0], "rb") as f:
            dim_data = json.load(f);

            lon = dim_data["lon"];
            lat = dim_data["lat"];
            lev = dim_data["lev"];

        Nx, Ny, Nz = len(lon), len(lat), len(lev)

        # assign time coord
        time = pd.date_range(f"{y:04d}-01-01", f"{y:04d}-12-31");
        epoch = np.datetime64("1900-01-01T00:00:00")
        time_hours = ((time.values - epoch) / np.timedelta64(1, "h")).astype("float64")
        Nt = len(time);

        qlw = np.empty((Nt, Nz, Nx, Ny), dtype=np.float32);
        qsw = np.empty((Nt, Nz, Nx, Ny), dtype=np.float32);

        for file in year_collect:
            date = int(file.split("_")[-1].split(".")[0])-1;

            with open(file, "rb") as f:
                data = json.load(f);

                qlw[date] = np.transpose(data["qlw"][:], (0, 2, 1));
                qsw[date] = np.transpose(data["qsw"][:], (0, 2, 1));
    
        with nc.Dataset(f"/work/b11209013/2024_Research/CloudSat/Stage3/{y:04d}.nc", "w") as f:
            f.title       = f"{y:04d} QLW & QSW"
            f.history     = "2025/08/25 07:51 UTC+8 create"
            f.Convections = "CF-1.8";

            f.createDimension("lon", Nx)
            f.createDimension("lat", Ny)
            f.createDimension("lev", Nz)
            f.createDimension("time", None)

            vlon = f.createVariable("lon", "f4", ("lon",))
            vlon.standard_name = "longitude"
            vlon.long_name     = "longitude"
            vlon.units         = "degrees_east"
            vlon.axis          = "X"
            vlon[:] = lon

            vlat = f.createVariable("lat", "f4", ("lat",))
            vlat.standard_name = "latitude"
            vlat.long_name     = "latitude"
            vlat.units         = "degrees_north"
            vlat.axis          = "Y"
            vlat[:] = lat

            vtime = f.createVariable("time", "f8", ("time",))
            vtime.standard_name = "time"
            vtime.long_name     = "time"
            vtime.units         = "days after 1900-01-01 00:00:00.00"
            vtime.calendar      = "standard"
            vtime.axis          = "T"
            vtime[:] = time_hours

            vqlw = f.createVariable(
                "qlw", "f4", ("time", "lev", "lat", "lon"),
            )
            vqlw.long_name = "Longwave Heating Rate"
            vqlw.long_name = "K/day"
            vqlw[...] = qlw

            vqsw = f.createVariable(
                "qsw", "f4", ("time", "lev", "lat", "lon"),
            )
            vqsw.long_name = "Shortwave Heating Rate"
            vqsw.long_name = "K/day"
            vqsw[...] = qsw


if __name__ == "__main__":
    main();
