import numpy as np;
import xarray as xr;

fpath = "/work/DATA/Satellite/CloudSat/CloudSat_gridded.nc"
ds = xr.open_dataset(fpath);

ds["qlw"] = ds["qlw"].where((ds["qlw"]>=10) | (ds["qlw"] <= -10));
ds["qsw"] = ds["qsw"].where((ds["qsw"]>=10) | (ds["qsw"] <= -10));

ds.to_netcdf("/data92/b11209013/CloudSat.nc");
