# This program is to pre-process the CloudSat dataset

########################
# 1, Import packages
########################

import h5py
import xarray as xr

########################
# 2, Import data
########################

with xr.open_dataset("/work/DATA/Satellite/CloudSat/CloudSat_filtered.nc", chunks={}) as f:
    f = f.sel(lat=slice(-10,10))

    coords = f.coords

    # keep operations in xarray for efficiency
    qlw = f['qlw'] - f['qlw'].mean(dim={'time', "lon"})
    qsw = f['qsw'] - f['qsw'].mean(dim={'time', "lon"})

    qlw = qlw.mean(dim='lat', skipna=True)
    qsw = qsw.mean(dim='lat', skipna=True)

qlw.to_netcdf("/work/b11209013/2025_Research/CloudSat/CloudSat_sub/qlw.nc")
qsw.to_netcdf("/work/b11209013/2025_Research/CloudSat/CloudSat_sub/qsw.nc")