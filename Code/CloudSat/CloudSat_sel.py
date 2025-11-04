# This program is to select LW/SW data from CloudSat data

#################################
# 1. Import packages
#################################
import h5py
import numpy as np
import xarray as xr

from scipy.ndimage import convolve1d

#################################
# 2. Import data
#################################
# CloudSat data
with xr.open_dataset("/work/DATA/Satellite/CloudSat/CloudSat_filtered.nc", chunks={}) as f:
    f = f.sel(lat=slice(-10,10))

    coords = f.coords

    # keep operations in xarray for efficiency
    qlw = f['qlw'] - f['qlw'].mean(dim={'time', "lon"})
    qsw = f['qsw'] - f['qsw'].mean(dim={'time', "lon"})

    qlw = qlw.mean(dim='lat', skipna=True).values
    qsw = qsw.mean(dim='lat', skipna=True).values

nt, nz, nx = qlw.shape

point_50lon = np.argmin(np.abs(coords["lon"].values - 50))
point_10lon = np.argmin(np.abs(coords["lon"].values - 10))

################################
# 3. Select data
#################################
# k_domains = [(2*i-1, 2*i+1) for i in range(1, 7)]
k_domains=[(1, 4)]
for i, k_domain in enumerate(k_domains):
    strt = k_domain[0]
    end = k_domain[1]

    with h5py.File(f"/home/b11209013/2025_Research/CloudSat/Files/olr_mjo_k_{strt}_{end}.h5", "r") as f:
        kw_times = np.array(f.get("max_times"), dtype=int)
        kw_lons = np.array(f.get("max_lons"), dtype=int)

    n_event = len(kw_times)

    qlw_sel = np.array([
        qlw[t,:,l-point_50lon:l+point_50lon+1]
        for t, l in zip(kw_times, kw_lons)
    ])
    qsw_sel = np.array([
        qsw[t,:,l-point_50lon:l+point_50lon+1]
        for t, l in zip(kw_times, kw_lons)
    ])

    with h5py.File(f"/work/b11209013/2025_Research/CloudSat/k_window/mjo_{strt}_{end}.h5", "w") as f:
        f.create_dataset("qlw", data=qlw_sel)
        f.create_dataset("qsw", data=qsw_sel)