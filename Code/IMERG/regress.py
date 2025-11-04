# This program is to regress the time series of CloudSat and KW time series

#######################
# 1. Import packages
#######################

import sys
import h5py
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from scipy.ndimage import convolve1d
from matplotlib.colors import TwoSlopeNorm

from tqdm import tqdm

#######################
# 2. Load data
#######################

k_domains = (int(sys.argv[1]), int(sys.argv[2]))

# Load CloudSat
with xr.open_dataset("/work/b11209013/2025_Research/CloudSat/CloudSat_sub/qlw.nc") as f:
    f = f.sel(lon=slice(80, 280))

    coords = f.coords
    lon = coords["lon"].values
    qlw = f["qlw"].values

nt, nz, nx = qlw.shape

with xr.open_dataset("/work/b11209013/2025_Research/CloudSat/CloudSat_sub/qsw.nc") as f:
    f = f.sel(lon=slice(80, 280))

    qsw = f["qsw"].values

# Load KW-filtered OLR
with h5py.File(f"/home/b11209013/2025_Research/Obs/Files/IMERG/prec_kw_k_{k_domains[0]}_{k_domains[1]}.h5", "r") as f:
# with h5py.File(f"/home/b11209013/2025_Research/CloudSat/Files/prec_mjo_k_{k_domains[0]}_{k_domains[1]}.h5", "r") as f:

    lon_180 = np.argmin(np.abs(lon - 180))

    prec = np.array(f.get("prec"))[:, lon_180]

#######################
# 3. Regression
#######################

# find non-nan indices
qlw_reshape = qlw.reshape(nt, nz*nx)
qsw_reshape = qsw.reshape(nt, nz*nx)

qlw_corr = np.empty((nx*nz))
qsw_corr = np.empty((nx*nz))
qlw_regr = np.empty((nx*nz))
qsw_regr = np.empty((nx*nz))

for i in tqdm(range(qlw_reshape.shape[1])):
    qlw_non_nan = ~np.isnan(qlw_reshape[:,i])
    qsw_non_nan = ~np.isnan(qsw_reshape[:,i])

    if ~np.any(qlw_non_nan):
        qlw_corr[i] = np.nan
        qsw_corr[i] = np.nan
        continue

    qlw_valid = qlw_reshape[:,i][qlw_non_nan]
    qsw_valid = qsw_reshape[:,i][qsw_non_nan]
    qlw_prec_valid = prec[qlw_non_nan]
    qsw_prec_valid = prec[qsw_non_nan]    

    qlw_valid -= np.nanmean(qlw_valid)
    qsw_valid -= np.nanmean(qsw_valid)
    qlw_prec_valid -= np.nanmean(qlw_prec_valid)
    qsw_prec_valid -= np.nanmean(qsw_prec_valid)

    qlw_corr[i] = np.nanmean(qlw_valid * qlw_prec_valid) / (np.nanstd(qlw_valid) * np.nanstd(qlw_prec_valid))
    qsw_corr[i] = np.nanmean(qsw_valid * qsw_prec_valid) / (np.nanstd(qsw_valid) * np.nanstd(qsw_prec_valid))

    qlw_regr[i] = np.nanmean(qlw_valid * qlw_prec_valid) / (np.nanvar(qsw_prec_valid))
    qsw_regr[i] = np.nanmean(qsw_valid * qsw_prec_valid) / (np.nanvar(qlw_prec_valid))

qlw_corr = qlw_corr.reshape(nz, nx)
qsw_corr = qsw_corr.reshape(nz, nx)

qlw_regr = qlw_regr.reshape(nz, nx)
qsw_regr = qsw_regr.reshape(nz, nx)

#######################
# 4. Plot
#######################
plt.rcParams["font.family"] = "DejaVu Sans" 
# Original
plt.figure(figsize=(15, 6))
cf = plt.pcolormesh(
    lon, np.linspace(1000, 100, 37), qlw_corr,
    cmap="RdBu_r",
    norm=TwoSlopeNorm(vcenter=0)
    )
plt.axvline(lon[lon_180], color="k", linewidth=3, linestyle="--")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Longitude (degree)", fontsize=18)
plt.ylabel("Level (hPa)", fontsize=18)
plt.title("Correlation between LW and KW Prec (w/o smoothing)", fontsize=20)
# plt.title("Correlation between LW and MJO OLR (w/o smoothing)", fontsize=20)
plt.gca().invert_yaxis()
cb = plt.colorbar()
cb.ax.set_ylabel("Correlation coefficient", fontsize=18)
cb.ax.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/IMERG/Correlation/Origin/qlw_kw_k_{k_domains[0]}_{k_domains[1]}.png", dpi=300)
# plt.savefig(f"/home/b11209013/2025_Research/CloudSat/Figure/CloudSat_profile/Correlation/Original/qlw_mjo_k_{k_domains[0]}_{k_domains[1]}.png", dpi=300)
plt.close()

plt.figure(figsize=(15, 6))
cf = plt.pcolormesh(
    lon, np.linspace(1000, 100, 37), qsw_corr,
    cmap="RdBu_r",
    norm=TwoSlopeNorm(vcenter=0)
    )
plt.axvline(lon[lon_180], color="k", linewidth=3, linestyle="--")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Longitude (degree)", fontsize=18)
plt.ylabel("Level (hPa)", fontsize=18)
plt.title("Correlation between SW and KW Prec (w/o smoothing)", fontsize=20)
# plt.title("Correlation between SW and MJO OLR (w/o smoothing)", fontsize=20)
plt.gca().invert_yaxis()
cb = plt.colorbar()
cb.ax.set_ylabel("Correlation coefficient", fontsize=18)
cb.ax.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/IMERG/Correlation/Origin/qsw_kw_k_{k_domains[0]}_{k_domains[1]}.png", dpi=300)
# plt.savefig(f"/home/b11209013/2025_Research/CloudSat/Figure/CloudSat_profile/Correlation/Original/qsw_mjo_k_{k_domains[0]}_{k_domains[1]}.png", dpi=300)
plt.close()