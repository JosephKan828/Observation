# This program is to regress the time series of CloudSat and KW time series

#######################
# 1. Import packages
#######################

import sys
import h5py
import warnings
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from scipy.ndimage import convolve1d
from matplotlib.colors import TwoSlopeNorm

from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

#######################
# 2. Load data
#######################

k_domains = (int(sys.argv[1]), int(sys.argv[2]))
central_lon = int(sys.argv[3])
# Load CloudSat
with xr.open_dataset("/work/b11209013/2025_Research/CloudSat/CloudSat_sub/qlw.nc") as f:
    lon_centered = (f["lon"] - central_lon + 180) % 360 - 180
    f = f.assign_coords(lon=lon_centered).sortby("lon")

    qlw = f["qlw"].values
    lon_plot = f["lon"].values                    # <<< use this for plotting

nt, nz, nx = qlw.shape

with xr.open_dataset("/work/b11209013/2025_Research/CloudSat/CloudSat_sub/qsw.nc") as f:
    lon_centered = (f["lon"] - central_lon + 180) % 360 - 180
    f = f.assign_coords(lon=lon_centered).sortby("lon")

    qsw = f["qsw"].values

# Load KW-filtered OLR
with h5py.File(f"/home/b11209013/2025_Research/Obs/Files/IMERG/prec_kw_k_{k_domains[0]}_{k_domains[1]}.h5", "r") as f:
# with h5py.File(f"/home/b11209013/2025_Research/Obs/Files/IMERG/prec_mjo_k_{k_domains[0]}_{k_domains[1]}.h5", "r") as f:

    lon_kw = np.array(f.get("lon"))
    lon_tar = int(np.argmin(np.abs(lon_kw - central_lon)))
    lon_cen = lon_kw.size // 2

    prec = np.array(f.get("prec"))[:, lon_tar]

#######################
# 3. Regression
#######################
def regress_slope(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x_mean = x.mean()
    y_mean = y.mean()

    beta = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    return beta


# find non-nan indices
qlw_reshape = qlw.reshape(nt, nz*nx)
qsw_reshape = qsw.reshape(nt, nz*nx)

qlw_corr = np.empty((nx*nz))
qsw_corr = np.empty((nx*nz))

qlw_reg  = np.empty((nx*nz))
qsw_reg  = np.empty((nx*nz))

for i in tqdm(range(qlw_reshape.shape[1])):
    qlw_non_nan = ~np.isnan(qlw_reshape[:,i])
    qsw_non_nan = ~np.isnan(qsw_reshape[:,i])

    if ~np.any(qlw_non_nan):
        qlw_corr[i] = np.nan
        qsw_corr[i] = np.nan

        qlw_reg[i]  = np.nan
        qsw_reg[i]  = np.nan
        continue

    qlw_valid      = qlw_reshape[:,i][qlw_non_nan]
    qsw_valid      = qsw_reshape[:,i][qsw_non_nan]
    qlw_prec_valid = prec[qlw_non_nan]
    qsw_prec_valid = prec[qsw_non_nan]

    qlw_corr[i] = np.corrcoef(qlw_valid, qlw_prec_valid)[0, 1]
    qsw_corr[i] = np.corrcoef(qsw_valid, qsw_prec_valid)[0, 1]

    qlw_reg[i]  = regress_slope(qlw_prec_valid, qlw_valid)
    qsw_reg[i]  = regress_slope(qsw_prec_valid, qsw_valid)

qlw_corr = qlw_corr.reshape(nz, nx)
qsw_corr = qsw_corr.reshape(nz, nx)

qlw_reg  = qlw_reg.reshape(nz, nx)
qsw_reg  = qsw_reg.reshape(nz, nx)

#######################
# 4. save file
#######################
with h5py.File(f"/home/b11209013/2025_Research/Obs/Files/IMERG/Corr/corr_kw_k_{k_domains[0]}_{k_domains[1]}_lon={central_lon}.h5", "w") as f:
# with h5py.File(f"/home/b11209013/2025_Research/Obs/Files/IMERG/Corr/corr_mjo_k_{k_domains[0]}_{k_domains[1]}_lon={central_lon}.h5", "w") as f:
    f.create_dataset("qlw_corr", data=qlw_corr)
    f.create_dataset("qsw_corr", data=qsw_corr)
    f.create_dataset("qlw_reg" , data=qlw_reg )
    f.create_dataset("qsw_reg" , data=qsw_reg )



#######################
# 5. Plot
#######################
plt.rcParams["font.family"] = "DejaVu Sans" 
# Original

if central_lon==180:

    plt.figure(figsize=(15, 6))
    cf = plt.pcolormesh(
        lon_plot, np.linspace(1000, 100, 37), qlw_corr,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0, vmin=-0.2, vmax=0.2)
        )
    plt.axvline(0, color="k", linewidth=3, linestyle="--")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Longitude (degree)", fontsize=18)
    plt.ylabel("Level (hPa)", fontsize=18)
    plt.title("Correlation between LW and KW Prec (w/o smoothing)", fontsize=20)
    # plt.title("Correlation between LW and MJO Prec (w/o smoothing)", fontsize=20)
    plt.gca().invert_yaxis()
    cb = plt.colorbar()
    cb.ax.set_ylabel("Correlation coefficient", fontsize=18)
    cb.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/IMERG_corr/qlw_kw_k_{k_domains[0]}_{k_domains[1]}.png", dpi=300)
    # plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/IMERG_corr/qlw_mjo_k_{k_domains[0]}_{k_domains[1]}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(15, 6))
    cf = plt.pcolormesh(
        lon_plot, np.linspace(1000, 100, 37), qsw_corr,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0, vmin=-0.2, vmax=0.2)
        )
    plt.axvline(0, color="k", linewidth=3, linestyle="--")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Longitude (degree)", fontsize=18)
    plt.ylabel("Level (hPa)", fontsize=18)
    plt.title("Correlation between SW and KW Prec (w/o smoothing)", fontsize=20)
    # plt.title("Correlation between SW and MJO Prec (w/o smoothing)", fontsize=20)
    plt.gca().invert_yaxis()
    cb = plt.colorbar()
    cb.ax.set_ylabel("Correlation coefficient", fontsize=18)
    cb.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/IMERG_corr/qsw_kw_k_{k_domains[0]}_{k_domains[1]}.png", dpi=300)
    # plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/IMERG_corr/qsw_mjo_k_{k_domains[0]}_{k_domains[1]}.png", dpi=300)
    plt.close()

    # Plot regression coefficient 
    plt.figure(figsize=(15, 6))
    cf = plt.pcolormesh(
        lon_plot, np.linspace(1000, 100, 37), qlw_reg,
        cmap="BrBG",
        norm=TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5)
        )
    plt.axvline(0, color="k", linewidth=3, linestyle="--")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Longitude (degree)", fontsize=18)
    plt.ylabel("Level (hPa)", fontsize=18)
    plt.title("Regression Coefficient between LW and KW Prec (w/o smoothing)", fontsize=20)
    # plt.title("Regression Coefficient between LW and MJO Prec (w/o smoothing)", fontsize=20)
    plt.gca().invert_yaxis()
    cb = plt.colorbar()
    cb.ax.set_ylabel("Regression coefficient", fontsize=18)
    cb.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/IMERG_reg/qlw_kw_k_{k_domains[0]}_{k_domains[1]}.png", dpi=300)
    # plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/IMERG_reg/qlw_mjo_k_{k_domains[0]}_{k_domains[1]}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(15, 6))
    cf = plt.pcolormesh(
        lon_plot, np.linspace(1000, 100, 37), qsw_reg,
        cmap="BrBG",
        norm=TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5)
        )
    plt.axvline(0, color="k", linewidth=3, linestyle="--")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Longitude (degree)", fontsize=18)
    plt.ylabel("Level (hPa)", fontsize=18)
    plt.title("Regression Coefficient between SW and KW Prec (w/o smoothing)", fontsize=20)
    # plt.title("Regression Coefficient between SW and MJO Prec (w/o smoothing)", fontsize=20)
    plt.gca().invert_yaxis()
    cb = plt.colorbar()
    cb.ax.set_ylabel("Regression coefficient", fontsize=18)
    cb.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/IMERG_reg/qsw_kw_k_{k_domains[0]}_{k_domains[1]}.png", dpi=300)
    # plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/IMERG_reg/qsw_mjo_k_{k_domains[0]}_{k_domains[1]}.png", dpi=300)
    plt.close()
