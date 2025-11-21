# =========================================================
# Regress moisture and temperature against KW precipitation
# =========================================================

##################
# Import package
##################

import sys
import h5py
import numpy as np
import xarray as xr

from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt

# accept parameters input from system
PATH_WINDOW = str(sys.argv[1]) # input windowing file name
central_lon = int(sys.argv[2]) # input longitude to be selected

####################
# Load data
####################

# Load filtered precipitation signal
PATH_WORK = "/work/b11209013/2024_Research/ERA5/"

with h5py.File(PATH_WINDOW, "r") as f:

    lon_kw  = np.array(f.get("lon"))                       # longitude axis of reconstructed data
    tar_idx = int(np.argmin(np.abs(lon_kw - central_lon))) # index of targeted longitude

    idx_cen = lon_kw.size // 2                             # index of central point

    prec    = np.array(f.get("prec"))[:,tar_idx]           # precipitation time series at given longitude

prec = prec.astype(np.float32)

split_fname = PATH_WINDOW.split("_")                       # split string with _

wave_type   = split_fname[-4]                              # identify the file is for mjo or kw
k_min       = split_fname[-2]                              # minimum of the k-window
k_max       = split_fname[-1].split(".")[0]                # maximum of the k-window

print("finish loading precipitation time series")

# Load temperature and moisture from ERA5
# load moisture
with xr.open_dataset(PATH_WORK + "q/q_sub.nc", chunks={}) as ds:

    lon_cen = (ds["lon"] - central_lon + 180) % 360 - 180
    ds = ds.assign_coords(lon=lon_cen).sortby("lon")

    lon_plot = ds["lon"].values                             # longitude for plotting 
    q        = ds["q"].values.astype(np.float32)            # moisture

nt, nz, nx = q.shape

print("finish loading moisture")

# load temperature
with xr.open_dataset(PATH_WORK + "t/t_sub.nc", chunks={}) as ds:

    lon_cen = (ds["lon"] - central_lon + 180) % 360 - 180
    ds = ds.assign_coords(lon=lon_cen).sortby("lon")

    t        = ds["t"].values.astype(np.float32)            # temperature

print("finish loading temperature")

#######################
# Regress ERA5 respective to precipitation data
#######################

def regress_slope(x, y):

    x = np.asarray(x)
    y = np.asarray(y)

    x_anom = x - x.mean()
    y_anom = y - y.mean(axis=0)

    beta   = np.sum(x_anom[:,None]*y_anom, axis=0) / np.sum(x_anom**2)

    return beta

t_reshape = t.reshape(nt, nz*nx)
q_reshape = q.reshape(nt, nz*nx)

# calculate regression coefficient between precipitation and temperature/moisture
t_reg = regress_slope(prec, t_reshape).reshape(nz, nx)
q_reg = regress_slope(prec, q_reshape).reshape(nz, nx)

print("finish regression")

###########################
# save file
###########################

with h5py.File(f"/home/b11209013/2025_Research/Obs/Files/ERA5/reg/prec_{wave_type}_k_{k_min}_{k_max}_lon={central_lon}.h5", "w") as f:
    f.create_dataset("t_reg", data=t_reg)
    f.create_dataset("q_reg", data=q_reg)

print("finish saving files")
