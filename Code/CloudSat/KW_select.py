# This program is to select data with OLR anomalies
##################################
# 1. Import packages
##################################

import os
import sys
import h5py
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

##################################
# 2. Import package
##################################

FILE_INPUT = "/work/DATA/Satellite/OLR/olr_anomaly.nc"

with xr.open_dataset(FILE_INPUT) as f:

    f = f.sel(lat=slice(-10,10), time=slice('2006-01-01','2017-12-31'))

    coords = f.coords

    olr = f['olr']

    olr -= olr.mean(dim={'time', "lon"})
    olr = olr.mean(dim='lat')

nt, nx = olr.shape

k_domain = (int(sys.argv[1]), int(sys.argv[2]))

################################
# 3. Apply bandpass filter
################################

# Apply 2D FFT
def fft2(data):
    data_fft = np.fft.fft(data, axis=0)
    data_fft = np.fft.ifft(data_fft, axis=1)

    return data_fft

olr_fft = fft2(olr)

# setup wavenumber and frequency
kn = np.fft.fftfreq(nx, d=1/nx)
fn = np.fft.fftfreq(nt, d=1)

kk, ff = np.meshgrid(kn, fn)

# setup filter
kel_cond = lambda k, ed: 86400.0/(2*np.pi*6.371e6)*k*np.sqrt(9.81*ed)


mask = np.where(
    (
        (kk >= k_domain[0]) & (kk <= k_domain[1]) &
        # (ff >= 1/20) & (ff <= 1/2.5) &                     # KW band
        (ff >= 1/90) & (ff <= 1/30)                          # MJO band
        # (ff >= kel_cond(kk, 8)) & (ff <= kel_cond(kk, 50)) # dispersion relation for KW
    ) |
    (
        (kk <= -k_domain[0]) & (kk >= -k_domain[1]) &
        # (ff <= -1/20) & (ff >= -1/2.5) &                   # KW band
        (ff <= -1/90) & (ff >= -1/30)                        # MJO band
        # (ff <= kel_cond(kk, 8)) & (ff >= kel_cond(kk, 50)) # dispersion relation for KW
    ), 1, 0
)



olr_fft_masked = olr_fft * mask

# Inverse 2D FFT
def ifft2(data_fft):
    data = np.fft.ifft(data_fft, axis=0)
    data = np.fft.fft(data, axis=1)

    return data.real

olr_kw = ifft2(olr_fft_masked)

################################
# 4. Apply threshold
################################

# Calculate threshold and significant areas
olr_kw_mean, olr_kw_std = olr_kw.mean(), olr_kw.std()

threshold = olr_kw_mean - 2.33 * olr_kw_std

sig_kw = np.where(olr_kw <= threshold, 1, 0)


# find local maximum points
max_times, max_lons = [], []

for i in range((nt-14)//14):
    strt = i*14 + 7
    end = strt + 14

    loc_min = np.where(
        (olr_kw[strt:end, :] == olr_kw[strt:end, :].min()) &
        (olr_kw[strt:end, :] < threshold)
    )

    if len(loc_min[0]) > 0:
        max_times.append(loc_min[0][0] + strt)
        max_lons.append(loc_min[1][0])

max_times = np.array(max_times)
max_lons = np.array(max_lons)

valid_criteria = np.where(
    (max_lons > np.argmin(np.abs(coords["lon"].values-50.0))) &
    (max_lons < np.argmin(np.abs(coords["lon"].values-310.0)))
    )

max_times_valid = max_times[valid_criteria]
max_lons_valid = max_lons[valid_criteria]

################################
# 5. Save data
################################
# FILE_OUTPUT = f"/home/b11209013/2025_Research/CloudSat/Files/olr_kw_k_{k_domain[0]}_{k_domain[1]}.h5" # For KW
FILE_OUTPUT = f"/home/b11209013/2025_Research/CloudSat/Files/olr_mjo_k_{k_domain[0]}_{k_domain[1]}.h5" # For MJO


with h5py.File(FILE_OUTPUT, 'w') as f:
    f.create_dataset("reconstruct_olr", data=olr_kw)
    f.create_dataset("max_times"      , data=max_times_valid)
    f.create_dataset("max_lons"       , data=max_lons_valid)