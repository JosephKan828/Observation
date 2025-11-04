# This program is to select Kelvin wave events over different wavenumber segments
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
# 2. Import data
##################################

k_domains = (int(sys.argv[1]), int(sys.argv[2]))

FILE_INPUT = "/work/b11209013/2025_Research/IMERG/IMERG_06_17_lowpass.nc"

with xr.open_dataset(FILE_INPUT) as f:

    f = f.sel(lat=slice(-10,10), time=slice('2006-01-01','2017-12-31'))

    coords = f.coords

    prec = f["precipitation"]

    prec -= prec.mean(dim={'time', "lon"})
    prec = prec.mean(dim='lat')

nt, nx = prec.shape

################################
# 3. Apply bandpass filter
################################

# Apply 2D FFT
def fft2(data):
    data_fft = np.fft.fft(data, axis=0)
    data_fft = np.fft.ifft(data_fft, axis=1)

    return data_fft

prec_fft = fft2(prec)

# setup wavenumber and frequency
kn = np.fft.fftfreq(nx, d=1/nx)
fn = np.fft.fftfreq(nt, d=1)

kk, ff = np.meshgrid(kn, fn)

# setup filter
kel_cond = lambda k, ed: 86400.0/(2*np.pi*6.371e6)*k*np.sqrt(9.81*ed)


mask = np.where(
    (
        (kk >= k_domains[0]) & (kk <= k_domains[1]) &
        (ff >= 1/20) & (ff <= 1/2.5) &                     # KW band
        # (ff >= 1/90) & (ff <= 1/30)                          # MJO band
        (ff >= kel_cond(kk, 8)) & (ff <= kel_cond(kk, 50)) # dispersion relation for KW
    ) |
    (
        (kk <= -k_domains[0]) & (kk >= -k_domains[1]) &
        (ff <= -1/20) & (ff >= -1/2.5) &                   # KW band
        # (ff <= -1/90) & (ff >= -1/30)                        # MJO band
        (ff <= kel_cond(kk, 8)) & (ff >= kel_cond(kk, 50)) # dispersion relation for KW
    ), 1, 0
)

prec_fft_masked = prec_fft * mask

# Inverse 2D FFT
def ifft2(data_fft):
    data = np.fft.ifft(data_fft, axis=0)
    data = np.fft.fft(data, axis=1)

    return data.real

prec_kw = ifft2(prec_fft_masked)

################################
# 4. Apply threshold
################################

# Calculate threshold and significant areas
prec_kw_mean, prec_kw_std = prec_kw.mean(), prec_kw.std()

threshold = prec_kw_mean + 2.33 * prec_kw_std

sig_kw = np.where(prec_kw <= threshold, 1, 0)


# find local maximum points
max_times, max_lons = [], []

for i in range((nt-14)//14):
    strt = i*14 + 7
    end = strt + 14

    loc_max = np.where(
        (prec_kw[strt:end, :] == prec_kw[strt:end, :].min()) &
        (prec_kw[strt:end, :] < threshold)
    )

    if len(loc_max[0]) > 0:
        max_times.append(loc_max[0][0] + strt)
        max_lons.append(loc_max[1][0])

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
FILE_OUTPUT = f"/home/b11209013/2025_Research/Obs/Files/IMERG/prec_kw_k_{k_domains[0]}_{k_domains[1]}.h5" # For KW
# FILE_OUTPUT = f"/home/b11209013/2025_Research/Obs/Files/IMERG/prec_mjo_k_{k_domains[0]}_{k_domains[1]}.h5" # For MJO


with h5py.File(FILE_OUTPUT, 'w') as f:
    f.create_dataset("prec"      , data=prec_kw)
    f.create_dataset("max_times" , data=max_times_valid)
    f.create_dataset("max_lons"  , data=max_lons_valid)