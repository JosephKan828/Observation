# This program is to compute the profile of CloudSat data
##################################
# 1. Import packages
##################################
import h5py
import numpy as np
import xarray as xr

from scipy.ndimage import convolve1d
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

##################################
# 2. Import data
##################################
k_domains = [
    (2*n+1, 2*n+3) for n in range(4)
]

qlws = np.empty((len(k_domains), 37, 159))
qsws = np.empty((len(k_domains), 37, 159))

kernel_lens = np.empty(len(k_domains))

for i, k_domain in enumerate(k_domains):
    # Selected CloudSat data
    with h5py.File(f"/work/b11209013/2025_Research/CloudSat/k_window/{k_domain[0]}_{k_domain[1]}.h5", "r") as f:
        qlw = np.array(f.get("qlw"))
        qsw = np.array(f.get("qsw"))

    k_ave      = (k_domain[0] + k_domain[1]) / 2
    # kernel_lens[i] = np.round(2*np.pi*6.371*1e6*1e-3/(k_ave*4*110))/0.625

    kernel_lens[i] = np.round(2*np.pi*6.371*1e6*1e-3/(9*4*110))/0.625

    kernel     = np.ones(int(kernel_lens[i]))/ int(kernel_lens[i])

    qlws[i] = convolve1d(np.nanmean(qlw, axis=0), kernel, axis=-1)
    qsws[i] = convolve1d(np.nanmean(qsw, axis=0), kernel, axis=-1)

x_half = 159//2
k_half = [ int(k//2) for k in kernel_lens]

qlw_means = np.array([
    np.nanmean(qlws[i][:,x_half-k_half[i]:x_half+k_half[i]+1], axis=-1)
    for i in range(len(k_domains))
])
qsw_means = np.array([
    np.nanmean(qsws[i][:,x_half-k_half[i]:x_half+k_half[i]+1], axis=-1)
    for i in range(len(k_domains))
])

levs = np.linspace(1000, 100, 37)

def pressure_weights(p):
    """
    Return layer thickness weights w_i for levels p_i.
    p : 1D array of pressures (Pa or hPa) — can be ascending or descending.
    Resulting w has same length as p and is positive.
    """
    p = np.asarray(p)
    # Use gradient: robust for uneven spacing
    dp = np.abs(np.gradient(p))
    return dp

qlw_norms = qlw_means*pressure_weights(levs) / np.sum(qlw_means * pressure_weights(levs), axis=-1, keepdims=True)
qsw_norms = qsw_means*pressure_weights(levs) / np.sum(qsw_means * pressure_weights(levs), axis=-1, keepdims=True)

plt.figure(figsize=(9, 15))
for i in range(len(k_domains)):
    plt.plot(qlw_norms[i], np.linspace(1000, 100, 37), label=f"{k_domains[i][0]} - {k_domains[i][1]}")
plt.subplots_adjust(bottom=0.05, top=0.95)
plt.xticks(np.linspace(-0.06, 0.06, 5), fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(-0.07, 0.07)
plt.ylim(1000, 100)
plt.legend(fontsize=18)
plt.grid(True)
plt.xlabel("Normalized LW Heating", fontsize=18)
plt.ylabel("Pressure (hPa)", fontsize=18)
plt.title("Normalized LW Heating Profiles", fontsize=20)
plt.savefig("/home/b11209013/2025_Research/CloudSat/Figure/CloudSat_profile/Fix_window/qlw_norms_wo_9-11.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 15))
for i in range(len(k_domains)):
    plt.plot(qsw_norms[i], np.linspace(1000, 100, 37), label=f"{k_domains[i][0]} - {k_domains[i][1]}")
plt.subplots_adjust(bottom=0.05, top=0.95)
plt.xticks(np.linspace(-0.4, 0.4, 5),fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(-0.5, 0.5)
plt.ylim(1000, 100)
plt.legend(fontsize=18)
plt.grid(True)
plt.xlabel("Normalized SW Heating", fontsize=18)
plt.ylabel("Pressure (hPa)", fontsize=18)
plt.title("Normalized SW Heating Profiles", fontsize=20)
plt.savefig("/home/b11209013/2025_Research/CloudSat/Figure/CloudSat_profile/Fix_window/qsw_norms_wo_9-11.png", dpi=300)