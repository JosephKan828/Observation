# This program is to average over convective phase of composite correlation map 
# to assess the top-heaviness of KW or MJO

#####################
# 1. impoert package
#####################

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

#####################
# 2. load data
#####################

k_domains = [
    (2*i+1, 2*i+3) for i in range(5)
]

PATH_INPUT = "/home/b11209013/2025_Research/Obs/Files/OLR/Corr/"

qlw = []; qsw = [];

for i, k in enumerate(k_domains):
    FILES = f"{PATH_INPUT}corr_kw_k_{k[0]}_{k[1]}_composite.h5"
    # FILES = glob(f"{PATH_INPUT}corr_mjo_k_{k[0]}_{k[1]}_lon=*.h5")

    with h5py.File(FILES, "r") as h:
        qlw.append(np.array(h.get("qlw_corr")))
        qsw.append(np.array(h.get("qsw_corr")))

qlw = np.array(qlw)
qsw = np.array(qsw)

levs = np.linspace(1000, 100, qlw.shape[1])
lon = np.linspace(-180, 180, qlw.shape[-1])
mid_point = np.argmin(np.abs(lon - 0))

# compute targeted window for average
k_ave = [(k[0] + k[1]) / 2 for k in k_domains]

kernel_lens = [np.round(2*np.pi*6.371*1e6*1e-3/(k_ave[i]*4*110))/0.625 for i in range(len(k_ave))]

half_kernel = [int(kernel_lens[i] // 2) for i in range(len(kernel_lens))]

qlw_means = np.array([
    np.nanmean(qlw[i][:, mid_point-half_kernel[i]:mid_point+half_kernel[i]+1], axis=-1)
    for i in range(len(half_kernel))
    ])
qsw_means = np.array([
    np.nanmean(qsw[i][:, mid_point-half_kernel[i]:mid_point+half_kernel[i]+1], axis=-1)
    for i in range(len(half_kernel))
    ])

qlw_means_max = np.nanmax(np.abs(qlw_means))
qsw_means_max = np.nanmax(np.abs(qsw_means))

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

qlw_norms = qlw_means*(pressure_weights(levs))[None,:] / np.sum(qlw_means * (pressure_weights(levs))[None,:], axis=-1, keepdims=True)
qsw_norms = qsw_means*(pressure_weights(levs))[None,:] / np.sum(qsw_means * (pressure_weights(levs))[None,:], axis=-1, keepdims=True)

qlw_norms_max = np.nanmax(np.abs(qlw_norms))
qsw_norms_max = np.nanmax(np.abs(qsw_norms))

plt.figure(figsize=(9, 15))
for i in range(len(k_domains)):
    plt.plot(qlw_means[i], np.linspace(1000, 100, 37), label=f"{k_domains[i][0]} - {k_domains[i][1]}")
plt.subplots_adjust(bottom=0.05, top=0.95)
plt.xticks(np.linspace(-round(qlw_means_max,2), round(qlw_means_max, 2), 5), fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(-qlw_means_max*1.05, qlw_means_max*1.05)
plt.ylim(1000, 100)
plt.legend(fontsize=18)
plt.grid(True)
plt.xlabel("Averaged LW Heating", fontsize=18)
plt.ylabel("Pressure (hPa)", fontsize=18)
plt.title("Averaged LW Heating Profiles", fontsize=20)
plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/OLR_corr/qlw_kw_means.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 15))
for i in range(len(k_domains)):
    plt.plot(qsw_means[i], np.linspace(1000, 100, 37), label=f"{k_domains[i][0]} - {k_domains[i][1]}")
plt.subplots_adjust(bottom=0.05, top=0.95)
plt.xticks(np.linspace(-round(qsw_means_max, 2), round(qsw_means_max, 2), 5), fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(-qsw_means_max*1.05, qsw_means_max*1.05)
plt.ylim(1000, 100)
plt.legend(fontsize=18)
plt.grid(True)
plt.xlabel("Averaged SW Heating", fontsize=18)
plt.ylabel("Pressure (hPa)", fontsize=18)
plt.title("Averaged SW Heating Profiles", fontsize=20)
plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/OLR_corr/qsw_kw_means.png", dpi=300)
plt.close()


plt.figure(figsize=(9, 15))
for i in range(len(k_domains)):
    plt.plot(qlw_norms[i], np.linspace(1000, 100, 37), label=f"{k_domains[i][0]} - {k_domains[i][1]}")
plt.subplots_adjust(bottom=0.05, top=0.95)
plt.xticks(np.linspace(-round(qlw_norms_max,2), round(qlw_norms_max, 2), 5), fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(-qlw_norms_max*1.05, qlw_norms_max*1.05)
plt.ylim(1000, 100)
plt.legend(fontsize=18)
plt.grid(True)
plt.xlabel("Normalized LW Heating", fontsize=18)
plt.ylabel("Pressure (hPa)", fontsize=18)
plt.title("Normalized LW Heating Profiles", fontsize=20)
plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/OLR_corr/qlw_kw_norms.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 15))
for i in range(len(k_domains)):
    plt.plot(qsw_norms[i], np.linspace(1000, 100, 37), label=f"{k_domains[i][0]} - {k_domains[i][1]}")
plt.subplots_adjust(bottom=0.05, top=0.95)
plt.xticks(np.linspace(-round(qsw_norms_max, 2), round(qsw_norms_max, 2), 5), fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(-qsw_norms_max*1.05, qsw_norms_max*1.05)
plt.ylim(1000, 100)
plt.legend(fontsize=18)
plt.grid(True)
plt.xlabel("Normalized SW Heating", fontsize=18)
plt.ylabel("Pressure (hPa)", fontsize=18)
plt.title("Normalized SW Heating Profiles", fontsize=20)
plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/OLR_corr/qsw_kw_norms.png", dpi=300)
plt.close()