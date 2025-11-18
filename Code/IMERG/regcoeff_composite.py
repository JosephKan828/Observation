# This program is to composite correlation coefficient of different reference longitude

#####################
# 1. impoert package
#####################

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from matplotlib.colors import TwoSlopeNorm

#####################
# 2. load data
#####################

k_domains = (int(sys.argv[1]), int(sys.argv[2]))

PATH_INPUT = "/home/b11209013/2025_Research/Obs/Files/IMERG/Corr/"

FILES = glob(f"{PATH_INPUT}corr_kw_k_{k_domains[0]}_{k_domains[1]}_lon=*.h5")
# FILES = glob(f"{PATH_INPUT}corr_mjo_k_{k_domains[0]}_{k_domains[1]}_lon=*.h5")

qlw_corr = []; qsw_corr = []
qlw_reg  = []; qsw_reg  = []

for f in FILES:
    with h5py.File(f, "r") as h:
        qlw_corr.append(np.array(h.get("qlw_corr")))
        qsw_corr.append(np.array(h.get("qsw_corr")))
        qlw_reg.append(np.array(h.get("qlw_reg")))
        qsw_reg.append(np.array(h.get("qsw_reg")))

qlw_corr = np.nanmean(np.array(qlw_corr), axis=0)
qsw_corr = np.nanmean(np.array(qsw_corr), axis=0)

qlw_reg  = np.nanmean(np.array(qlw_reg) , axis=0)
qsw_reg  = np.nanmean(np.array(qsw_reg) , axis=0)

lev = np.linspace(1000, 100, 37)
lon = np.linspace(-180, 180, 576)

plt.rcParams["font.family"] = "DejaVu Sans" 
# Original
plt.figure(figsize=(15, 6))
cf = plt.pcolormesh(
    lon, np.linspace(1000, 100, 37), qlw_corr,
    cmap="RdBu_r",
    norm=TwoSlopeNorm(vcenter=0, vmin=-0.1, vmax=0.1)
    )
plt.xlim(-100, 100)
plt.axvline(0, color="k", linewidth=3, linestyle="--")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Longitude (degree)", fontsize=18)
plt.ylabel("Level (hPa)", fontsize=18)
plt.title("Correlation between LW and KW Prec (composite)", fontsize=20)
# plt.title("Correlation between LW and MJO Prec (composite)", fontsize=20)
plt.gca().invert_yaxis()
cb = plt.colorbar()
cb.ax.set_ylabel("Correlation coefficient", fontsize=18)
cb.ax.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/IMERG_corr/qlw_kw_k_{k_domains[0]}_{k_domains[1]}_compsite.png", dpi=300)
# plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/IMERG_corr/qlw_mjo_k_{k_domains[0]}_{k_domains[1]}_composite.png", dpi=300)
plt.close()

plt.figure(figsize=(15, 6))
cf = plt.pcolormesh(
    lon, np.linspace(1000, 100, 37), qsw_corr,
    cmap="RdBu_r",
    norm=TwoSlopeNorm(vcenter=0, vmin=-0.1, vmax=0.1)
    )
plt.xlim(-100, 100)
plt.axvline(0, color="k", linewidth=3, linestyle="--")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Longitude (degree)", fontsize=18)
plt.ylabel("Level (hPa)", fontsize=18)
plt.title("Correlation between SW and KW Prec (composite)", fontsize=20)
# plt.title("Correlation between SW and MJO Prec (composite)", fontsize=20)
plt.gca().invert_yaxis()
cb = plt.colorbar()
cb.ax.set_ylabel("Correlation coefficient", fontsize=18)
cb.ax.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/IMERG_corr/qsw_kw_k_{k_domains[0]}_{k_domains[1]}_compsite.png", dpi=300)
# plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/IMERG_corr/qsw_mjo_k_{k_domains[0]}_{k_domains[1]}_composite.png", dpi=300)
plt.close()


plt.figure(figsize=(15, 6))
cf = plt.pcolormesh(
    lon, np.linspace(1000, 100, 37), qlw_reg,
    cmap="BrBG",
    norm=TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5)
    )
plt.xlim(-100, 100)
plt.axvline(0, color="k", linewidth=3, linestyle="--")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Longitude (degree)", fontsize=18)
plt.ylabel("Level (hPa)", fontsize=18)
plt.title("Regression Coefficient between LW and KW Prec (composite)", fontsize=20)
# plt.title("Regression Coefficient between LW and MJO Prec (composite)", fontsize=20)
plt.gca().invert_yaxis()
cb = plt.colorbar()
cb.ax.set_ylabel("Regression coefficient", fontsize=18)
cb.ax.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/IMERG_reg/qlw_kw_k_{k_domains[0]}_{k_domains[1]}_compsite.png", dpi=300)
# plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/IMERG_reg/qlw_mjo_k_{k_domains[0]}_{k_domains[1]}_composite.png", dpi=300)
plt.close()

plt.figure(figsize=(15, 6))
cf = plt.pcolormesh(
    lon, np.linspace(1000, 100, 37), qsw_reg,
    cmap="BrBG",
    norm=TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5)
    )
plt.xlim(-100, 100)
plt.axvline(0, color="k", linewidth=3, linestyle="--")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Longitude (degree)", fontsize=18)
plt.ylabel("Level (hPa)", fontsize=18)
plt.title("Regression Coefficient between SW and KW Prec (composite)", fontsize=20)
# plt.title("Regression Coefficient between SW and MJO Prec (composite)", fontsize=20)
plt.gca().invert_yaxis()
cb = plt.colorbar()
cb.ax.set_ylabel("Regression coefficient", fontsize=18)
cb.ax.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/IMERG_reg/qsw_kw_k_{k_domains[0]}_{k_domains[1]}_compsite.png", dpi=300)
# plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/IMERG_reg/qsw_mjo_k_{k_domains[0]}_{k_domains[1]}_composite.png", dpi=300)
plt.close()


#####################
# 3. save composited correlation
#####################
with h5py.File(f"{PATH_INPUT}corr_kw_k_{k_domains[0]}_{k_domains[1]}_composite.h5", "w") as h:
# with h5py.File(f"{PATH_INPUT}corr_mjo_k_{k_domains[0]}_{k_domains[1]}_composite.h5", "w") as h:
    h.create_dataset("qlw_corr", data=qlw_corr)
    h.create_dataset("qsw_corr", data=qsw_corr)
    h.create_dataset("qlw_reg" , data=qlw_reg )
    h.create_dataset("qsw_reg" , data=qsw_reg )
