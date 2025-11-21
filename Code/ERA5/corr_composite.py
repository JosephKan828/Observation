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

# accept parameter from the system
k_domains = (int(sys.argv[1]), int(sys.argv[2]))
dis_type  = str(sys.argv[3])

#####################
# 2. load data
#####################

# load ERA5 regression coefficient
PATH_INPUT = "/home/b11209013/2025_Research/Obs/Files/ERA5/reg/"

FILES = glob(f"{PATH_INPUT}prec_{dis_type}_k_{k_domains[0]}_{k_domains[1]}_lon=*.h5")

t_reg = []; q_reg = []

for f in FILES:
    with h5py.File(f, "r") as h:
        t_reg.append(np.array(h.get("t_reg")))
        q_reg.append(np.array(h.get("q_reg")))

t_reg = np.nanmean(np.array(t_reg), axis=0)
q_reg = np.nanmean(np.array(q_reg), axis=0)

t_max = np.max(np.abs(t_reg))
q_max = np.max(np.abs(q_reg))

# load corresponding wave regression coefficient

with h5py.File(f"/home/b11209013/2025_Research/Obs/Files/IMERG/Corr/corr_{dis_type}_k_{k_domains[0]}_{k_domains[1]}_composite.h5", "r") as f:
    qlw_reg = np.array(f.get("qlw_reg"))
    qsw_reg = np.array(f.get("qsw_reg"))

######################
# Plot figure
######################

lev = np.linspace(1000, 100, 37)
lon = np.linspace(-180, 180, 576)

t_level = np.linspace(-t_max, t_max, 21); t_level = t_level[~np.isclose(t_level, 0)]
q_level = np.linspace(-q_max, q_max, 11); q_level = q_level[~np.isclose(q_level, 0)]

plt.rcParams["font.family"] = "DejaVu Sans"
# Original
plt.figure(figsize=(15, 6))
lw_cf = plt.pcolormesh(
    lon, lev, qlw_reg,
    cmap="RdBu_r",
    norm=TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5)
    )
t_lw = plt.contour(
    lon, lev, t_reg,
    colors="k", linewidths=1, levels=t_level
)
q_lw = plt.contour(
    lon, lev, q_reg,
    colors="seagreen", linewidths=1, levels=q_level
)
plt.xlim(-100, 100)
plt.axvline(0, color="k", linewidth=3, linestyle="--")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Longitude (degree)", fontsize=18)
plt.ylabel("Level (hPa)", fontsize=18)
plt.title(f"Regression Coefficient between LW and {dis_type.upper()} Prec (composite)", fontsize=20)
plt.clabel(t_lw, inline=True, fontsize=10)
plt.clabel(q_lw, inline=True, fontsize=10)
plt.gca().invert_yaxis()
cb = plt.colorbar(lw_cf)
cb.ax.set_ylabel("Regression coefficient", fontsize=18)
cb.ax.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/ERA5_reg/qlw_{dis_type}_k_{k_domains[0]}_{k_domains[1]}_compsite.png", dpi=300)
plt.close()

plt.figure(figsize=(15, 6))
sw_cf = plt.pcolormesh(
    lon, lev, qsw_reg,
    cmap="RdBu_r",
    norm=TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5)
    )
t_sw = plt.contour(
    lon, lev, t_reg,
    colors="black", linewidths=1, levels=t_level
)
q_sw = plt.contour(
    lon, lev, q_reg,
    colors="seagreen", linewidths=1, levels=q_level
)
plt.xlim(-100, 100)
plt.axvline(0, color="k", linewidth=3, linestyle="--")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Longitude (degree)", fontsize=18)
plt.ylabel("Level (hPa)", fontsize=18)
plt.title(f"Regression Coefficient between SW and {dis_type.upper()} Prec (composite)", fontsize=20)
plt.clabel(t_sw, inline=True, fontsize=10)
plt.clabel(q_sw, inline=True, fontsize=10)
plt.gca().invert_yaxis()
cb = plt.colorbar(sw_cf)
cb.ax.set_ylabel("Regression coefficient", fontsize=18)
cb.ax.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig(f"/home/b11209013/2025_Research/Obs/Figure/ERA5_reg/qsw_{dis_type}_k_{k_domains[0]}_{k_domains[1]}_compsite.png", dpi=300)
plt.close()

#####################
# 3. save composited correlation
#####################
with h5py.File(f"{PATH_INPUT}corr_{dis_type}_k_{k_domains[0]}_{k_domains[1]}_composite.h5", "w") as h:

    h.create_dataset("t_reg", data=t_reg)
    h.create_dataset("q_reg", data=q_reg)
