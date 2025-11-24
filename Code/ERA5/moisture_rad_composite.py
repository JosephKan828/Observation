# ===============================================================
# This program is to composite moisture and its radiative heating
# ===============================================================

# ###################
# Import Package
# ###################

import h5py
import numpy as np
import xarray as xr

from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ###################
# Load data
# ###################

# moisture radiation profile
with h5py.File("/home/b11209013/2025_Research/Obs/Files/ERA5/moisture_rad.h5", "r") as f:

    grp_q_lw = f["q_lw"]
    grp_q_sw = f["q_sw"]
    grp_t_lw = f["t_lw"]
    grp_t_sw = f["t_sw"]

    q_lw = {key: grp_q_lw[key][...] for key in grp_q_lw.keys()}
    q_sw = {key: grp_q_sw[key][...] for key in grp_q_sw.keys()}
    t_lw = {key: grp_t_lw[key][...] for key in grp_t_lw.keys()}
    t_sw = {key: grp_t_sw[key][...] for key in grp_t_sw.keys()}

# moisture and temperature profile
FILES = sorted(
    glob("/home/b11209013/2025_Research/Obs/Files/ERA5/reg/*_composite.h5"))

t_reg, q_reg = {}, {}

for i, key in enumerate(q_lw.keys()):
    with h5py.File(FILES[i], "r") as f:
        t_reg[key] = np.array(f.get("t_reg"))
        q_reg[key] = np.array(f.get("q_reg"))

# ####################
# Plot the composite
# ####################

lon = np.linspace(-180, 180, 576)
lev = np.linspace(1000, 100, 37)

tmax = np.max(np.abs(t_reg[key]))
tlevel = np.linspace(-tmax, tmax, 11)
qmax = np.max(np.abs(q_reg[key]))
qlevel = np.linspace(-qmax, qmax, 11)

t_level = [t for t in tlevel if t != 0]
q_level = [q for q in qlevel if q != 0]


for key in tqdm(q_lw.keys()):
    dis_type = key.split("_")[0]
    k_left = key.split("_")[1]
    k_right = key.split("_")[2]

    plt.rcParams["font.family"] = "DejaVu Sans"
    # Original
    plt.figure(figsize=(15, 6))
    lw_cf = plt.pcolormesh(
        lon, lev, q_lw[key]+t_lw[key],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0)
    )
    t_lw_c = plt.contour(
        lon, lev, t_reg[key],
        colors="k", linewidths=1, levels=t_level
    )
    q_lw_c = plt.contour(
        lon, lev, q_reg[key],
        colors="seagreen", linewidths=1, levels=q_level
    )
    plt.xlim(-100, 100)
    plt.axvline(0, color="k", linewidth=3, linestyle="--")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Longitude (degree)", fontsize=18)
    plt.ylabel("Level (hPa)", fontsize=18)
    plt.title(
        f"Regression Coefficient between LW and {dis_type.upper()} Prec (composite)",
        fontsize=20)
    plt.clabel(t_lw_c, inline=True, fontsize=10)
    plt.clabel(q_lw_c, inline=True, fontsize=10)
    plt.gca().invert_yaxis()
    cb = plt.colorbar(lw_cf)
    cb.ax.set_ylabel("Regression coefficient", fontsize=18)
    cb.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(
        f"/home/b11209013/2025_Research/Obs/Figure/ERA5_reg/qlw_wo_cloud_{dis_type}_k_{k_left}_{k_right}_compsite.png", dpi=300)
    plt.close()

    plt.figure(figsize=(15, 6))
    lw_cf = plt.pcolormesh(
        lon, lev, q_sw[key] + t_sw[key],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0)
    )
    t_lw_c = plt.contour(
        lon, lev, t_reg[key],
        colors="k", linewidths=1, levels=t_level
    )
    q_lw_c = plt.contour(
        lon, lev, q_reg[key],
        colors="seagreen", linewidths=1, levels=q_level
    )
    plt.xlim(-100, 100)
    plt.axvline(0, color="k", linewidth=3, linestyle="--")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Longitude (degree)", fontsize=18)
    plt.ylabel("Level (hPa)", fontsize=18)
    plt.title(
        f"Regression Coefficient between SW and {dis_type.upper()} Prec (composite)",
        fontsize=20)
    plt.clabel(t_lw_c, inline=True, fontsize=10)
    plt.clabel(q_lw_c, inline=True, fontsize=10)
    plt.gca().invert_yaxis()
    cb = plt.colorbar(lw_cf)
    cb.ax.set_ylabel("Regression coefficient", fontsize=18)
    cb.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(
        f"/home/b11209013/2025_Research/Obs/Figure/ERA5_reg/qsw_wo_cloud_{dis_type}_k_{k_left}_{k_right}_compsite.png", dpi=300)
    plt.close()
