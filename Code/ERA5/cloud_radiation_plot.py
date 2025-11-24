# =============================
# Plot cloud radiation only
# =============================

# #################
# Import package
# #################

import h5py
import numpy as np

from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ################
# Load data
# ################

# load cloud radiation
with h5py.File("/home/b11209013/2025_Research/Obs/Files/ERA5/cloud_rad.h5", "r") as f:
    lw_grp = f["lw"]

    lw = {
        key: np.array(lw_grp.get(key))
        for key in lw_grp.keys()
    }

    sw_grp = f["sw"]

    sw = {
        key: np.array(sw_grp.get(key))
        for key in sw_grp.keys()
    }

# load temperature and moisture
FILES = sorted(
    glob("/home/b11209013/2025_Research/Obs/Files/ERA5/reg/*_composite.h5")
)

t_reg = {}
q_reg = {}

for i, key in enumerate(lw.keys()):
    with h5py.File(FILES[i], "r") as f:

        t_reg[key] = np.array(f["t_reg"])
        q_reg[key] = np.array(f["q_reg"])

# ####################
# Plot figure
# ####################

lon = np.linspace(-180, 180, 576)
lev = np.linspace(1000, 100, 37)

for key in tqdm(lw.keys()):
    tmax = np.max(np.abs(t_reg[key]))
    tlevel = np.linspace(-tmax, tmax, 11)
    qmax = np.max(np.abs(q_reg[key]))
    qlevel = np.linspace(-qmax, qmax, 11)

    t_level = [t for t in tlevel if t != 0]
    q_level = [q for q in qlevel if q != 0]

    lw_max = np.max(np.abs(lw[key][1:-1]))
    sw_max = np.max(np.abs(sw[key][1:-1]))

    dis_type = key.split("_")[0]
    k_left = key.split("_")[1]
    k_right = key.split("_")[2]

    plt.rcParams["font.family"] = "Dejavu Sans"

    plt.figure(figsize=(15, 6))
    lw_cf = plt.pcolormesh(
        lon, lev, lw[key],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0, vmin=-3, vmax=3)
    )
    t_lw = plt.contour(
        lon, lev, t_reg[key],
        colors="k", linewidths=1, levels=t_level
    )
    q_lw = plt.contour(
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
        f"Regression Coefficient between LW and {dis_type.upper()} Prec (composite)", fontsize=20)
    plt.clabel(t_lw, inline=True, fontsize=10)
    plt.clabel(q_lw, inline=True, fontsize=10)
    plt.gca().invert_yaxis()
    cb = plt.colorbar(lw_cf)
    cb.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(
        f"/home/b11209013/2025_Research/Obs/Figure/ERA5_reg/lw_cloud_{dis_type}_k_{k_left}_{k_right}_compsite.png", dpi=300)
    plt.close()

    plt.figure(figsize=(15, 6))
    sw_cf = plt.pcolormesh(
        lon, lev, sw[key],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0, vmin=-3, vmax=3)
    )
    t_sw = plt.contour(
        lon, lev, t_reg[key],
        colors="k", linewidths=1, levels=t_level
    )
    q_sw = plt.contour(
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
        f"Regression Coefficient between SW and {dis_type.upper()} Prec (composite)", fontsize=20)
    plt.clabel(t_sw, inline=True, fontsize=10)
    plt.clabel(q_sw, inline=True, fontsize=10)
    plt.gca().invert_yaxis()
    cb = plt.colorbar(sw_cf)
    cb.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(
        f"/home/b11209013/2025_Research/Obs/Figure/ERA5_reg/sw_cloud_{dis_type}_k_{k_left}_{k_right}_compsite.png", dpi=300)
    plt.close()
