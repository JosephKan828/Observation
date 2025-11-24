# ==========================================================
# This program is to calculate moisture-radiation relation
# with linear approximation
# ==========================================================

# ###################
# Import Package
# ###################

import h5py
import numpy as np
import xarray as xr

from glob import glob
from matplotlib import pyplot as plt

# ###################
# Load data
# ###################

# Load linear response function
LRF = {}

with h5py.File("/work/b11209013/2025_Research/MSI/Rad_Stuff/LRF.h5", "r") as f:
    LRF["t_lw"] = np.array(f.get("t_lw"))[::-1, ::-1]
    LRF["t_sw"] = np.array(f.get("t_sw"))[::-1, ::-1]
    LRF["q_lw"] = np.array(f.get("q_lw"))[::-1, ::-1]
    LRF["q_sw"] = np.array(f.get("q_sw"))[::-1, ::-1]

# Load ERA5 regression profile
PATH_REG = "/home/b11209013/2025_Research/Obs/Files/ERA5/reg/"

FILES = sorted(glob(f"{PATH_REG}*_composite.h5"))

t_reg = {}  # dictionary to save temperature regression coefficient
q_reg = {}  # dictionary to save moisture regression coefficient

for file in FILES:
    key_dis_type = file.split("_")[2]
    key_k_left = file.split("_")[4]
    key_k_right = file.split("_")[5]

    with h5py.File(file, "r") as f:

        t_reg[f"{key_dis_type}_{key_k_left}_{key_k_right}"] = np.array(
            f.get("t_reg"))
        q_reg[f"{key_dis_type}_{key_k_left}_{key_k_right}"] = np.array(
            f.get("q_reg"))*1000.0

nz, nx = t_reg[list(t_reg.keys())[0]].shape  # shape of level and longitude

# ###################################################################
# Compute radiative heating corresponding to moisture and temperature
# ###################################################################

q_lw = {}
q_sw = {}
t_lw = {}
t_sw = {}

for i, key in enumerate(t_reg.keys()):
    q_lw[key] = LRF["q_lw"] @ q_reg[key]
    q_sw[key] = LRF["q_sw"] @ q_reg[key]
    t_lw[key] = LRF["t_lw"] @ t_reg[key]
    t_sw[key] = LRF["t_sw"] @ t_reg[key]

# ###############
# save to file
# ###############

PATH_OUTPUT = "/home/b11209013/2025_Research/Obs/Files/ERA5/moisture_rad.h5"

with h5py.File(PATH_OUTPUT, "w") as f:
    grp_q_lw = f.create_group("q_lw")

    for key, val in q_lw.items():
        grp_q_lw.create_dataset(key, data=np.asarray(val))

    grp_q_sw = f.create_group("q_sw")

    for key, val in q_sw.items():
        grp_q_sw.create_dataset(key, data=np.asarray(val))

    grp_t_lw = f.create_group("t_lw")

    for key, val in t_lw.items():
        grp_t_lw.create_dataset(key, data=np.asarray(val))

    grp_t_sw = f.create_group("t_sw")

    for key, val in t_sw.items():
        grp_t_sw.create_dataset(key, data=np.asarray(val))
