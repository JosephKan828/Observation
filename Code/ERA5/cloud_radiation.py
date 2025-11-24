# =================================================
# This program is to compute cloud radiative effect
# =================================================

# #######################
# Import Package
# #######################

import h5py
import numpy as np

from glob import glob
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# #######################
# Load data
# #######################

# Load moisture radiation
with h5py.File("/home/b11209013/2025_Research/Obs/Files/ERA5/rad_wo_cloud.h5", "r") as f:

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

# load cloud radiation
PATH_INPUT = "/home/b11209013/2025_Research/Obs/Files/IMERG/Corr/"

FILES = sorted(glob(f"{PATH_INPUT}*_composite.h5"))

tot_rad = {
    "lw": {},
    "sw": {}
}

for i, key in enumerate(lw.keys()):

    with h5py.File(FILES[i], "r") as h:
        tot_rad["lw"][key] = np.array(h.get("qlw_reg"))
        tot_rad["sw"][key] = np.array(h.get("qsw_reg"))

# ################################################
# Subtract total radiation with moisture radiation
# ################################################
cloud_rad = {
    "lw": {
        key: tot_rad["lw"][key] - lw[key]
        for key in lw.keys()
    },
    "sw": {
        key: tot_rad["sw"][key] - sw[key]
        for key in sw.keys()
    }
}

with h5py.File("/home/b11209013/2025_Research/Obs/Files/ERA5/cloud_rad.h5", "w") as f:
    grp_lw = f.create_group("lw")

    for key in cloud_rad["lw"].keys():
        grp_lw.create_dataset(key, data=np.asarray(cloud_rad["lw"][key]))

    grp_sw = f.create_group("sw")

    for key in cloud_rad["sw"].keys():
        grp_sw.create_dataset(key, data=np.asarray(cloud_rad["sw"][key]))
