# This program is to compute analytical LRF

import sys;
import h5py;
import numpy as np;
import climlab as cl;

from copy import deepcopy;
from matplotlib import pyplot as plt;

sys.path.append("/home/b11209013/Package/"); # path to my packages
import Plot_Style as ps;

ps.apply_custom_plot_style();

# ==== 1. Load mean state ==== #
with h5py.File("/home/b11209013/2025_Research/CloudSat/Files/era5_qmean.h5") as f:
    q = np.array(f.get("q"))[::-1];

with h5py.File("/home/b11209013/2025_Research/CloudSat/Files/era5_tmean.h5") as f:
    t = np.array(f.get("t"))[::-1];

# constructing RRTMG
levs = np.linspace(100.0, 1000.0, 37)
lev_lim = np.argmin(np.abs(levs-300.0))

nlev = len(levs)

state = cl.column_state(num_lev=nlev, water_depth=1)
state["Tatm"][:] = np.array(t);
state["Ts"][:]   = state["Tatm"][-1];

rad_model = cl.radiation.RRTMG(
    name="Radiation Model",
    state=state,
    specific_humidity=q,
    albedo=0.3,
);

rad_model.compute_diagnostics();

LW_ref = rad_model.diagnostics["TdotLW"].copy();
SW_ref = rad_model.diagnostics["TdotSW"].copy();

LRF = {
    "q_lw": np.zeros((nlev, nlev)),
    "q_sw": np.zeros((nlev, nlev)),
    "t_lw": np.zeros((nlev, nlev)),
    "t_sw": np.zeros((nlev, nlev)),
    }

for l in range(nlev):
    q_perturb = deepcopy(q);
    pert = q_perturb[l]*0.01
    q_perturb[l] += pert; # perturb specific humidity by 0.01 kg/kg

    rad_perturb = cl.radiation.RRTMG(
        name="Radiation Model",
        state=state,
        specific_humidity=q_perturb,
        albedo=0.3,
    );

    rad_perturb.compute_diagnostics();

    LRF["q_lw"][l] = (rad_perturb.diagnostics["TdotLW"] - LW_ref) / pert*1e-3;
    LRF["q_sw"][l] = (rad_perturb.diagnostics["TdotSW"] - SW_ref) / pert*1e-3;

    del q_perturb, rad_perturb;

for l in range(nlev):
    perturb_state = deepcopy(state);
    perturb_state["Tatm"][l] += 1;
    perturb_state["Ts"][:] = perturb_state["Tatm"][-1];

    rad_perturb = cl.radiation.RRTMG(
        name="Radiation Model",
        state=perturb_state,
        specific_humidity=q,
        albedo=0.3,
    );

for key in LRF.keys():
    tmp_lrf = np.zeros((nlev, nlev));

    tmp_lrf[lev_lim:, lev_lim:] += LRF[key][lev_lim:, lev_lim:];

    LRF[key] = tmp_lrf.T;

with h5py.File("/home/b11209013/2025_Research/CloudSat/Files/RRTMG_LRF.h5", "w") as f:
    f.create_dataset("q_lw", data=LRF["q_lw"]);
    f.create_dataset("q_sw", data=LRF["q_sw"]);
    f.create_dataset("t_lw", data=LRF["t_lw"]);
    f.create_dataset("t_sw", data=LRF["t_sw"]);

