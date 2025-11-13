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
# 1.1. define functions
#####################

def phase_avg(
    central_idx : np.int64,
    k_ave       : np.float64,
    qlw         : np.ndarray,
    qsw         : np.ndarray
):

    # calculate size of averaging domain
    kernel_lens = np.round(2*np.pi*6.371*1e6*1e-3/(k_ave*4*110))/0.625
    half_kernel = int(kernel_lens // 2)

    # slice of window indexing
    sel_idx = slice(central_idx-half_kernel, central_idx+half_kernel+1)

    # select radiative heating profile
    qlw_comp = qlw[:,sel_idx]
    qsw_comp = qsw[:,sel_idx]

    # average over slected domain
    qlw_mean, qsw_mean = np.nanmean(qlw_comp,axis=-1), np.nanmean(qsw_comp, axis=-1)

    return qlw_mean, qsw_mean

#####################
# 2. load data
#####################

# wavenumber window
k_domains = [
    (2*i+1, 2*i+3) for i in range(5)
]

kaves = [(k[0]+k[1])/2 for k in k_domains]

PATH_INPUT = "/home/b11209013/2025_Research/Obs/Files/IMERG/Corr/"

# calculate central point for averaging

lon = np.linspace(-180, 180, 576)
mid_point = np.argmin(np.abs(lon - 0))

qlw_comps = {}; qsw_comps = {}
qlw_means = {}; qsw_means = {}

# calculate average profile for each smapling longitude
for i, k in enumerate(k_domains):
    KW_FILES = glob(f"{PATH_INPUT}corr_kw_k_{k[0]}_{k[1]}_lon=*.h5")

    kave = kaves[i]

    for kw_f in KW_FILES:
        lon = kw_f.split("=")[-1].split(".")[0]

        with h5py.File(kw_f, "r") as h:
            kw_qlw = np.array(h.get("qlw_corr"))
            kw_qsw = np.array(h.get("qsw_corr"))

        qlw_mean, qsw_mean = phase_avg(mid_point, kave, kw_qlw, kw_qsw)

        qlw_means[f"kw_{kave}_{lon}"]=qlw_mean; qsw_means[f"kw_{kave}_{lon}"]=qsw_mean

    with h5py.File(f"{PATH_INPUT}corr_kw_k_{k[0]}_{k[1]}_composite.h5","r") as kw_comp:
        kw_qlw_comp = np.array(kw_comp.get("qlw_corr"))
        kw_qsw_comp = np.array(kw_comp.get("qsw_corr"))

        kw_qlw_prof, kw_qsw_prof = phase_avg(mid_point, kave, kw_qlw_comp, kw_qsw_comp)

        qlw_comps[f"kw_{kave}"] = kw_qlw_prof
        qsw_comps[f"kw_{kave}"] = kw_qsw_prof

MJO_FILES = glob(f"{PATH_INPUT}corr_mjo_k_1_4_lon=*.h5")

mjo_kave = 2.5

for mjo_f in MJO_FILES:
    lon = mjo_f.split("=")[-1].split(".")[0]

    with h5py.File(mjo_f, "r") as h:
        mjo_qlw = np.array(h.get("qlw_corr"))
        mjo_qsw = np.array(h.get("qsw_corr"))

    qlw_mean, qsw_mean = phase_avg(mid_point, kave, mjo_qlw, mjo_qsw)

    qlw_means[f"mjo_{mjo_kave}_{lon}"]=qlw_mean; qsw_means[f"mjo_{mjo_kave}_{lon}"]=qsw_mean

with h5py.File(f"{PATH_INPUT}corr_mjo_k_1_4_composite.h5","r") as mjo_comp:
    mjo_qlw_comp = np.array(mjo_comp.get("qlw_corr"))
    mjo_qsw_comp = np.array(mjo_comp.get("qsw_corr"))

    mjo_qlw_prof, mjo_qsw_prof = phase_avg(mid_point, kave, mjo_qlw_comp, mjo_qsw_comp)

    qlw_comps[f"mjo_{mjo_kave}"] = mjo_qlw_prof
    qsw_comps[f"mjo_{mjo_kave}"] = mjo_qsw_prof

########################
# 3. Calculate composite
########################

qlw_means_stack = {
    f"kw_{kave}": np.stack([
        v for k, v in qlw_means.items()
        if k.startswith(f"kw_{kave}_")
    ], axis=0)
    for kave in kaves
}
qlw_means_stack["mjo_2.5"] = np.stack([
    v
    for k,v in qlw_means.items()
    if k.startswith("mjo_2.5_")
], axis=0)

qsw_means_stack = {
    f"kw_{kave}": np.stack([
        v for k, v in qsw_means.items()
        if k.startswith(f"kw_{kave}_")
    ], axis=0)
    for kave in kaves
}
qsw_means_stack["mjo_2.5"] = np.stack([
    v
    for k,v in qsw_means.items()
    if k.startswith("mjo_2.5_")
], axis=0)


qlw_comps_stack = np.stack([
    v for k, v in qlw_comps.items()
], axis=0)

qsw_comps_stack = np.stack([
    v for k, v in qsw_comps.items()
], axis=0)

#########################
# 4. Calculate percentile
#########################
qlw_mean_se = {
    key: qlw_means_stack[key].std(axis=0, ddof=1)/np.sqrt(qlw_means_stack[key].shape[0])
    for key in qlw_means_stack.keys()
}
qsw_mean_se = {
    key: qsw_means_stack[key].std(axis=0, ddof=1)/np.sqrt(qsw_means_stack[key].shape[0])
    for key in qsw_means_stack.keys()
}

##################
# Calculate relative magnitude of mean profile
##################
qlw_comps_ratio = qlw_comps_stack / np.nanmax(qlw_comps_stack, axis=1)[:,None]
qsw_comps_ratio = qsw_comps_stack / np.nanmax(qsw_comps_stack, axis=1)[:,None]

qlw_means_ratio = {
    key: qlw_means_stack[key] / np.nanmax(qlw_means_stack[key], axis=1)[:,None]
    for key in qlw_means_stack.keys()
}

qsw_means_ratio = {
    key: qsw_means_stack[key] / np.nanmax(qsw_means_stack[key], axis=1)[:,None]
    for key in qsw_means_stack.keys()
}

qlw_ratio_se = {
    key: qlw_means_ratio[key].std(axis=0, ddof=1)/np.sqrt(qlw_means_ratio[key].shape[0])
    for key in qlw_means_stack.keys()
}
qsw_ratio_se = {
    key: qsw_means_ratio[key].std(axis=0, ddof=1)/np.sqrt(qsw_means_ratio[key].shape[0])
    for key in qsw_means_stack.keys()
}


##################
# Compute normalized profile and spread
#################

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

p_weight = pressure_weights(levs)[None,:]

qlw_mean_norms = {
    key: qlw_means_stack[key]*p_weight / np.sum(qlw_means_stack[key]*p_weight, axis=-1, keepdims=True)
    for key in qlw_means_stack.keys()
}
print("qlw_mean_norms keys: ", qlw_mean_norms.keys())
qsw_mean_norms = {
    key: qsw_means_stack[key]*p_weight / np.sum(qsw_means_stack[key]*p_weight, axis=-1, keepdims=True)
    for key in qsw_means_stack.keys()
}

qlw_comp_norm = qlw_comps_stack*p_weight / np.sum(qlw_comps_stack*p_weight, axis=-1, keepdims=True)
qsw_comp_norm = qsw_comps_stack*p_weight / np.sum(qsw_comps_stack*p_weight, axis=-1, keepdims=True)

qlw_norm_se = {
    key: qlw_mean_norms[key].std(axis=0, ddof=1)/np.sqrt(qlw_mean_norms[key].shape[0])
    for key in qlw_mean_norms.keys()
}
qsw_norm_se = {
    key: qsw_mean_norms[key].std(axis=0, ddof=1)/np.sqrt(qsw_mean_norms[key].shape[0])
    for key in qsw_mean_norms.keys()
}


########################
# 4. plot figure
########################
cmap = plt.get_cmap("Spectral")

clist = [cmap(ratio) for ratio in [0.1, 0.2, 0.35, 0.65, 0.8, 1] ]

fig, ax = plt.subplots(figsize=(9, 15))

# plot qlw mean profile
for i, key in enumerate(qlw_comps.keys()):
    ax.plot(
        qlw_comps_stack[i], levs,
        label=key, color=clist[i]
    )
    ax.errorbar(qlw_comps_stack[i], levs, xerr=qlw_mean_se[key], fmt="none",
                color=clist[i], lw=1, capsize=2
                )
ax.set_xticks(np.linspace(-0.1, 0.1, 5))
ax.set_xticklabels(["-0.10","-0.05","0","0.05","0.10"], fontsize=16)
ax.set_yticks(np.linspace(1000, 100, 10))
ax.set_yticklabels(["1000","900","800","700","600","500","400","300","200","100"], fontsize=16)
ax.set_xlim(-0.1, 0.1)
ax.set_ylim(1000, 100)
ax.legend(fontsize=18)
ax.grid(True)
ax.set_xlabel("Averaged LW Heating", fontsize=18)
ax.set_ylabel("Pressure (hPa)", fontsize=18)
ax.set_title("Averaged LW Heating Profiles", fontsize=20)
plt.savefig("/home/b11209013/2025_Research/Obs/Figure/IMERG_corr/qlw_means.png", dpi=500)
plt.close()

# plot qsw mean profile 
fig, ax = plt.subplots(figsize=(9, 15))

for i, key in enumerate(qlw_comps.keys()):
    ax.plot(
        qsw_comps_stack[i], levs,
        label=key, color=clist[i]
    )
    ax.errorbar(qsw_comps_stack[i], levs, xerr=qsw_mean_se[key], fmt="none",
                color=clist[i], lw=1, capsize=2
                )
ax.set_xticks(np.linspace(-0.06, 0.06, 5))
ax.set_xticklabels(["-0.06","-0.03","0","0.03","0.06"], fontsize=16)
ax.set_yticks(np.linspace(1000, 100, 10))
ax.set_yticklabels(["1000","900","800","700","600","500","400","300","200","100"], fontsize=16)
ax.set_xlim(-0.1, 0.1)
ax.set_ylim(1000, 100)
ax.legend(fontsize=18)
ax.grid(True)
ax.set_xlabel("Averaged SW Heating", fontsize=18)
ax.set_ylabel("Pressure (hPa)", fontsize=18)
ax.set_title("Averaged SW Heating Profiles", fontsize=20)
plt.savefig("/home/b11209013/2025_Research/Obs/Figure/IMERG_corr/qsw_means.png", dpi=500)
plt.close()

# plot qlw norm profile 
fig, ax = plt.subplots(figsize=(9, 15))

for i, key in enumerate(qlw_mean_norms.keys()):
    ax.plot(
        qlw_comp_norm[i], levs,
        label=key, color=clist[i]
    )
    ax.errorbar(qlw_comp_norm[i], levs, xerr=qlw_norm_se[key], fmt="none",
                color=clist[i], lw=1, capsize=2
                )
ax.set_xticks(np.linspace(-0.1, 0.1, 5))
ax.set_xticklabels(["-0.10","-0.05","0","0.05","0.10"], fontsize=16)
ax.set_yticks(np.linspace(1000, 100, 10))
ax.set_yticklabels(["1000","900","800","700","600","500","400","300","200","100"], fontsize=16)
ax.set_xlim(-0.1, 0.1)
ax.set_ylim(1000, 100)
ax.legend(fontsize=18)
ax.grid(True)
ax.set_xlabel("Normalized LW Heating", fontsize=18)
ax.set_ylabel("Pressure (hPa)", fontsize=18)
ax.set_title("Normalized LW Heating Profiles", fontsize=20)
plt.savefig("/home/b11209013/2025_Research/Obs/Figure/IMERG_corr/qlw_norms.png", dpi=500)
plt.close()

# plot qsw norm profile 
fig, ax = plt.subplots(figsize=(9, 15))

for i, key in enumerate(qsw_mean_norms.keys()):
    ax.plot(
        qsw_comp_norm[i], levs,
        label=key, color=clist[i]
    )
    ax.errorbar(qsw_comp_norm[i], levs, xerr=qsw_norm_se[key], fmt="none",
                color=clist[i], lw=1, capsize=2
                )
ax.set_xticks(np.linspace(-0.4, 0.4, 5))
ax.set_xticklabels(["-0.4","-0.2","0","0.2","0.4"], fontsize=16)
ax.set_yticks(np.linspace(1000, 100, 10))
ax.set_yticklabels(["1000","900","800","700","600","500","400","300","200","100"], fontsize=16)
ax.set_xlim(-0.4, 0.4)
ax.set_ylim(1000, 100)
ax.legend(fontsize=18)
ax.grid(True)
ax.set_xlabel("Normalized SW Heating", fontsize=18)
ax.set_ylabel("Pressure (hPa)", fontsize=18)
ax.set_title("Normalized SW Heating Profiles", fontsize=20)
plt.savefig("/home/b11209013/2025_Research/Obs/Figure/IMERG_corr/qsw_norms.png", dpi=500)
plt.close()

# plot qlw ratio
fig, ax = plt.subplots(figsize=(9, 15))

# plot qlw mean profile
for i, key in enumerate(qlw_means_ratio.keys()):
    ax.plot(
        qlw_comps_ratio[i], levs,
        label=key, color=clist[i]
    )
    ax.errorbar(qlw_comps_ratio[i], levs, xerr=qlw_ratio_se[key], fmt="none",
                color=clist[i], lw=1, capsize=2
                )
ax.set_xticks(np.linspace(-1, 1, 5))
ax.set_xticklabels(["-1.0","-0.5","0","0.5","1.0"], fontsize=16)
ax.set_yticks(np.linspace(1000, 100, 10))
ax.set_yticklabels(["1000","900","800","700","600","500","400","300","200","100"], fontsize=16)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(1000, 100)
ax.legend(fontsize=18)
ax.grid(True)
ax.set_xlabel("Relative LW Heating (divide by max)", fontsize=18)
ax.set_ylabel("Pressure (hPa)", fontsize=18)
ax.set_title("Relative LW Heating Profiles", fontsize=20)
plt.savefig("/home/b11209013/2025_Research/Obs/Figure/IMERG_corr/qlw_ratio.png", dpi=500)
plt.close()

# plot qsw mean profile 
fig, ax = plt.subplots(figsize=(9, 15))

for i, key in enumerate(qsw_means_ratio.keys()):
    ax.plot(
        qsw_comps_ratio[i], levs,
        label=key, color=clist[i]
    )
    ax.errorbar(qsw_comps_ratio[i], levs, xerr=qsw_mean_se[key], fmt="none",
                color=clist[i], lw=1, capsize=2
                )
ax.set_xticks(np.linspace(-1, 1, 5))
ax.set_xticklabels(["-1.0","-0.5","0","0.5","1.0"], fontsize=16)
ax.set_yticks(np.linspace(1000, 100, 10))
ax.set_yticklabels(["1000","900","800","700","600","500","400","300","200","100"], fontsize=16)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(1000, 100)
ax.legend(fontsize=18)
ax.grid(True)
ax.set_xlabel("Relative SW Heating (divide by max)", fontsize=18)
ax.set_ylabel("Pressure (hPa)", fontsize=18)
ax.set_title("Relative SW Heating Profiles", fontsize=20)
plt.savefig("/home/b11209013/2025_Research/Obs/Figure/IMERG_corr/qsw_ratio.png", dpi=500)
plt.close()


