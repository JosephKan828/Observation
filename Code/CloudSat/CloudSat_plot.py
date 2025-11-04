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
# k_domains = [
#     (2*n+1, 2*n+3) for n in range(5)
# ]

k_domains=[(1, 4)]

for k_domain in k_domains:
    # Selected CloudSat data
    with h5py.File(f"/work/b11209013/2025_Research/CloudSat/k_window/mjo_{k_domain[0]}_{k_domain[1]}.h5", "r") as f:
        qlw = np.array(f.get("qlw"))
        qsw = np.array(f.get("qsw"))

    k_ave      = (k_domain[0] + k_domain[1]) / 2
    kernel_len = np.round(2*np.pi*6.371*1e6*1e-3/(k_ave*4*110))/0.625

    # kernel_len = np.round(2*np.pi*6.371*1e6*1e-3/(9*4*110))/0.625

    kernel     = np.ones(int(kernel_len))/ int(kernel_len)

    qlw_smooth = convolve1d(np.nanmean(qlw, axis=0), kernel, axis=-1)
    qsw_smooth = convolve1d(np.nanmean(qsw, axis=0), kernel, axis=-1)

    x = np.arange(-50, 49.3, 0.625)
    z = np.linspace(100, 1000, 37)[::-1]

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

    def polish_axes(ax):
        ax.tick_params(direction="in", length=6, width=1.1, top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

    def _common_axis_format(ax, x, z):
        """
        x: 1D array [m] horizontal distance
        z: 1D array [m] height
        """

        # ax.set_yscale("log")
        # X ticks every ~20 * 100 km assuming [-4e6,4e6] like your movie
        ax.set_xlabel("Relative Longitude [ degree ]", fontsize=18)
        ax.set_xticks(np.linspace(-50, 50, 5))
        ax.set_xticklabels(["-50","-25","0","25","50"], fontsize=16)


        # Z ticks every 2 km up to 14 km
        ax.set_ylabel("Level [ hPa ]", fontsize=18)
        ax.set_yticks(np.linspace(1000, 100, 10))
        ax.set_yticklabels(["1000","900","800","700","600","500","400","300","200","100"], fontsize=16)

        ax.set_xlim(-50, 50)
        ax.set_ylim(1000, 100)


        polish_axes(ax)

    def plot_qlw_panel(x, z, qlw_smooth, window, out_path):
        """
        Make academic-style figure for longwave heating anomaly.
        qlw_smooth: 2D [z, x] in K/day
        """
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(12.0, 10.5), 
            sharey=True, sharex=False,
            gridspec_kw={'width_ratios': [3, 1]})

        # If qlw_smooth is [z,x], we want Z on y-axis, X on x-axis.
        # contourf expects (Y,X,Zdata) shape match, so we pass (x,z,field) with transpose.
        cf = ax1.contourf(
            x, z, qlw_smooth,         # shape (len(z), len(x))
            levels=np.linspace(-1, 1, 21),
            cmap="RdBu_r",
            # norm=norm,
            extend="both"
        )

        cbar = fig.colorbar(cf, ax=ax1, shrink=0.8, aspect=35)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label("[ K day$^{-1}$ ]", fontsize=18)
        # ax1.set_yscale("log")
        _common_axis_format(ax1, x, z)

        title_txt = (
            r"$Q_{\rm LW}^\prime$ " + f"k={k_domain[0]} - {k_domain[1]}"
        #    f"(λ = {lam_km:.0f} km)   Max = {np.nanmax(np.abs(qlw_smooth)):.2f} K day$^{{-1}}$"
        )
        ax1.set_title(title_txt, fontsize=18)


        half = x.shape[0] // 2
        half_window = window.shape[0]//2

        qlw_mean = np.nanmean(qlw_smooth[:, half-half_window:half+half_window+1], axis=-1)  

        qlw_norm = qlw_mean*pressure_weights(z) / np.sum(qlw_mean * pressure_weights(z), axis=-1, keepdims=True)

        vmax = np.max(np.abs(qlw_norm))*1.05

        ax2.plot(qlw_norm, z, color="k")
        ax2.set_xlim(-vmax, vmax)
        ax2.vlines(0, 1000, 100, color="k", linestyles="dashed")
        ax2.set_xlabel(r"Normalized $Q_{LW}^\prime$", fontsize=18)

        fig.tight_layout()
        fig.savefig(out_path, dpi=500, bbox_inches="tight")
        plt.close(fig)


    def plot_qsw_panel(x, z, qsw_smooth, window, out_path):
        """
        Make academic-style figure for longwave heating anomaly.
        qlw_smooth: 2D [z, x] in K/day
        """
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(12.0, 10.5), 
            sharey=True, sharex=False,
            gridspec_kw={'width_ratios': [3, 1]})

        # If qlw_smooth is [z,x], we want Z on y-axis, X on x-axis.
        # contourf expects (Y,X,Zdata) shape match, so we pass (x,z,field) with transpose.
        cf = ax1.contourf(
            x, z, qsw_smooth,         # shape (len(z), len(x))
            levels=np.linspace(-1, 1, 21),
            cmap="RdBu_r",
            # norm=norm,
            extend="both"
        )

        cbar = fig.colorbar(cf, ax=ax1, shrink=0.8, aspect=35)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label("[ K day$^{-1}$ ]", fontsize=18)
        # ax1.set_yscale("log")
        _common_axis_format(ax1, x, z)

        title_txt = (
            r"$Q_{\rm SW}^\prime$ " + f"k={k_domain[0]} - {k_domain[1]}"
        #    f"(λ = {lam_km:.0f} km)   Max = {np.nanmax(np.abs(qlw_smooth)):.2f} K day$^{{-1}}$"
        )
        ax1.set_title(title_txt, fontsize=18)


        half = x.shape[0] // 2
        half_window = window.shape[0]//2

        qsw_mean = np.nanmean(qsw_smooth[:, half-half_window:half+half_window+1], axis=-1)  

        qsw_norm = qsw_mean*pressure_weights(z) / np.sum(qsw_mean * pressure_weights(z), axis=-1, keepdims=True)

        vmax = np.max(np.abs(qsw_norm))*1.05

        ax2.plot(qsw_norm, z, color="k")
        ax2.set_xlim(-vmax, vmax)
        ax2.vlines(0, 1000, 100, color="k", linestyles="dashed")
        ax2.set_xlabel(r"Normalized $Q_{SW}^\prime$", fontsize=18)

        fig.tight_layout()
        fig.savefig(out_path, dpi=500, bbox_inches="tight")
        plt.close(fig)


    plot_qlw_panel(x, z, qlw_smooth, window=kernel, out_path=f"/home/b11209013/2025_Research/CloudSat/Figure/CloudSat_profile/Vary_window/qlw_comp_mjo_k_{k_domain[0]}_{k_domain[1]}.png")
    plot_qsw_panel(x, z, qsw_smooth, window=kernel, out_path=f"/home/b11209013/2025_Research/CloudSat/Figure/CloudSat_profile/Vary_window/qsw_comp_mjo_k_{k_domain[0]}_{k_domain[1]}.png")
