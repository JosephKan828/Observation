# ========================================================================
# This program is to write perform linear regression with ridge regression
# ========================================================================

import h5py
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm
from sklearn.linear_model import RidgeCV

def rmse(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> np.float64:
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of true and predicted values must match")

    return np.sqrt( np.mean( ( y_true - y_pred )**2 ) )

def main() -> None:
    # -------------------------------
    #  Load data
    # -------------------------------

    # Input directory
    INPUT_DIR: str = "/work/b11209013/2025_Research/regression/"

    # Load composited longwave and shortwave profile
    lw_comp: dict[str, np.ndarray] = dict()
    sw_comp: dict[str, np.ndarray] = dict()

    with h5py.File(
        INPUT_DIR + "IMERG_CLOUDSAT.h5",
        "r"
    ) as f:
        lw_comp_group: h5py.Group = f["lw_composite"] #type: ignore
        sw_comp_group: h5py.Group = f["sw_composite"] #type: ignore

        for key in lw_comp_group.keys():
            lw_comp[key] = np.array( lw_comp_group.get( key ) )
            sw_comp[key] = np.array( sw_comp_group.get( key ) )

    # Load composited vertical velocity profile
    w_comp: dict[str, np.ndarray] = dict()

    with h5py.File(
        INPUT_DIR + "IMERG_ERA5.h5",
        "r"
    ) as f:
        w_comp_group: h5py.Group = f["w_composite"] #type: ignore

        for key in w_comp_group.keys():
            w_comp[key] = np.array( w_comp_group.get( key ) )

    # -------------------------------
    # Prepare data for regression
    # -------------------------------

    # concatenate data for regression
    # shape: ( nz, nsamples )
    lw: np.ndarray = np.concatenate(
        list( lw_comp.values() ),
        axis=1
    )
    sw: np.ndarray = np.concatenate(
        list( sw_comp.values() ),
        axis=1
    )
    w: np.ndarray = np.concatenate(
        list( w_comp.values() ),
        axis=1
    )

    # split data into training and testing set
    training_size : int = 4*576

    training: dict[ str, np.ndarray ] = {
        "lw": lw[:, :training_size].T,
        "sw": sw[:, :training_size].T,
        "w" : w[:, :training_size].T
    }

    testing: dict[ str, np.ndarray ] = {
        "lw": lw[:, training_size:].T,
        "sw": sw[:, training_size:].T,
        "w" : w[:, training_size:].T
    }

    # -------------------------------
    # Perform regression
    # -------------------------------
    # define a rane of alphas for ridge regression
    alphas: np.ndarray = np.logspace( -5, 5, 10 )

    # regresion lw and w
    ridge_lw: RidgeCV = RidgeCV( alphas=alphas, scoring='neg_mean_squared_error', alpha_per_target=True )
    ridge_lw.fit( training["w"], training["lw"] )

    lw_lrf: np.ndarray = ridge_lw.coef_ # axis0: target, axis1: features

    # regresion sw and w
    ridge_sw: RidgeCV = RidgeCV( alphas=alphas, scoring='neg_mean_squared_error', alpha_per_target=True )
    ridge_sw.fit( training["w"], training["sw"] )

    sw_lrf: np.ndarray = ridge_sw.coef_

    # -------------------------------
    # Verify regression
    # -------------------------------
    lw_pred: np.ndarray = testing["w"] @ lw_lrf.T
    sw_pred: np.ndarray = testing["w"] @ sw_lrf.T

    lw_rmse: np.float64 = rmse( testing["lw"], lw_pred )
    sw_rmse: np.float64 = rmse( testing["sw"], sw_pred )

    # -------------------------------
    # Plot verification results
    # -------------------------------

    lw_levels_max = np.nanmax( np.abs( lw_pred ) )

    lw_levels : np.ndarray = np.array([ -0.05, -0.04, -0.03, 0.03, 0.04, 0.05 ])

    fig = plt.figure( figsize=( 10.5, 6.2 ) )
    lw_ver_pcm = plt.pcolormesh(
        np.linspace( 1,  testing["lw"].shape[0], testing["lw"].shape[0] ),
        np.linspace( 1000, 100, 37 ),
        testing["lw"].T,
        cmap="RdBu_r", norm=TwoSlopeNorm( vcenter=0.0, vmin=-0.1, vmax=0.1 )
    )
    lw_ct = plt.contour(
        np.linspace( 1,  testing["lw"].shape[0], testing["lw"].shape[0] ),
        np.linspace( 1000, 100, 37 ),
        lw_pred.T,
        colors="k",
        levels=lw_levels,
        linewidths=0.8
    )
    plt.axvline( x=576, color="k", linestyle="--", linewidth=1 )
    plt.axvline( x=1152, color="k", linestyle="--", linewidth=1 )

    plt.clabel( lw_ct, fmt="%.3f", fontsize=10 )
    plt.ylim( 1000, 100 )
    plt.minorticks_on()
    plt.xlabel( "Sample Index", fontsize=14 )
    plt.ylabel( "Pressure (hPa)", fontsize=14 )
    cbar = plt.colorbar( lw_ver_pcm, orientation="horizontal", pad=0.1, label="LW" )
    cbar.set_label("LW", fontsize=12 )
    cbar.ax.tick_params( labelsize=12 )


    plt.tight_layout()
    plt.savefig( f"/home/b11209013/2025_Research/Obs/Figure/lw_sw_ver.png", dpi=300 )
    plt.close()


    # -------------------------------
    # Plot regression results
    # -------------------------------

    fig, axs = plt.subplots( 1, 2, figsize=( 18.4, 10.5 ), sharey=True )

    lw_pcm = axs[0].pcolormesh(
        np.linspace( 1000, 100, 37 ),
        np.linspace( 1000, 100, 37 ),
        lw_lrf,
        cmap="RdBu_r", norm=TwoSlopeNorm( vcenter=0.0 )
    )
    axs[0].plot( [1000, 100], [1000, 100], color="k", linestyle="--", linewidth=1 )
    axs[0].minorticks_on()
    axs[0].tick_params( axis='both', labelsize=18 )
    axs[0].set_xlim( 1000, 100 )
    axs[0].set_ylim( 1000, 100 )
    axs[0].set_xlabel( "Feature", fontsize=18 )
    axs[0].set_ylabel( "Target", fontsize=18 )
    cbar = fig.colorbar( lw_pcm, ax=axs[0], orientation="horizontal", pad=0.1, label="LW" )
    cbar.set_label("LW", fontsize=16 )
    cbar.ax.tick_params( labelsize=16 )

    sw_pcm = axs[1].pcolormesh(
        np.linspace( 1000, 100, 37 ),
        np.linspace( 1000, 100, 37 ),
        sw_lrf,
        cmap="RdBu_r", norm=TwoSlopeNorm( vcenter=0.0 )
    )
    axs[1].plot( [1000, 100], [1000, 100], color="k", linestyle="--", linewidth=1 )
    axs[1].minorticks_on()
    axs[1].tick_params( axis='both', labelsize=18 )
    axs[1].set_xlim( 1000, 100 )
    axs[1].set_ylim( 1000, 100 )
    axs[1].set_xlabel( "Feature", fontsize=18 )
    cbar = fig.colorbar( sw_pcm, ax=axs[1], orientation="horizontal", pad=0.1 )
    cbar.set_label("SW", fontsize=16 )
    cbar.ax.tick_params( labelsize=16 )

    plt.tight_layout()
    plt.savefig( f"/home/b11209013/2025_Research/Obs/Figure/w_lrf.png", dpi=600 )
    plt.close( fig )

if __name__ == "__main__":
    main()