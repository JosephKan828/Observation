# ========================================================================
# This program is to write perform linear regression with ridge regression
# ========================================================================

import os

# Set the number of threads (e.g., to 1 or 4)
n_threads = "1"

os.environ["OMP_NUM_THREADS"] = n_threads
os.environ["MKL_NUM_THREADS"] = n_threads
os.environ["OPENBLAS_NUM_THREADS"] = n_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads
os.environ["NUMEXPR_NUM_THREADS"] = n_threads

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

    return np.sqrt( np.mean( ( y_true - y_pred )**2 ) ).max()

def ridge_regression(
        X: np.ndarray,
        y: np.ndarray,
        alpha: float
) -> np.ndarray:
    
    assert X.ndim == 2, "X must be a 2D array"
    assert y.ndim == 2, "y must be a 2D array"
    assert X.shape[0] == y.shape[0], "Number of samples in X and y must match"

    XtY: np.ndarray = X.T @ y
    XtX: np.ndarray = X.T @ X + alpha * np.eye( X.shape[1] )

    beta_ridge: np.ndarray = np.linalg.solve( XtX, XtY )

    return beta_ridge

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
    # define a range of alphas for ridge regression
    alphas: np.ndarray = np.arange( 0.01, 0.02, 0.0001 )

    lw_rmse: list = []
    sw_rmse: list = []

    for alpha in alphas:
        # Acquire regression coefficients
        lw_lrf = ridge_regression( training["w"], training["lw"], alpha )
        sw_lrf = ridge_regression( training["w"], training["sw"], alpha )

        # examine rmse
        lw_pred_tmp = testing["w"] @ lw_lrf
        sw_pred_tmp = testing["w"] @ sw_lrf

        lw_rmse.append( [ rmse( testing["lw"], lw_pred_tmp ) ] )
        sw_rmse.append( [ rmse( testing["sw"], sw_pred_tmp ) ] )
    # print alpha with lowest rmse
    best_lw_alpha: float = alphas[ np.argmin( lw_rmse ) ]
    best_sw_alpha: float = alphas[ np.argmin( sw_rmse ) ]

    lw_lrf: np.ndarray = ridge_regression( training["w"], training["lw"], best_lw_alpha )
    sw_lrf: np.ndarray = ridge_regression( training["w"], training["sw"], best_sw_alpha )

    # best prediction
    lw_pred: np.ndarray = testing["w"] @ lw_lrf
    sw_pred: np.ndarray = testing["w"] @ sw_lrf

    print( f"Best LW alpha: {best_lw_alpha}, RMSE: {rmse( testing['lw'], lw_pred )}" )
    print( f"Best SW alpha: {best_sw_alpha}, RMSE: {rmse( testing['sw'], sw_pred )}" )

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
        -lw_lrf.T,
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
        -sw_lrf.T,
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

    # -------------------------------
    # Save regression results
    # -------------------------------
    OUTPUT_DIR: str = "/work/b11209013/2025_Research/MSI/Rad_Stuff/w_LRF.h5"

    with h5py.File(
        OUTPUT_DIR,
        "w"
    ) as f:
        f.create_dataset(
            "lw_lrf",
            data=lw_lrf
        )
        f.create_dataset(
            "sw_lrf",
            data=sw_lrf
        )

if __name__ == "__main__":
    main()