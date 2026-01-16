# ======================================================================================
# Calculate linear response function between vertical motion and cloud radiative heating
# ======================================================================================

# #####################
# Import Package
# #####################

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def EOF(
        data        : np.ndarray,
        n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    if data.ndim != 2:
        raise ValueError("Input data must be 2D")

    from sklearn.utils.extmath import randomized_svd

    # apply randomized SVD
    if n_components == None:
        U, S, Vt = randomized_svd(
        data,
        n_components=data.shape[-1],
        n_iter=5
    )

    else:
        U, S, Vt = randomized_svd(
            data,
            n_components=n_components,
            n_iter=5
        )

    EOFs: np.ndarray = Vt[ :n_components, : ] # shape: ( nmodes, nz )
    PCs : np.ndarray = ( data @ EOFs.T ) @ np.linalg.inv( EOFs @ EOFs.T ) # shape: ( nmodes, n_samples )
    print( EOFs.shape, PCs.shape )

    return EOFs, PCs.T

def normal_equation(
        data: np.ndarray,
        EOFs: np.ndarray
) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError("Input data must be 2D")
    if EOFs.ndim != 2:
        raise ValueError("Input EOFs must be 2D")

    PCs: np.ndarray = data @ EOFs.T @ np.linalg.inv( EOFs @ EOFs.T )

    return PCs.T # shape: ( nmodes, n_samples )

def main() -> None:
    # ###############
    # Load Data
    # ###############
    # input directory
    INPUT_DIR: str = "/work/b11209013/2025_Research/regression/"

    # Load composited longwave and shortwave profile
    lw_comp: dict[str, np.ndarray] = dict()
    sw_comp: dict[str, np.ndarray] = dict()

    with h5py.File(
        INPUT_DIR + "IMERG_CLOUDSAT.h5",
        "r"
    ) as f:
        lw_comp_group = f["lw_composite"] #type: ignore
        sw_comp_group = f["sw_composite"] #type: ignore

        assert isinstance(lw_comp_group, h5py.Group)
        assert isinstance(sw_comp_group, h5py.Group)

        lw_comp_group: h5py.Group = lw_comp_group
        sw_comp_group: h5py.Group = sw_comp_group

        for key in lw_comp_group.keys():
            lw_comp[key] = np.array( lw_comp_group.get( key ) )
            sw_comp[key] = np.array( sw_comp_group.get( key ) )

    # Load composited vertical velocity profile
    w_comp: dict[str, np.ndarray] = dict()

    with h5py.File(
        INPUT_DIR + "IMERG_ERA5.h5",
        "r"
    ) as f:
        w_comp_group = f["w_composite"] #type: ignore

        assert isinstance(w_comp_group, h5py.Group)

        w_comp_group: h5py.Group = w_comp_group

        for key in w_comp_group.keys():
            w_comp[key] = np.array( w_comp_group.get( key ) )

    # ##################
    # Apply EOF
    # ##################

    # concatenate data for EOF
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

    # split data into training and verifying dataset
    lw_train: np.ndarray = lw[:, :3000].T
    sw_train: np.ndarray = sw[:, :3000].T
    w_train : np.ndarray = w[:, :3000].T

    lw_verify: np.ndarray = lw[:, 3000:].T
    sw_verify: np.ndarray = sw[:, 3000:].T
    w_verify : np.ndarray = w[:, 3000:].T

    # Apply EOF
    n_components = 7
    w_EOFs, w_pcs = EOF(
        w_train - np.nanmean( w_train, axis=0, keepdims=True ),
        n_components=n_components
    )

    lw_pcs: np.ndarray = normal_equation(
        lw_train - np.nanmean( lw_train, axis=0, keepdims=True ),
        w_EOFs
    )

    sw_pcs: np.ndarray = normal_equation(
        sw_train - np.nanmean( sw_train, axis=0, keepdims=True ),
        w_EOFs
    )

    # ##################################
    # Calculate linear response function
    # ##################################

    lrf_lw_w: np.ndarray = lw_pcs @ w_pcs.T @ np.linalg.inv( w_pcs @ w_pcs.T )
    lrf_sw_w: np.ndarray = sw_pcs @ w_pcs.T @ np.linalg.inv( w_pcs @ w_pcs.T )

    with h5py.File(
        "/work/b11209013/2025_Research/MSI/Rad_Stuff/w_LRF.h5",
        "w"
    ) as f:
        f.create_dataset( "EOF"   , data=w_EOFs )
        f.create_dataset( "LRF_lw", data=lrf_lw_w )
        f.create_dataset( "LRF_sw", data=lrf_sw_w )

    # ###########################
    # Verify the validity of LRF
    # ###########################

    # Calculate the PCs
    w_pcs_verify: np.ndarray = normal_equation(
        w_verify - np.nanmean( w_verify, axis=0, keepdims=True ),
        w_EOFs
    )

    # predict radiative heating
    lw_pred: np.ndarray = w_EOFs.T @ ( lrf_lw_w @ w_pcs_verify )
    sw_pred: np.ndarray = w_EOFs.T @ ( lrf_sw_w @ w_pcs_verify )

    # ##########################
    # Demonstract errors
    # ##########################

    fig = plt.figure( figsize=(13, 8) )

    pcm = plt.pcolormesh(
        np.arange( lw_verify.shape[0] ),
        np.linspace( 1000, 100, 37 ),
        lw_verify.T,
        cmap="RdBu_r", norm=TwoSlopeNorm( vcenter=0, vmin=-0.1, vmax=0.1 )
    )
    ct = plt.contour(
        np.arange( lw_pred.shape[1] ),
        np.linspace( 1000, 100, 37 ),
        lw_pred,
        colors="black",
        linewidths=1.25,
        levels=[ -0.05, -0.03, -0.02, 0.02, 0.03, 0.05 ],
    )
    ct2 = plt.contour(
        np.arange( w_verify.shape[0] ),
        np.linspace( 1000, 100, 37 ),
        w_verify.T,
        colors="seagreen",
        linewidths=1.25
    )
    plt.minorticks_on()
    plt.xticks([])
    plt.yticks(np.linspace( 1000, 100, 10 ), fontsize=16)
    plt.ylim(1000, 100)
    plt.clabel(ct, colors="black", fontsize=8)
    plt.clabel(ct2, colors="seagreen", fontsize=8)
    cbar = plt.colorbar(pcm, orientation="horizontal", aspect=50, shrink=0.8, pad=0.10)
    cbar.set_ticks( 
        [-0.1, -0.05, 0, 0.05, 0.1]
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("[ K day$^{-1}$ / mm day$^{-1}$ ]", fontsize=18)
    plt.xlabel("Samples", fontsize=18)
    plt.ylabel("Pressure ( hPa )", fontsize=18)
    plt.savefig(
        "lw_cld_verify.png", dpi=600, bbox_inches="tight"
    )
    plt.close()

    fig = plt.figure( figsize=(13, 8) )

    pcm = plt.pcolormesh(
        np.arange( sw_verify.shape[0] ),
        np.linspace( 1000, 100, 37 ),
        sw_verify.T,
        cmap="RdBu_r", norm=TwoSlopeNorm( vcenter=0, vmin=-0.1, vmax=0.1 )
    )
    ct = plt.contour(
        np.arange( lw_pred.shape[1] ),
        np.linspace( 1000, 100, 37 ),
        sw_pred,
        colors="black",
        linewidths=1.25,
        levels=[ -0.05, -0.03, -0.02, 0.02, 0.03, 0.05 ],
    )
    plt.minorticks_on()
    plt.xticks([])
    plt.yticks(np.linspace( 1000, 100, 10 ), fontsize=16)
    plt.ylim(1000, 100)
    plt.clabel(ct, colors="black", fontsize=8)
    cbar = plt.colorbar(pcm, orientation="horizontal", aspect=50, shrink=0.8, pad=0.10)
    cbar.set_ticks( 
        [-0.1, -0.05, 0, 0.05, 0.1]
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("[ K day$^{-1}$ / mm day$^{-1}$ ]", fontsize=18)
    plt.xlabel("Samples", fontsize=18)
    plt.ylabel("Pressure ( hPa )", fontsize=18)
    plt.savefig(
        "sw_cld_verify.png", dpi=600, bbox_inches="tight"
    )
    plt.close()

    def rmse(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.float64:
        return np.nanmax( np.sqrt( np.nanmean( ( y_true - y_pred )**2 ) ) )

    print( f"lw_rmse max: {rmse( lw_verify.T, lw_pred )}" )
    print( f"sw_rmse max: {rmse( sw_verify.T, sw_pred )}" )

if __name__ == "__main__":
    main()
