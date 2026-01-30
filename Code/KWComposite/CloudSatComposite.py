# =============================================================================
# Composite CloudSat Radiative Heating
# =============================================================================

# =============================================================================
# Import Package
# =============================================================================
import h5py
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List, Dict, Any, Hashable
from scipy.ndimage import convolve1d
from pprint import pprint
from pathlib import Path
from matplotlib.colors import TwoSlopeNorm


# =============================================================================
# Helper Functions
# =============================================================================
def render_plot(data, levels, save_path):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    # Using 'shading=nearest' or 'gouraud' can be faster/smoother for large grids
    pcm = ax.pcolormesh(
        np.arange(-180, 180, 0.625),
        levels,
        data,
        cmap="RdBu_r",
        shading="auto",
        rasterized=True,
        norm=TwoSlopeNorm(vcenter=0),
    )

    ax.set_xlim(-100, 100)
    ax.set_ylim(1000, 100)
    ax.set_xlabel(r"Relative Longitude [$^\circ$]")
    ax.set_ylabel("Pressure [hPa]")

    cbar = fig.colorbar(pcm, ax=ax, extend="both")
    cbar.set_label(r"Heating Rate [K day$^{-1}$]")

    plt.savefig(
        save_path, dpi=300
    )  # 300 is usually plenty for journals; 600 is very heavy
    plt.close(fig)


# =============================================================================
# Main Function
# =============================================================================


def main(minkn: float, maxkn: float, ds_mean: xr.Dataset) -> None:
    # ------------------------------------------
    # Load Files
    # ------------------------------------------

    # File Path
    IdxPath: Path = Path(f"/work/b11209013/KWComposite/KWIdx/{minkn}_{maxkn}.csv")
    LWFigPath: Path = Path(
        f"/home/b11209013/2025_Research/Obs/Figure/KWComposite/LW_{minkn}_{maxkn}.png"
    )
    SWFigPath: Path = Path(
        f"/home/b11209013/2025_Research/Obs/Figure/KWComposite/SW_{minkn}_{maxkn}.png"
    )

    # Load Events
    Idx: pd.DataFrame = pd.read_csv(IdxPath)

    # Load CloudSat data
    ## Select time indices
    subset: xr.Dataset = ds_mean.isel(time=Idx["time_idx"].values)
    QlwAnom: xr.DataArray = subset["qlw"] - subset["qlw"].mean("lon", skipna=True)
    QswAnom: xr.DataArray = subset["qsw"] - subset["qsw"].mean("lon", skipna=True)

    Nt, Nz, Nx = QlwAnom.shape
    NxHalf: int = int(Nx // 2)

    # ------------------------------------------
    # Composite
    # ------------------------------------------

    # Roll qlw and qsw
    # Centering with lon_idx

    ## setup shifts
    shifts: np.ndarray = NxHalf - np.array(Idx["lon_idx"].values)

    ## Roll qlw and qsw
    QlwRoll: List[xr.DataArray] = [
        QlwAnom.isel(time=i).roll(lon=int(s), roll_coords=False)
        for i, s in enumerate(shifts)
    ]

    QswRoll: List[xr.DataArray] = [
        QswAnom.isel(time=i).roll(lon=int(s), roll_coords=False)
        for i, s in enumerate(shifts)
    ]

    # Composite qlw and qsw
    QlwComp: np.ndarray = xr.concat(QlwRoll, dim="time").mean("time").values
    QswComp: np.ndarray = xr.concat(QswRoll, dim="time").mean("time").values

    # Running mean the profile
    RunningWindow: int = int(10 / 0.625)

    QlwComp = convolve1d(QlwComp, np.ones(RunningWindow) / RunningWindow, axis=-1)
    QswComp = convolve1d(QswComp, np.ones(RunningWindow) / RunningWindow, axis=-1)

    # ------------------------------------------
    # Visulization
    # ------------------------------------------
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    render_plot(QlwComp, np.arange(1000, 100 - 25, -25), LWFigPath)
    render_plot(QswComp, np.arange(1000, 100 - 25, -25), SWFigPath)


# =============================================================================
# Execute Main Function
# =============================================================================
if __name__ == "__main__":
    # ------------------------------------------
    # Load CloudSat Data
    # ------------------------------------------
    CloudSatPath: Path = Path("/work/DATA/Satellite/CloudSat/CloudSat_filtered.nc")

    with xr.open_dataset(CloudSatPath, chunks={}) as ds:
        ds_mean: xr.Dataset = ds.sel(lat=slice(-15, 15)).mean("lat", skipna=True)

        Minkn: List[float] = [1, 3, 5, 7, 9, 11, 13, 1, 1]
        Maxkn: List[float] = [3, 5, 7, 9, 11, 13, 15, 15, 11]

        for i in tqdm(range(len(Minkn)), desc=f"Progrssing {CloudSatPath.name}"):
            main(Minkn[i], Maxkn[i], ds_mean)
