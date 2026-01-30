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
def render_plot(
    CSData: np.ndarray,
    QData: np.ndarray,
    TData: np.ndarray,
    WData: np.ndarray,
    levels: np.ndarray,
    save_path: Path,
):
    # Calculate levels for contourf
    ## Find Maximum
    QMax: float = np.nanmax(np.abs(QData * 1000.0)) // 0.1
    TMax: float = np.nanmax(np.abs(TData * 1.0)) // 0.1
    WMax: float = np.nanmax(np.abs(WData * -100.0)) // 0.5

    ## Set levels
    QLevels: np.ndarray = np.arange(-QMax, QMax + 0.2, 0.2)
    TLevels: np.ndarray = np.arange(-TMax, TMax + 0.2, 0.2)
    WLevels: np.ndarray = np.arange(-WMax, WMax + 0.6, 0.6)

    ## Omit zeros
    QLevels = QLevels[QLevels != 0.0]
    TLevels = TLevels[TLevels != 0.0]
    WLevels = WLevels[WLevels != 0.0]

    # Create Figure
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    pcm = ax.pcolormesh(
        np.arange(-180, 180, 0.625),
        levels,
        CSData,
        cmap="RdBu_r",
        shading="auto",
        rasterized=True,
        norm=TwoSlopeNorm(vcenter=0),
    )
    ctq = ax.contour(
        np.arange(-180, 180, 0.625),
        levels,
        QData * 1000.0,
        colors="seagreen",
        levels=QLevels,
        linewidths=1.5,
    )
    ctt = ax.contour(
        np.arange(-180, 180, 0.625),
        levels,
        TData * 1.0,
        colors="darkviolet",
        levels=TLevels,
        linewidths=1.5,
    )
    ctw = ax.contour(
        np.arange(-180, 180, 0.625),
        levels,
        WData * -100.0,
        colors="black",
        levels=WLevels,
        linewidths=1.5,
    )

    ax.set_xlim(-100, 100)
    ax.set_ylim(1000, 100)
    ax.set_xlabel(r"Relative Longitude [$^\circ$]")
    ax.set_ylabel("Pressure [hPa]")

    ax.clabel(ctq, inline=True, fontsize=12)
    ax.clabel(ctt, inline=True, fontsize=12)
    ax.clabel(ctw, inline=True, fontsize=12)
    cbar = fig.colorbar(pcm, ax=ax, extend="both")
    cbar.set_label(r"Heating Rate [K day$^{-1}$]")

    plt.savefig(
        save_path, dpi=300
    )  # 300 is usually plenty for journals; 600 is very heavy
    plt.close(fig)


# =============================================================================
# Main Function
# =============================================================================


def main(
    minkn: float,
    maxkn: float,
    CSDataset: xr.Dataset,
    QDataset: xr.Dataset,
    TDataset: xr.Dataset,
    WDataset: xr.Dataset,
) -> None:
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

    # Select Data
    CSSubset: xr.Dataset = CSDataset.isel(time=Idx["time_idx"].values)
    QSubset: xr.Dataset = QDataset.isel(time=Idx["time_idx"].values)
    TSubset: xr.Dataset = TDataset.isel(time=Idx["time_idx"].values)
    WSubset: xr.Dataset = WDataset.isel(time=Idx["time_idx"].values)

    # Calculate anomalous data
    QlwAnom: xr.DataArray = CSSubset["qlw"] - CSSubset["qlw"].mean("lon", skipna=True)
    QswAnom: xr.DataArray = CSSubset["qsw"] - CSSubset["qsw"].mean("lon", skipna=True)
    QAnom: xr.DataArray = QSubset["q"] - QSubset["q"].mean("lon", skipna=True)
    TAnom: xr.DataArray = TSubset["t"] - TSubset["t"].mean("lon", skipna=True)
    WAnom: xr.DataArray = WSubset["w"] - WSubset["w"].mean("lon", skipna=True)

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
    QRoll: List[xr.DataArray] = [
        QAnom.isel(time=i).roll(lon=int(s), roll_coords=False)
        for i, s in enumerate(shifts)
    ]

    TRoll: List[xr.DataArray] = [
        TAnom.isel(time=i).roll(lon=int(s), roll_coords=False)
        for i, s in enumerate(shifts)
    ]

    WRoll: List[xr.DataArray] = [
        WAnom.isel(time=i).roll(lon=int(s), roll_coords=False)
        for i, s in enumerate(shifts)
    ]

    # Composite qlw and qsw
    QlwComp: np.ndarray = xr.concat(QlwRoll, dim="time").mean("time").values
    QswComp: np.ndarray = xr.concat(QswRoll, dim="time").mean("time").values
    QComp: np.ndarray = xr.concat(QRoll, dim="time").mean("time").values
    TComp: np.ndarray = xr.concat(TRoll, dim="time").mean("time").values
    WComp: np.ndarray = xr.concat(WRoll, dim="time").mean("time").values

    # Running mean the profile
    RunningWindow: int = int(10 / 0.625)

    QlwComp = convolve1d(QlwComp, np.ones(RunningWindow) / RunningWindow, axis=-1)
    QswComp = convolve1d(QswComp, np.ones(RunningWindow) / RunningWindow, axis=-1)

    QComp = convolve1d(QComp, np.ones(RunningWindow) / RunningWindow, axis=-1)
    TComp = convolve1d(TComp, np.ones(RunningWindow) / RunningWindow, axis=-1)
    WComp = convolve1d(WComp, np.ones(RunningWindow) / RunningWindow, axis=-1)

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

    render_plot(QlwComp, QComp, TComp, WComp, np.arange(1000, 100 - 25, -25), LWFigPath)
    render_plot(QswComp, QComp, TComp, WComp, np.arange(1000, 100 - 25, -25), SWFigPath)


# =============================================================================
# Execute Main Function
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------
    # Load Data
    # ------------------------------------------
    # Setup File Path
    CloudSatPath: Path = Path("/work/DATA/Satellite/CloudSat/CloudSat_filtered.nc")
    ERA5qPath: Path = Path("/work/b11209013/2024_Research/ERA5/q/q_sub.nc")
    ERA5tPath: Path = Path("/work/b11209013/2024_Research/ERA5/t/t_sub.nc")
    ERA5wPath: Path = Path("/work/b11209013/2024_Research/ERA5/w/w_Itp_sub.nc")

    # Load CloudSat
    with xr.open_dataset(CloudSatPath, chunks={}) as ds:
        CSDataset: xr.Dataset = ds.sel(lat=slice(-15, 15)).mean("lat", skipna=True)

    # Load q data
    with xr.open_dataset(ERA5qPath, chunks={}) as ds:
        QDataset: xr.Dataset = ds

    # Load t data
    with xr.open_dataset(ERA5tPath, chunks={}) as ds:
        TDataset: xr.Dataset = ds

    # Load w data
    with xr.open_dataset(ERA5wPath, chunks={}) as ds:
        WDataset: xr.Dataset = ds.mean("lat", skipna=True)

    Minkn: List[float] = [1, 3, 5, 7, 9, 11, 13, 1, 1]
    Maxkn: List[float] = [3, 5, 7, 9, 11, 13, 15, 15, 11]

    for i in tqdm(range(len(Minkn)), desc=f"Progrssing {CloudSatPath.name}"):
        main(Minkn[i], Maxkn[i], CSDataset, QDataset, TDataset, WDataset)
