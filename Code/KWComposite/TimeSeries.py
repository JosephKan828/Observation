# ======================================================================
# Acquiring filtered time series of IMERG data
# ======================================================================

# ======================================================================
# Import modules
# ======================================================================
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from typing import List, Dict, Any, Tuple
from pprint import pprint
from scipy import ndimage

sys.path.append("/home/b11209013/demo_code/")
import SpaceTimeReconstruct as ST  # type: ignore

# ======================================================================
# Main Function
# ======================================================================


def main(minkn: float, maxkn: float) -> None:
    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------

    # File path
    IMERGPATH: Path = Path("/work/b11209013/2025_Research/IMERG/IMERG_06_17_lowpass.nc")
    OUTPUTPATH: Path = Path(f"/work/b11209013/KWComposite/KWIdx/{minkn}_{maxkn}.csv")

    # Load IMERG data
    with xr.open_dataset(IMERGPATH, chunks={}) as ds:
        ds = ds.sel(lat=slice(-15, 15))

        IMERGCoords: Any = ds.coords

        Prec: xr.DataArray = ds["precipitation"]

    # ----------------------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------------------

    # Calculate anomalous precipitation
    PrecAnom: np.ndarray = (Prec - Prec.mean({"time", "lon"})).values

    # Calculate symmetric precipitation
    PrecSym: np.ndarray = (PrecAnom + np.flip(PrecAnom, axis=1)) / 2

    # pre-allocate wavenumber and frequency
    wn: np.ndarray = np.fft.fftfreq(
        IMERGCoords["lon"].size, d=1 / IMERGCoords["lon"].size
    )
    fr: np.ndarray = np.fft.fftfreq(IMERGCoords["time"].size, d=1)

    FreqMesh: List[np.ndarray] = np.meshgrid(wn, fr)

    wnm: np.ndarray = FreqMesh[0]
    frm: np.ndarray = FreqMesh[1]

    # ----------------------------------------------------------------
    # Reconstruct
    # ----------------------------------------------------------------
    # setup filter
    filter: ST.SpaceTimeFilter = ST.SpaceTimeFilter(
        dispersion=ST.DispersionParams(
            n_planetary_wave=576,
            rlat=0.0,
            equivalent_depth=(8, 90),
            s_min=-288,
            s_max=287,
        ),
        bandpass=ST.BandpassParams(
            k_range=(minkn, maxkn), f_range=(1 / 20, 1 / 2.5), nan_to_inf=True
        ),
    )

    Recon, _ = filter.compute(
        data=PrecSym,
        fr_grid=frm,
        wn_grid=wnm,
        wave_type="Kelvin",
        return_mask=True,
    )

    Recon = Recon.squeeze()

    # ----------------------------------------------------------------
    # Statistical test
    # ----------------------------------------------------------------

    # Find data that surpass the 95% percentile
    Prec99: np.ndarray = np.nanpercentile(Recon.flatten(), 99)

    # Apply binary_closing to the data
    structure: np.ndarray = np.ones((3, 3))
    ReconMask: np.ndarray = (Recon > Prec99).astype(int)
    ClosedMask: np.ndarray = ndimage.binary_closing(ReconMask, structure=structure)

    # ----------------------------------------------------------------
    # Labeling KW events
    # ----------------------------------------------------------------
    labeled_array, num_features = ndimage.label(ClosedMask, structure=structure)  # type: ignore

    if num_features > 0:
        # Find the position of the maximum
        max_positions = ndimage.maximum_position(
            Recon, labels=labeled_array, index=np.arange(1, num_features + 1)
        )

        # Mapping indices to actual time and longitude
        times: np.ndarray = IMERGCoords["time"].values[[p[0] for p in max_positions]]
        lons: np.ndarray = IMERGCoords["lon"].values[[p[1] for p in max_positions]]

        df: pd.DataFrame = pd.DataFrame(
            {
                "time_idx": [p[0] for p in max_positions],
                "lon_idx": [p[1] for p in max_positions],
                "time": times,
                "lon": lons,
            }
        )

        df.to_csv(OUTPUTPATH, index=False)
        pprint(f"Saved {num_features} event indices to {OUTPUTPATH}")

    else:
        pprint(f"{OUTPUTPATH} is skipped")


# ======================================================================
# Execute main function
# ======================================================================

if __name__ == "__main__":
    # Minkn: List[float] = [1, 3, 5, 7, 9, 11, 13, 1]
    # Maxkn: List[float] = [3, 5, 7, 9, 11, 13, 15, 15]

    Minkn: List[float] = [1]
    Maxkn: List[float] = [11]

    for i in range(len(Minkn)):
        main(Minkn[i], Maxkn[i])
