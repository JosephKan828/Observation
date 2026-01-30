# ======================================================================
# Organize CloudSat data to ERA5 grid and interpolate to specific levels
# ======================================================================

# ======================================================================
# Import package
# ======================================================================

import sys
import h5py
import numpy as np
import netCDF4 as nc
import pandas as pd

from matplotlib import pyplot as plt
from pprint import pprint
from typing import Tuple, Dict, List, Any, LiteralString
from pathlib import Path
from scipy.spatial import cKDTree  # type: ignore
from scipy.interpolate import interp1d

# ======================================================================
# Helper Functions
# ======================================================================


def Load_CloudSat(
    file: Path,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:

    Daily_SD: Dict[str, Dict[str, np.ndarray]] = {}
    Daily_VD: Dict[str, Dict[str, np.ndarray]] = {}

    with h5py.File(file, "r") as f:
        SwathKey: List[str] = list(f.keys())  # key list for each swath

        for key in SwathKey:
            # preallocate dictionary for saving data
            Daily_SD[key] = {}
            Daily_VD[key] = {}

            # Specify Keys
            GranuleGrp: Any = f.get(key)

            SDKeys: List[Any] = list(GranuleGrp["Scientific_Datasets"].keys())
            VDKeys: List[Any] = list(GranuleGrp["Vdata"].keys())

            for SDKey in SDKeys:
                Daily_SD[key][SDKey] = np.array(
                    GranuleGrp["Scientific_Datasets"][SDKey]
                )

            for VDKey in VDKeys:
                Daily_VD[key][VDKey] = np.array(GranuleGrp["Vdata"][VDKey])

    return Daily_SD, Daily_VD


def nearest_neighbors(
    tree: cKDTree,
    radius: float,
    grid_lons: np.ndarray,
    grid_lats: np.ndarray,
    swath_lons: np.ndarray,
    swath_lats: np.ndarray,
) -> pd.DataFrame:

    # check the shape of grid_lons and grid_lats to be 2D
    assert grid_lons.ndim == 2, "grid_lons must be a 2D array"
    assert grid_lats.ndim == 2, "grid_lats must be a 2D array"

    # flatten swath coordinates
    swath_lons = swath_lons.flatten()
    swath_lats = swath_lats.flatten()

    if np.any(swath_lons < 0):
        swath_lons[swath_lons < 0] += 360

    # acquire distance and indices
    dist, indices = tree.query(
        np.c_[swath_lats, swath_lons], distance_upper_bound=radius
    )

    # find valid indices where fall in the radius ball
    valid_mask: np.ndarray = indices < tree.n

    # calculate the actual index in 2d array
    iy, ix = np.unravel_index(indices[valid_mask], grid_lons.shape)

    # form Dataframe
    df: pd.DataFrame = pd.DataFrame(
        {
            "grid_idx": indices[valid_mask],
            "swath_sample_idx": np.where(valid_mask)[0],
            "iy": iy,
            "ix": ix,
        }
    )

    return df


def Regrid3D(
    RegridDF: Dict[str, pd.DataFrame],
    SwathData: Dict[str, np.ndarray],
    SwathZ: Dict[str, np.ndarray],
    GridZ: np.ndarray,
    LowerValue: float,
    UpperValue: float,
    MissingValue: float,
) -> np.ndarray:
    # preallocate nan array to save regrid file
    RegridData: np.ndarray = np.full((2, *GridZ.shape), np.nan)

    # loop for each swath
    SwathName: List[str] = list(RegridDF.keys())

    for SName in SwathName:
        SwathDF: pd.DataFrame = RegridDF[SName]

        # group by the nearest cell
        grouped: Any = SwathDF.groupby(["grid_idx"])

        # loop for each group
        for name, group in grouped:
            iy: np.ndarray = group["iy"].iloc[0]
            ix: np.ndarray = group["ix"].iloc[0]
            swath_idx: np.ndarray = group["swath_sample_idx"].values

            # Select subset for each group
            ZSubset: np.ndarray = np.array(SwathZ[SName][swath_idx]).astype(float)
            DataSubset: np.ndarray = np.array(SwathData[SName][:, swath_idx, :]).astype(
                float
            )
            if DataSubset.shape[-1] != ZSubset.shape[-1]:
                DataSubset: np.ndarray = DataSubset[..., :-1]

            ZRef: np.ndarray = np.array(GridZ[:, iy, ix])

            # Find valid level
            ## substitute missing value with nan
            ZMask: np.ndarray = (
                (ZSubset < LowerValue) | (ZSubset > UpperValue) | (ZSubset == -9999)
            )
            DMask: np.ndarray = (
                (DataSubset < LowerValue)
                | (DataSubset > UpperValue)
                | (DataSubset == MissingValue)
            )

            ZSubset[ZMask] = np.nan
            DataSubset[DMask] = np.nan

            # Interpolate
            ItpResults: List[List[np.ndarray]] = [[], []]

            for i in range(DataSubset.shape[1]):
                ZProfile: np.ndarray = ZSubset[i, :].squeeze()

                for var_idx in [0, 1]:
                    data_profile: np.ndarray = DataSubset[var_idx, i, :]

                    # masking missing value
                    mask: np.ndarray = ~np.isnan(ZProfile) & ~np.isnan(data_profile)

                    if np.any(mask) and np.sum(mask) > 1:
                        try:
                            f = interp1d(
                                ZProfile[mask],
                                data_profile[mask],
                                bounds_error=False,
                                fill_value=np.nan,
                            )
                            ItpResults[var_idx].append(f(ZRef))
                        except Exception:
                            ItpResults[var_idx].append(np.full_like(ZRef, np.nan))
                    else:
                        ItpResults[var_idx].append(np.full_like(ZRef, np.nan))

            if ItpResults[0]:
                try:
                    RegridData[0, :, iy, ix] = np.nanmean(ItpResults[0], axis=0)
                except Exception:
                    RegridData[0, :, iy, ix] = np.full_like(ZRef, np.nan)
            if ItpResults[1]:
                try:
                    RegridData[1, :, iy, ix] = np.nanmean(ItpResults[1], axis=0)
                except Exception:
                    RegridData[1, :, iy, ix] = np.full_like(ZRef, np.nan)
    return RegridData


def Regrid2D(
    RegridDF: Dict[str, pd.DataFrame],
    SwathData: Dict[str, np.ndarray],
    GridZ: np.ndarray,
    LowerValue: float,
    UpperValue: float,
    MissingValue: float,
) -> np.ndarray:
    # preallocate nan array to save regrid file
    RegridData: np.ndarray = np.full((2, *GridZ.shape[1:]), np.nan)

    # loop for each swath
    SwathName: List[str] = list(RegridDF.keys())

    for SName in SwathName:
        SwathDF: pd.DataFrame = RegridDF[SName]

        # group by the nearest cell
        grouped: Any = SwathDF.groupby(["grid_idx"])

        # loop for each group
        for name, group in grouped:
            iy: np.ndarray = group["iy"].iloc[0]
            ix: np.ndarray = group["ix"].iloc[0]
            swath_idx: np.ndarray = group["swath_sample_idx"].values

            # Select subset for each group
            DataSubset: np.ndarray = np.array(SwathData[SName][:, swath_idx]).astype(
                float
            )

            # Find valid level
            ## substitute missing value with nan
            DMask: np.ndarray = (
                (DataSubset < LowerValue)
                | (DataSubset > UpperValue)
                | (DataSubset == MissingValue)
            )
            DataSubset[DMask] = np.nan

            # Interpolate
            ItpResults: List[List[np.ndarray]] = [[], []]

            for var_idx in [0, 1]:
                mask: np.ndarray = ~np.isnan(DataSubset[var_idx, :])
                ItpResults[var_idx].append(DataSubset[var_idx, :][mask])

            if ItpResults[0]:
                try:
                    RegridData[0, iy, ix] = np.nanmean(ItpResults[0], axis=0)[0]
                except Exception:
                    RegridData[0, iy, ix] = np.nan
            if ItpResults[1]:
                try:
                    RegridData[1, iy, ix] = np.nanmean(ItpResults[1], axis=0)[0]
                except Exception:
                    RegridData[1, iy, ix] = np.nan

    return RegridData


def Regrid1D(
    RegridDF: Dict[str, pd.DataFrame],
    SwathData: Dict[str, np.ndarray],
    GridZ: np.ndarray,
    LowerValue: float,
    UpperValue: float,
    MissingValue: float,
) -> np.ndarray:
    # preallocate nan array to save regrid file
    RegridData: np.ndarray = np.full((*GridZ.shape[1:],), np.nan)

    # loop for each swath
    SwathName: List[str] = list(RegridDF.keys())

    for SName in SwathName:
        SwathDF: pd.DataFrame = RegridDF[SName]

        # group by the nearest cell
        grouped: Any = SwathDF.groupby(["grid_idx"])

        # loop for each group
        for name, group in grouped:
            iy: np.ndarray = group["iy"].iloc[0]
            ix: np.ndarray = group["ix"].iloc[0]
            swath_idx: np.ndarray = group["swath_sample_idx"].values

            # Select subset for each group
            DataSubset: np.ndarray = np.array(SwathData[SName][swath_idx]).astype(float)

            # Find valid level
            ## substitute missing value with nan
            DMask: np.ndarray = (
                (DataSubset < LowerValue)
                | (DataSubset > UpperValue)
                | (DataSubset == MissingValue)
            )
            DataSubset[DMask] = np.nan

            mask = ~np.isnan(DataSubset)
            DataSubset = DataSubset[mask]

            try:
                RegridData[iy, ix] = np.nanmean(DataSubset, axis=0)[0]
            except Exception:
                RegridData[iy, ix] = np.nan

    return RegridData


# ======================================================================
# Main function
# ======================================================================
def main(year: int, date: int) -> None:

    # -----------------------------------------------------------------
    # Load Data
    # -----------------------------------------------------------------

    # File path
    CloudSat_PATH: Path = Path(
        f"/data92/b11209013/CloudSat_tmp/{year:04d}-{date:03d}.h5"
    )
    ERA5_PATH: Path = Path(f"/work/b11209013/2024_Research/ERA5/z/z_{year:04d}.nc")

    # Load ERA5 data
    with nc.Dataset(ERA5_PATH, "r") as z_dataset:
        lon: np.ndarray = z_dataset.variables["lon"][:].squeeze()
        lat: np.ndarray = z_dataset.variables["lat"][:].squeeze()
        z: np.ndarray = z_dataset.variables["z"][date - 1] / 9.81  # shape: (nz, ny, nx)

    # Load CloudSat data
    Daily_SD, Daily_VD = Load_CloudSat(CloudSat_PATH)
    SDVarList: List[str] = list(Daily_SD["Granule_00"].keys())
    VDVarList: List[str] = list(Daily_VD["Granule_00"].keys())

    # Categorize variables
    VarDict: Dict[str, List[str]] = {
        "3DRegrid": ["COD", "FD", "FD_NA", "FD_NC", "FU", "FU_NA", "FU_NC", "QR"],
        "2DRegrid": ["BOACRE", "FU_NA_TOA", "FU_NC_TOA", "FU_TOA", "RH", "TOACRE"],
        "1DRegrid": ["FD_TOA_IncomingSolar", "Albedo"],
    }
    LowerDict: Dict[str, List[float]] = {
        "3DRegrid": [0, 0, 0, 0, 0, 0, 0, -20000],
        "2DRegrid": [0, 0, 0, 0, -1000, 0],
        "1DRegrid": [0, 0],
    }

    UpperDict: Dict[str, List[float]] = {
        "3DRegrid": [500000000, 15000, 15000, 15000, 15000, 15000, 15000, 20000],
        "2DRegrid": [1500, 15000, 15000, 15000, 1000, 1500],
        "1DRegrid": [15000, 10],
    }

    MissingDict: Dict[str, List[float]] = {
        "3DRegrid": [-9990, -9990, -9990, -9990, -9990, -9990, -9990, -9999],
        "2DRegrid": [-9990, -9990, -9990, -9990, -9999, -9990],
        "1DRegrid": [-9990, -9990],
    }
    # -----------------------------------------------------------------
    # Build up cKDTree and regrid
    # -----------------------------------------------------------------
    # make mesh
    mesh: List[np.ndarray] = np.meshgrid(lon, lat)

    lon_2d: np.ndarray = mesh[0]
    lat_2d: np.ndarray = mesh[1]

    # build up tree
    grid_points: np.ndarray = np.c_[lat_2d.ravel(), lon_2d.ravel()]
    tree: cKDTree = cKDTree(grid_points)

    # radius of tree
    radius: float = 0.4

    # Construct dataframe for each swath
    df_dict: Dict[str, pd.DataFrame] = {
        key: nearest_neighbors(
            tree,
            radius,
            grid_lons=lon_2d,
            grid_lats=lat_2d,
            swath_lons=Daily_VD[key]["Longitude"],
            swath_lats=Daily_VD[key]["Latitude"],
        )
        for key in Daily_SD.keys()
    }

    # -----------------------------------------------------------------
    # Calculate mean profile
    # -----------------------------------------------------------------

    # Dictionary for regrid data
    RegridDict: Dict[str, np.ndarray] = {}

    # For 3d data
    HeightDict = {key: Daily_SD[key]["Height"] for key in Daily_SD.keys()}

    for i, Vkey in enumerate(VarDict["3DRegrid"]):
        pprint("Regridding " + Vkey)

        if Vkey in SDVarList:
            DataDict: Dict[str, np.ndarray] = {
                SWKey: Daily_SD[SWKey][Vkey] for SWKey in Daily_SD.keys()
            }
        elif Vkey in VDVarList:
            DataDict: Dict[str, np.ndarray] = {
                SWKey: Daily_VD[SWKey][Vkey] for SWKey in Daily_VD.keys()
            }

        RegridDict[Vkey] = Regrid3D(
            RegridDF=df_dict,
            SwathData=DataDict,
            SwathZ=HeightDict,
            GridZ=z,
            LowerValue=LowerDict["3DRegrid"][i],
            UpperValue=UpperDict["3DRegrid"][i],
            MissingValue=MissingDict["3DRegrid"][i],
        )

        pprint("Finished regridding " + Vkey)

    # For 2d data
    for i, Vkey in enumerate(VarDict["2DRegrid"]):
        pprint("Regridding " + Vkey)

        if Vkey in SDVarList:
            DataDict: Dict[str, np.ndarray] = {
                SWKey: Daily_SD[SWKey][Vkey] for SWKey in Daily_SD.keys()
            }
        elif Vkey in VDVarList:
            DataDict: Dict[str, np.ndarray] = {
                SWKey: Daily_VD[SWKey][Vkey] for SWKey in Daily_VD.keys()
            }

        RegridDict[Vkey] = Regrid2D(
            RegridDF=df_dict,
            SwathData=DataDict,
            GridZ=z,
            LowerValue=LowerDict["2DRegrid"][i],
            UpperValue=UpperDict["2DRegrid"][i],
            MissingValue=MissingDict["2DRegrid"][i],
        )

        pprint("Finished regridding " + Vkey)

    # For 1d data
    for i, Vkey in enumerate(VarDict["1DRegrid"]):
        pprint("Regridding " + Vkey)

        if Vkey in SDVarList:
            DataDict: Dict[str, np.ndarray] = {
                SWKey: Daily_SD[SWKey][Vkey] for SWKey in Daily_SD.keys()
            }
        elif Vkey in VDVarList:
            DataDict: Dict[str, np.ndarray] = {
                SWKey: Daily_VD[SWKey][Vkey] for SWKey in Daily_VD.keys()
            }

        RegridDict[Vkey] = Regrid1D(
            RegridDF=df_dict,
            SwathData=DataDict,
            GridZ=z,
            LowerValue=LowerDict["1DRegrid"][i],
            UpperValue=UpperDict["1DRegrid"][i],
            MissingValue=MissingDict["1DRegrid"][i],
        )

        pprint("Finished regridding " + Vkey)

    # -----------------------------------------------------------------
    # Save data
    # -----------------------------------------------------------------
    output_path = Path(f"/data92/b11209013/CloudSat_regrid/{year:04d}-{date:03d}.nc")

    with nc.Dataset(output_path, "w", format="NETCDF4") as f:
        # 1. Create Dimensions
        f.createDimension("time", None)  # Unlimited
        f.createDimension("level", z.shape[0])
        f.createDimension("lat", lat.shape[0])
        f.createDimension("lon", lon.shape[0])
        f.createDimension("band", 2)  # 0 for SW, 1 for LW

        # 2. Create Coordinate Variables
        times = f.createVariable("time", "f8", ("time",))
        levels = f.createVariable("level", "f4", ("level",))
        lats = f.createVariable("lat", "f4", ("lat",))
        lons = f.createVariable("lon", "f4", ("lon",))
        bands = f.createVariable(
            "band", "i4", ("band",)
        )  # Integer type for coordinates

        # 3. Assign Attributes and Units
        lons.units = "degrees_east"
        lats.units = "degrees_north"
        levels.units = "hPa"
        levels.long_name = "Pressure Levels"
        times.units = f"days since {year:04d}-01-01 00:00:00"
        times.calendar = "gregorian"
        bands.long_name = "Radiation Band"
        bands.comment = "0: Shortwave (SW), 1: Longwave (LW)"

        # 4. Fill Coordinate Data
        lons[:] = lon
        lats[:] = lat
        levels[:] = np.linspace(1000, 100, 37)
        times[0] = float(date - 1)
        bands[:] = [0, 1]  # SW=0, LW=1

        # 5. Create and Save Data Variables
        for key, data in RegridDict.items():
            # Handle 3D Data: shape (band=2, level=37, lat, lon)
            if data.ndim == 4:
                v = f.createVariable(
                    key, "f4", ("band", "time", "level", "lat", "lon"), zlib=True
                )
                # data is (2, 37, ny, nx), we insert time at index 1
                v[:, 0, :, :, :] = data

            # Handle 2D Data: shape (band=2, lat, lon)
            elif data.ndim == 3:
                v = f.createVariable(
                    key, "f4", ("band", "time", "lat", "lon"), zlib=True
                )
                # data is (2, ny, nx), we insert time at index 1
                v[:, 0, :, :] = data

            # Handle 1D Data: shape (lat, lon)
            elif data.ndim == 2:
                v = f.createVariable(key, "f4", ("time", "lat", "lon"), zlib=True)
                # data is (ny, nx), we insert time at index 0
                v[0, :, :] = data

    pprint(f"Successfully saved to {output_path}")
    return None


# ======================================================================
# Execute main function
# ======================================================================

if __name__ == "__main__":
    FileName: str = sys.argv[1]

    year_date: List[str] = FileName.split(".")[0].split("-")

    year: int = int(year_date[0])
    date: int = int(year_date[1])

    main(year, date)
