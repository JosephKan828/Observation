from joblib import Parallel, delayed
import numpy as np
import os
import json
import netCDF4 as nc
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt

def vertical_interp(data, height, z_levels):
    out = np.full((z_levels.shape[0], z_levels.shape[1]), np.nan, dtype=np.float32)
    for i in range(z_levels.shape[1]):
        valid = ~np.isnan(data[:, i]) & ~np.isnan(height[:, i])
        if np.any(valid):
            h = height[valid, i]
            v = data[valid, i]
            sort_idx = np.argsort(h)
            f = interp1d(h[sort_idx], v[sort_idx], bounds_error=False, fill_value=np.nan)
            out[:, i] = f(z_levels[:, i])
    return out

def regrid_with_kdtree(swath_lats, swath_lons, swath_data, grid_lats, grid_lons, radius_deg=0.5):
    grid_points = np.column_stack((grid_lats.ravel(), grid_lons.ravel()))
    valid = ~np.isnan(swath_data) & ~np.isnan(swath_lats) & ~np.isnan(swath_lons)
    swath_points = np.column_stack((swath_lats[valid], swath_lons[valid]))
    swath_values = swath_data[valid]
    tree = cKDTree(swath_points)
    neighbors = tree.query_ball_point(grid_points, r=radius_deg)
    result = np.full(grid_points.shape[0], np.nan, dtype=np.float32)
    for i, idxs in enumerate(neighbors):
        if idxs:
            result[i] = np.mean(swath_values[idxs])
    return result.reshape(grid_lats.shape)

def process_one_day(year, date):
    try:
        input_path = f"/work/b11209013/2024_Research/CloudSat/Stage1/{year:04d}_{date:03d}.json"
        if not os.path.exists(input_path):
            return

        with open(input_path, "rb") as f:
            data = json.load(f)
            
        data = {key: np.array(value) for key, value in data.items()}
        lat, lon, hgt, qlw, qsw = data.items()
        lats, lons = lat[1], lon[1]

        era_path = f"/work/b11209013/2024_Research/ERA5/z/z_{year:04d}.nc"
        with nc.Dataset(era_path, "r") as z_dataset:
            era_lats = z_dataset.variables["lat"][:]
            era_lons = z_dataset.variables["lon"][:]
            grid_lon2d, grid_lat2d = np.meshgrid(era_lons, era_lats)
            z = z_dataset["z"][date - 1] / 9.81

        original_lev = np.array([1000.0, 925.0, 850.0, 700.0, 500.0, 250.0, 200.0, 100.0])
        interp_lev = np.linspace(1000.0, 100.0, 38)
        z_itp = interp1d(original_lev, z, axis=0)(interp_lev).reshape(38, -1)

        native_levels = hgt[1].shape[1]
        grid_shape = (len(era_lats), len(era_lons))
        qlw_native = np.empty((native_levels,) + grid_shape, dtype=np.float32)
        qsw_native = np.empty((native_levels,) + grid_shape, dtype=np.float32)
        hgt_native = np.empty((native_levels,) + grid_shape, dtype=np.float32)

        for level in range(native_levels):
            qlw_native[level] = regrid_with_kdtree(
                lats.ravel(), lons.ravel(), qlw[1][:, level],
                grid_lat2d, grid_lon2d, radius_deg=0.5
            )
            qsw_native[level] = regrid_with_kdtree(
                lats.ravel(), lons.ravel(), qsw[1][:, level],
                grid_lat2d, grid_lon2d, radius_deg=0.5
            )
            hgt_native[level] = regrid_with_kdtree(
                lats.ravel(), lons.ravel(), hgt[1][:, level],
                grid_lat2d, grid_lon2d, radius_deg=0.5
            )

        qlw_vert = vertical_interp(qlw_native.reshape(native_levels, -1), hgt_native.reshape(native_levels, -1), z_itp)
        qsw_vert = vertical_interp(qsw_native.reshape(native_levels, -1), hgt_native.reshape(native_levels, -1), z_itp)

        qlw_vert = qlw_vert.reshape(38, *grid_shape)
        qsw_vert = qsw_vert.reshape(38, *grid_shape)

        output_dict = {
            "lon": era_lons.tolist(),
            "lat": era_lats.tolist(),
            "qlw": qlw_vert.tolist(),
            "qsw": qsw_vert.tolist(),
            "lev": interp_lev.tolist(),
        }

        output_path = f"/work/b11209013/2024_Research/CloudSat/Stage2/{year:04d}_{date:03d}.json"
        with open(output_path, "w") as f:
            json.dump(output_dict, f)

        print(f"{year:04d}_{date:03d} done")
    except Exception as e:
        print(f"Error processing {year:04d}_{date:03d}: {e}")

if __name__ == "__main__":
    years = np.arange(2006, 2018)
    dates = np.arange(1, 367)

    tasks = [(y, d) for y in years for d in dates]
    Parallel(n_jobs=8)(delayed(process_one_day)(y, d) for y, d in tasks)
