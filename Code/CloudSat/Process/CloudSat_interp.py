# This program is to regrid data to the same grid as ERA5
import json
import numpy as np
import netCDF4 as nc

from glob import glob
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt

def categorize(data_cs, lon_era, lat_era):
    grid_points = np.column_stack((lat_era.ravel(), lon_era.ravel()))
    
    swath_points = np.column_stack((data_cs["lat"], data_cs["lon"]))
    
    tree = cKDTree(grid_points)
    
    

def process_one_day(fname):
    # load CloudSat data
    date = int(fname.split("_")[-1].split(".")[0])-1
    
    with open(fname, "rb") as f:
        data = json.load(f)
        
    data = {key: np.array(data[key]) for key in data.keys()}
    
    # load ERA5 z data
    era_path = f"/work/b11209013/2024_Research/ERA5/z/z_2006.nc"
    with nc.Dataset(era_path, "r") as z_dataset:
        era_lats = z_dataset.variables["lat"][:]
        era_lons = z_dataset.variables["lon"][:]
        grid_lon2d, grid_lat2d = np.meshgrid(era_lons, era_lats)
        z = z_dataset["z"][date] / 9.81



def main():
    # ==== 1. Load data ==== #
    fpath = "/work/b11209013/2024_Research/CloudSat/Stage1/"
    fcollect = glob(fpath+"*.json")
    
    process_one_day(fcollect[0])
    
if __name__ == "__main__":
    main()