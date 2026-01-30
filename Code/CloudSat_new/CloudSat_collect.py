# =============================================
# Collect CloudSat Veriables
# =============================================

# =============================================
# Import modules
# =============================================

import os
import sys
import h5py
import numpy as np
from glob import glob
from pathlib import Path
from typing import Tuple, Dict, Any, List
from pyhdf.HC import HC
from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF
from pyhdf.VS import VS

from pprint import pprint
from tqdm import tqdm
from datetime import datetime


# =============================================
# Helper functions
# =============================================
def load_file(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load CloudSat HDF4 file and return two dictionaries.
    """

    # Open file
    hdf: HDF = HDF(path, SDC.READ)
    sd: SD = SD(path, SDC.READ)
    vs: VS = hdf.vstart()

    # ------------------------------------------
    # Load vdata
    # ------------------------------------------
    vs_dict: dict[str, np.ndarray] = dict()  # dictionary for saving vdata

    for info in vs.vdatainfo():
        vname: str = info[0]
        ref_num: int = info[2]

        vd: Any = vs.attach(vname)
        vs_dict[vname] = np.array(vd[:])
        vd.detach()
    vs.end()

    # ------------------------------------------
    # Load Scientific Datasets
    # ------------------------------------------
    var_dict: dict[str, np.ndarray] = {}
    for name in sd.datasets():
        sds = sd.select(name)
        var_dict[name] = sds.get()
        sds.endaccess()

    # Close file
    sd.end()
    hdf.close()
    return vs_dict, var_dict


def save_as_h5(data_list: List[Tuple[Dict, Dict]], out_path: str) -> None:
    """Save the extracted dictionaries into a structured HDF5 file."""
    with h5py.File(out_path, "w") as h5:
        for i, (vs_dict, var_dict) in enumerate(data_list):
            # Create a unique group for each file
            granule_grp = h5.create_group(f"Granule_{i:02d}")

            v_grp = granule_grp.create_group("Vdata")
            for key, val in vs_dict.items():
                v_grp.create_dataset(key, data=val, compression="gzip", shuffle=True)

            sd_grp = granule_grp.create_group("Scientific_Datasets")
            for key, val in var_dict.items():
                sd_grp.create_dataset(key, data=val, compression="gzip", shuffle=True)

    # print(f"--- Saved: {os.path.basename(out_path)} ---")


# =============================================
# Main function
# =============================================
def main(
    year: int,
    date: int,
) -> None:

    # ------------------------------------------
    # Load files
    # ------------------------------------------

    # set file path
    WORK: str = "/work/DATA/Satellite/CloudSat"
    OUTPUT: str = "/data92/b11209013/CloudSat_tmp"
    FPATH: str = f"{WORK}/{year}/{date:03d}"

    os.makedirs(OUTPUT, exist_ok=True)  # ensure the existence of OUTPUT directory

    # Collect files
    Files: list[str] = list(glob(f"{FPATH}/*.hdf"))

    vs_var_list = []

    for f in Files:
        vs_var = load_file(f)

        vs_var_list.append(vs_var)

    # save file
    save_as_h5(vs_var_list, f"{OUTPUT}/{year:04d}-{date:03d}.h5")


if __name__ == "__main__":
    # ------------------------------------------
    # import argument
    # ------------------------------------------
    if len(sys.argv) == 3:
        input_year = int(sys.argv[1])
        input_date = int(sys.argv[2])
        main(year=input_year, date=input_date)
