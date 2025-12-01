import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("prec_wo_cloud_kw_k_11_13_lon=180.h5", "r") as f:
    lw = np.array(f.get("lw_reg"))
    sw = np.array(f.get("sw_reg"))

plt.pcolormesh(np.linspace(-180, 180, 576), np.linspace(1000, 100, 37), lw)
plt.colorbar()
plt.savefig("test.png")
