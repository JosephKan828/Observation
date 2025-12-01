import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("prec_wo_cloud_kw_k_1_3_lon=180.h5", "r") as f:
    q_lw1 = np.array(f.get("q_lw_reg"))

with h5py.File("prec_wo_cloud_kw_k_1_3_lon=100.h5", "r") as f:
    q_lw2 = np.array(f.get("q_lw_reg"))


with h5py.File("prec_wo_cloud_kw_k_1_3_lon=260.h5", "r") as f:
    q_lw3 = np.array(f.get("q_lw_reg"))


plt.pcolormesh(q_lw1)
plt.colorbar()
plt.show()


plt.pcolormesh(q_lw2)
plt.colorbar()
plt.show()

plt.pcolormesh(q_lw3)
plt.colorbar()
plt.show()

