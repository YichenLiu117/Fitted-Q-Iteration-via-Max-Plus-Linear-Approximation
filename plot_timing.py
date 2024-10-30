import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as np
from run_experiment import GRID_SIZE, comp_time_mp, comp_time_v, comp_time_fqi, iter_time_mp, iter_time_v, iter_time_fqi, iter_mp, iter_v, iter_fqi




plt.plot(GRID_SIZE**2, np.array(comp_time_mp) * (GRID_SIZE**2), 'go-', label='MP-FQI')
plt.plot(GRID_SIZE**2, np.array(comp_time_v) * (GRID_SIZE**2), 'b^-', label='v-MP-FQI')
plt.plot(GRID_SIZE**2, np.array(comp_time_fqi) * (GRID_SIZE**2), 'rx-', label='standard FQI')
plt.plot(GRID_SIZE**2, np.array(iter_time_mp)/np.array(iter_mp), 'go--')
plt.plot(GRID_SIZE**2, np.array(iter_time_v)/np.array(iter_v), 'b^--')
plt.plot(GRID_SIZE**2, np.array(iter_time_fqi)/np.array(iter_fqi), 'rx--')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('number of basis functions')
plt.ylabel('time (s)')
plt.grid(True, which="both")
plt.legend()
# plt.savefig("time_basis.pdf", format="pdf", bbox_inches="tight")
plt.show()

