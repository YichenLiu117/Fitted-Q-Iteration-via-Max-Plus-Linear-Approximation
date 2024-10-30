import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as np
from run_experiment import GRID_SIZE, mp_results, v_results, fqi_results, mp_pwc_results, v_pwc_results, fqi_pwc_results

plt.plot(GRID_SIZE**2, np.mean(np.array(mp_results), axis=1), "go-", label='MP-FQI')
plt.plot(GRID_SIZE**2, np.mean(np.array(v_results), axis=1), "b^-", label='v-MP-FQI')
plt.plot(GRID_SIZE**2, np.mean(np.array(fqi_results), axis=1), "rx-", label='standard FQI')
plt.plot(GRID_SIZE**2, np.mean(np.array(mp_pwc_results), axis=1), "go--")
plt.plot(GRID_SIZE**2, np.mean(np.array(v_pwc_results), axis=1), "b^--")
plt.plot(GRID_SIZE**2, np.mean(np.array(fqi_pwc_results), axis=1), "rx--")
plt.xlabel('number of basis functions')
plt.ylabel('scaled discounted reward')
#plt.title("Mean reward against optimal policy vs. number of basis functions per state space dimension")
plt.legend()
# plt.savefig("rew_basis.pdf", format="pdf", bbox_inches="tight")
plt.show()