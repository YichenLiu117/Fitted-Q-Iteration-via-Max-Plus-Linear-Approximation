import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as np
from utils import extend_array_to_size, replace_nan_with_inf
from run_experiment import t_mp, t_v, t_fqi, t_mp_pwc, t_v_pwc, t_fqi_pwc

plt.plot(np.arange(1000), extend_array_to_size(np.array(t_mp[0][1:]), 1000), "go-", markevery=100, label='mp FQI')
plt.plot(np.arange(1000), extend_array_to_size(np.array(t_v[0][1:]), 1000), "b^-", markevery=100, label='v-mp FQI')
plt.plot(np.arange(1000), replace_nan_with_inf(extend_array_to_size(np.array(t_fqi[0][1:]), 1000)), "rx-", markevery=100, label='standard FQI')
plt.plot(np.arange(1000), extend_array_to_size(np.array(t_mp_pwc[0][1:]), 1000), "go--", markevery=100)
plt.plot(np.arange(1000), extend_array_to_size(np.array(t_v_pwc[0][1:]), 1000), "b^--", markevery=100)
plt.plot(np.arange(1000), extend_array_to_size(np.array(t_fqi_pwc[0][1:]), 1000), "rx--", markevery=100)
plt.yscale('log')
# plt.xscale('log')
plt.xlabel('iteration')
plt.ylabel('$|| \\theta - \\theta^+ ||_\\infty$')
plt.grid(True, which="both")
plt.legend(loc=1)
plt.gca().set_ylim(top=1e6, bottom=1e-4)
plt.savefig("convergence_tr.pdf", format="pdf", bbox_inches="tight")

plt.show()