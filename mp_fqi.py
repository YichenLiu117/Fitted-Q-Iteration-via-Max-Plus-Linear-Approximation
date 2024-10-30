import jax.numpy as np
import jax
import time
jax.config.update("jax_enable_x64", True)

def fit_mpfqi(F, G, gamma, epsilon_theta=0.001, n_iterations=1000):

    p = F.shape[1]
    iter = 0
    delta_theta_FQI_max = []
    #theta_FQI_max = np.full(p, -np.inf)
    theta_FQI_max = np.zeros(p)
    #theta_FQI_max = np.full(p,10)
    t = time.perf_counter()
    theta0 = []
    Q0 = []
    temp_delta_theta = 1000
    theta_trajectory = [0]
    while temp_delta_theta >= epsilon_theta:
        Q = max_plus_product_optimized(G, (gamma*theta_FQI_max).reshape(-1,1))
        theta_plus = -max_plus_product_optimized(F.T, -Q)
        temp_delta_theta = np.max(np.abs(theta_FQI_max - theta_plus))/(np.abs(theta_FQI_max.max())+0.001)
        theta_FQI_max = theta_plus
        theta_trajectory.append(temp_delta_theta)
        iter += 1
    run_time = time.perf_counter() - t   # Iteration time

    #return run_time_comp, run_time_iter, delta_theta_FQI_max, theta_FQI_max, iter, F, F_plus, G, Q0, theta0
    return run_time, delta_theta_FQI_max, theta_FQI_max, iter, theta_trajectory