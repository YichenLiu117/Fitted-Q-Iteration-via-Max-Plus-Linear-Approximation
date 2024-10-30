import jax.numpy as np
import jax
import time
jax.config.update("jax_enable_x64", True)

def fit_v_mpfqi(Fs, Gs, gamma, epsilon_theta=0.001, n_iterations=1000):

    n = F.shape[0]
    p = F.shape[1]
    p_v = env.action_space.n
    p_y = F.shape[1]/p_v

    iter = 0
    delta_theta_FQI_max = []
    theta_FQI_max = np.zeros(p)
    t = time.perf_counter()
    theta0 = []
    Q0 = []
    temp_delta_theta = 1000
    theta_trajectory = [0]
    while temp_delta_theta >= epsilon_theta:
        Q = max_plus_product_optimized(Gs, (gamma*theta_FQI_max).reshape(-1,1))
        theta_plus = -max_plus_product_optimized(Fs.T, -Q)
        temp_delta_theta = np.max(np.abs(theta_FQI_max - theta_plus))/(np.abs(theta_FQI_max.max())+0.001)
        theta_FQI_max = theta_plus
        if temp_delta_theta <= epsilon_theta:
            break
        theta_trajectory.append(temp_delta_theta)
        iter += 1
    run_time = time.perf_counter() - t  # Iteration time
    return run_time, delta_theta_FQI_max, theta_FQI_max, iter, theta_trajectory