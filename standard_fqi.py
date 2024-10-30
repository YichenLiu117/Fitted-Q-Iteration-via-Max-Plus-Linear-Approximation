import jax.numpy as np
import jax
import time
jax.config.update("jax_enable_x64", True)

def fit_rgfqi(F_star, F, M, gamma, epsilon_theta=0.001, n_iterations=1000):
    p_y = Y.shape[1]
    p_v = env.action_space.n
    p = p_y * p_v
    

    #F_star, F = parallel_compute_F_Fplus(n, p_y, p_v, states, actions, next_states, Y, V, C)
    theta_kernel_max = np.zeros(p)
    delta_theta_kernel_max = []
    F = np.array(F)
    M = np.array(M)
    F_star = np.array(F_star)
    iter = 0
    t = time.perf_counter()
    theta_trajectory = [0]
    while iter < n_iterations:
        theta_kernel_reshaped = theta_kernel_max.reshape(p_y, p_v)
        # Vectorized computation
        Q_x = F_star @ theta_kernel_reshaped # Resulting shape is (n, p_v)
        # Find the max over the second axis (p_v) for each set of computations
        Q_xplus = np.max(Q_x, axis=1)
        g = rewards + gamma * Q_xplus
        theta_plus = M @ g
        temp_delta_theta = np.max(np.abs(theta_kernel_max - theta_plus))/(theta_kernel_max.max()+0.001)
        theta_kernel_max = theta_plus
        if temp_delta_theta <= epsilon_theta:
            break
        theta_trajectory.append(temp_delta_theta)
        iter += 1

    run_time = time.perf_counter() - t  # Iteration time

    return run_time, delta_theta_kernel_max, theta_kernel_max, iter, theta_trajectory