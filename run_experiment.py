import os
from functools import partial
import time
# Set the number of threads before importing JAX and its modules
import jax.numpy as np
from jax import jit, vmap
import jax.random as jnd
from jax import lax
import jax
from utils import *
from data_gen import RANGE_MAT, states_actions, rewards, next_states, keys
from basis_compilation import *
from standard_fqi import fit_rgfqi
from mpfqi import fit_mpfqi
from v_mp_fqi import fit_v_mpfqi
jax.config.update("jax_enable_x64", True)


GRID_SIZE = np.arange(3, 52, 2)
EPI_LEN = 100
mp_results = []
v_results = []
fqi_results = []
mp_pwc_results = []
v_pwc_results = []
fqi_pwc_results = []
comp_time_mp = []
comp_time_v = []
comp_time_fqi = []
iter_time_mp = []
iter_time_v = []
iter_time_fqi = []
iter_mp = []
iter_v = []
iter_fqi = []
iter_mp_pwc = []
iter_v_pwc = []
iter_fqi_pwc = []
mp_theta = []
basis = []
t_mp = []
t_v = []
t_fqi = []
t_mp_pwc = []
t_v_pwc = []
t_fqi_pwc = []


for i in GRID_SIZE:
    RANG = np.ptp(RANGE_MAT, axis=1)/(i-1)
    COEF = 1
    C = i * COEF
    #C = 25
    Y = grid_gen_2d(i)
    rang = [2*np.pi/np.sqrt(Y.shape[1]), 32*np.pi/np.sqrt(Y.shape[1])]
    #Y = np.array(states).T
    # F, G, comp_run_time_mp = fit_jax_basis_comp(states_actions, Y, rewards, next_states, gamma=0.98)
    # F = F.reshape(sample_size,i**2, 5).reshape(sample_size,5*i**2, order='F')
    # mp FQI
    F, G, run_time_mp_comp = fit_mp_basis_comp_seq(states_actions, Y, rewards, next_states, gamma=0.98)
    comp_time_mp.append(run_time_mp_comp)
    run_time_mp_iter, delta_theta_mp_max, theta_mp_max, mp_iter, traj = fit_mpfqi(F, G, gamma=0.98)
    iter_time_mp.append(run_time_mp_iter)
    iter_mp.append(mp_iter)
    t_mp.append(traj)

    F, G, run_time_mp_comp = fit_mp_basis_comp_seq(states_actions, Y, rewards, next_states, gamma=0.98, pwc=True)
    F_mp = F
    run_time_mp_iter, delta_theta_mp_max, theta_mp_pwc_max, mp_iter, traj = fit_mpfqi(F, G, gamma=0.98)
    t_mp_pwc.append(traj)
    iter_mp_pwc.append(mp_iter)

    # mp_v FQI
    Fs, Gs, run_time_v_comp = fit_v_mp_basis_comp_seq(states_actions, Y, rewards, next_states, gamma=0.98)
    F_out_q = Fs
    comp_time_v.append(run_time_v_comp)
    run_time_v_iter, delta_theta_v_max, theta_v_max, v_iter, traj = fit_v_mpfqi(Fs, Gs, gamma=0.98)
    iter_time_v.append(run_time_v_iter)
    iter_v.append(v_iter)
    t_v.append(traj)

    Fs, Gs, run_time_v_comp = fit_v_mp_basis_comp_seq(states_actions, Y, rewards, next_states, gamma=0.98, pwc=True)
    F_out_p = Fs
    run_time_v_iter, delta_theta_v_max, theta_v_pwc_max, v_iter, traj = fit_v_mpfqi(Fs, Gs, gamma=0.98)
    t_v_pwc.append(traj)
    iter_v_pwc.append(v_iter)


    # standard FQI
    F_star, F, M, run_time_fqi_comp = parallel_compute_F_Fplus(states_actions, Y, rewards, next_states)
    comp_time_fqi.append(run_time_fqi_comp)
    run_time_fqi_iter, delta_theta_fqi_max, theta_fqi_max, fqi_iter, traj = fit_rgfqi(F_star, F, M, gamma=0.98)
    iter_time_fqi.append(run_time_fqi_iter)
    iter_fqi.append(fqi_iter)
    t_fqi.append(traj)

    F_star, F, M, run_time_fqi_comp = parallel_compute_F_Fplus(states_actions, Y, rewards, next_states, pwc=True)
    run_time_fqi_iter, delta_theta_fqi_max, theta_fqi_pwc_max, fqi_iter, traj = fit_rgfqi(F_star, F, M, gamma=0.98)
    t_fqi_pwc.append(traj)
    iter_fqi_pwc.append(fqi_iter)


    init_states = []
    actions = []
    @partial(jit, static_argnames=["env", "p_v", "p_y", "mode"])
    def run_episode(key, env, p_y, p_v, theta_max, Y, C, rang, mode):
        # Initialize state and other variables
        state = env.reset(key)
        state_temp  = state
        # state = np.array([1.950931209892532, 27.904684321321927])
        optimal_rew = 0
        for i in range(100):
          action = - K_CL @ state_temp
          action = np.clip(action, -10, 10)
          state_temp, reward, done, _, _ = env.step(state_temp, action, actual_action=True)
          optimal_rew += reward * (0.98**i)
        #   jax.debug.print("action = {}", action)
        # jax.debug.print("optimal reward = {}", optimal_rew)
          #print(optimal_rew)

        done = False
        total_reward = 0
        B = np.zeros([p_y,1])
        Q = np.zeros([1,p_v])
        iter = 0

        # Function to check if the loop should continue
        def condition(loop_vars):
            _, _, done, _, iter = loop_vars
            #print(type((~done) * (iter < 100)))
            return ((1 - done) * (iter < 100)).astype("bool")


        # Function to perform operations at each step of the loop
        def body_mp(loop_vars):
            state, B, total_reward, done, iter = loop_vars
            for i in range(p_y):

                B = B.at[i].set(basis_x(state, Y[:, i], C))

            theta_max_res = theta_max.reshape(p_v, p_y)
            Q = max_plus_product_optimized(theta_max_res, B.reshape(-1,1))
            action = np.argmax(Q)
            state, reward, done, _, _ = env.step(state, action)  # Adapting to unpack correctly
            #total_reward += reward* (0.98**(EPI_LEN-iter))
            total_reward += reward* (0.98**iter)
            iter += 1
            return (state, B, total_reward, done, iter)

        def body_fqi(loop_vars):
            state, B, total_reward, done, iter = loop_vars
            for i in range(p_y):

                B = B.at[i].set(basis_rbf(state, Y[:, i], C))

            theta_max_res = theta_max.reshape(p_v, p_y)
            Q = theta_max_res @ B.reshape(-1,1)
            action = np.argmax(Q)
            state, reward, done, _, _ = env.step(state, action)  # Adapting to unpack correctly
            #total_reward += reward* (0.98**(EPI_LEN-iter))
            total_reward += reward* (0.98**iter)
            iter += 1
            return (state, B, total_reward, done, iter)

        def body_mp_pwc(loop_vars):
            state, B, total_reward, done, iter = loop_vars
            for i in range(p_y):

                B = B.at[i].set(basis_pwc_mp(state, Y[:, i], rang))

            theta_max_res = theta_max.reshape(p_v, p_y)
            Q = theta_max_res @ B.reshape(-1,1)
            action = np.argmax(Q)
            state, reward, done, _, _ = env.step(state, action)  # Adapting to unpack correctly
            #total_reward += reward* (0.98**(EPI_LEN-iter))
            total_reward += reward* (0.98**iter)
            iter += 1
            return (state, B, total_reward, done, iter)

        def body_fqi_pwc(loop_vars):
            state, B, total_reward, done, iter = loop_vars
            for i in range(p_y):

                B = B.at[i].set(basis_pwc_fqi(state, Y[:, i], rang))

            theta_max_res = theta_max.reshape(p_v, p_y)
            Q = theta_max_res @ B.reshape(-1,1)
            action = np.argmax(Q)
            state, reward, done, _, _ = env.step(state, action)  # Adapting to unpack correctly
            #total_reward += reward* (0.98**(EPI_LEN-iter))
            total_reward += reward* (0.98**iter)
            iter += 1
            return (state, B, total_reward, done, iter)

        # Initial values for the while loop
        initial_values = (state, B, total_reward, done, iter)

        # Running the while loop
        if mode == 0:
          state, B, total_reward, done, iter = lax.while_loop(condition, body_mp, initial_values)
        elif mode == 1:
          state, B, total_reward, done, iter = lax.while_loop(condition, body_fqi, initial_values)
        elif mode == 2:
          state, B, total_reward, done, iter = lax.while_loop(condition, body_mp_pwc, initial_values)
        elif mode == 3:
          state, B, total_reward, done, iter = lax.while_loop(condition, body_fqi_pwc, initial_values)

        return optimal_rew/total_reward

    def run_episodes(num_episodes, env, p_y, p_v, theta_max, Y, C, keys, rang, mode):

        # episodes = vmap(run_episode, in_axes=(0, None, None, None, None, None, None))(keys, env, p_y, p_v, theta_FQI_max, Y, C)
        run_episode_vectorized = vmap(run_episode, in_axes=(0, None, None, None, None, None, None, None, None))

        # Call the vectorized function over keys and other parameters
        episodes = run_episode_vectorized(keys[:num_episodes], env, p_y, p_v, theta_max, Y, C, rang, mode)
        return episodes

    # Configuration
    p_y = Y.shape[1]
    p_v = env.action_space.n
    p = p_y * p_v
    r = []
    eps = 100
    env = DCMotorEnv()

    # Run simulation
    mp_result = run_episodes(eps, env, p_y, p_v, theta_mp_max, Y, C, keys, rang, 0)
    v_result = run_episodes(eps, env, p_y, p_v, theta_v_max, Y, C, keys, rang, 0)
    fqi_result = run_episodes(eps, env, p_y, p_v, theta_fqi_max, Y, C, keys, rang, 1)
    mp_pwc_result = run_episodes(eps, env, p_y, p_v, theta_mp_pwc_max, Y, C, keys, rang, 2)
    v_pwc_result = run_episodes(eps, env, p_y, p_v, theta_v_pwc_max, Y, C, keys, rang, 2)
    fqi_pwc_result = run_episodes(eps, env, p_y, p_v, theta_fqi_pwc_max, Y, C, keys, rang, 3)

    mp_results.append(mp_result)
    v_results.append(v_result)
    fqi_results.append(fqi_result)
    mp_pwc_results.append(mp_pwc_result)
    v_pwc_results.append(v_pwc_result)
    fqi_pwc_results.append(fqi_pwc_result)


