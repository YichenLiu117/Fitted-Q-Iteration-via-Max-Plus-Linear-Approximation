from functools import partial
import jax.numpy as np
from jax import jit, vmap
import jax
import time
from utils import *

jax.config.update("jax_enable_x64", True)

@partial(jit, static_argnames=["pwc"])
def parallel_compute_F_Fplus(states_actions, Y, rewards, next_states, gamma=0.98, pwc=False):
    # Vectorizing the computation of F_star and F
    p_y = Y.shape[1]
    p_v = env.action_space.n
    p = p_y * p_v
    states_actions = np.array(states_actions)
    n = states_actions.shape[0]
    states = states_actions[:, 0:2]
    actions = states_actions[:, 2]
    V = np.array([0,1,2,3,4])
    F = np.zeros((n, p))
    G = np.zeros((n, p))
    F_star = np.zeros((n, p_y))
    rang = [2*np.pi/np.sqrt(Y.shape[1]), 32*np.pi/np.sqrt(Y.shape[1])]
    def compute_slice(i):
    # Calculate F_star for each i, j combination
        if pwc:
            F_star_slice = vmap(lambda j: basis_pwc_fqi(next_states[i], Y[:, j], rang))(np.arange(p_y))

            # Calculate F for each i, j, k combination
            # Concatenate results over the innermost dimension
            F_slice = np.concatenate([
                vmap(lambda j: basis_pwc_fqi(states[i], Y[:, j], rang) * basis_1(actions[i], V[k]))(np.arange(p_y))
                for k in range(p_v)
            ], axis=0)
        

        else:
            F_star_slice = vmap(lambda j: basis_rbf(next_states[i], Y[:, j], C))(np.arange(p_y))

            # Calculate F for each i, j, k combination
            # Concatenate results over the innermost dimension
            F_slice = np.concatenate([
                vmap(lambda j: basis_rbf(states[i], Y[:, j], C) * basis_1(actions[i], V[k]))(np.arange(p_y))
                for k in range(p_v)
            ], axis=0)

        return F_star_slice, F_slice
    rg = 0.001
    # Apply compute_slice across all elements i in parallel
    t = time.perf_counter()
    # for i in range(n):
    #     for j in range(p_y):
    #         F_star = F_star[i, j].set(basis_rbf(next_states[i], Y[:, j], C))
    #         for k in range(p_v):
    #             F = F.at[i, (k) * p_y + j].set(basis_rbf(states[i], Y[:, j], C) * basis_1(actions[i], V[k]))

    F_star_all, F_all = vmap(compute_slice)(np.arange(n))
    F = F_all.reshape(n, -1, order='F')
    M = (np.linalg.inv((F.T @ F) + rg * np.eye(p))) @ F.T
    run_time = time.perf_counter() - t
    # Reshape F_all to have proper shape (n, p)
    return F_star_all, F, M, run_time

@partial(jit, static_argnames=["pwc"])
def fit_mp_basis_comp_seq(states_actions, Y, rewards, next_states, gamma, pwc=False):


    p_y = Y.shape[1]
    p_v = env.action_space.n
    p = p_y * p_v
    states_actions = np.array(states_actions)
    n = states_actions.shape[0]
    states = states_actions[:, 0:2]
    actions = states_actions[:, 2]
    V = np.array([0,1,2,3,4])
    F = np.zeros((n, p))
    #G1 = np.zeros((n, p))
    F_plus = np.zeros((n, p_y))
    rang = [2*np.pi/np.sqrt(Y.shape[1]), 32*np.pi/np.sqrt(Y.shape[1])]
    
    def parallel_compute_F_Fplus(n, p_y, p_v, states, actions, next_states, Y, V, C, pwc):
    # Vectorizing the computation of F_plus and F
    # Define the fully vectorized function for one slice (i.e., one 'i')
        def compute_slice(i):
            if pwc: 
                F_plus_slice = vmap(lambda j: basis_pwc_mp(next_states[i], Y[:, j], rang))(np.arange(p_y))
                F_slice = np.concatenate([
                    vmap(lambda j: basis_pwc_mp(states[i], Y[:, j], rang) + basis_u(actions[i], V[k]))(np.arange(p_y))
                    for k in range(p_v)
                ])
            else:
                F_plus_slice = vmap(lambda j: basis_x(next_states[i], Y[:, j], C))(np.arange(p_y))
                F_slice = np.concatenate([
                    vmap(lambda j: basis_x(states[i], Y[:, j], C) + basis_u(actions[i], V[k]))(np.arange(p_y))
                    for k in range(p_v)
                ])
            return F_plus_slice, F_slice

        # Vectorize compute_slice over all states/actions/next_states
        F_plus_all, F_all = vmap(compute_slice)(np.arange(n))

        return F_plus_all, F_all.reshape(n, -1)

    t = time.perf_counter()

    # for i in range(n):
    #     for j in range(p_y):
    #         F_plus = F_plus.at[i,j].set(basis_x(next_states[i], Y[:, j], C))
    #         for k in range(p_v):
    #             F = F.at[i, (k) * p_y + j].set(basis_x(states[i], Y[:, j], C) + basis_u(actions[i], V[k]))
                # G1[i, (k) * p_y + j] = rewards[i] + gamma * basis_x(next_states[i], Y[:, j], C)
    
    F_plus, F = parallel_compute_F_Fplus(n, p_y, p_v, states, actions, next_states, Y, V, C, pwc)

    Rew_s_replicated = np.tile(rewards.reshape(-1,1), (1, p))

# Replicate Fx_Xplus across columns p_v times
    F_plus_replicated = np.tile(F_plus, (1, p_v))

# Calculate G
    G = Rew_s_replicated + gamma * F_plus_replicated

    run_time = time.perf_counter() - t

    return F, G, run_time

@partial(jit, static_argnames=["pwc"])
def fit_v_mp_basis_comp_seq(states_actions, Y, rewards, next_states, gamma, pwc=False):


    p_y = Y.shape[1]
    p_v = env.action_space.n
    p = p_y * p_v
    states_actions = np.array(states_actions)
    n = states_actions.shape[0]
    states = states_actions[:, 0:2]
    actions = states_actions[:, 2]
    V = np.array([0,1,2,3,4])
    F = np.zeros((n, p))
    #G1 = np.zeros((n, p))
    F_plus = np.zeros((n, p_y))
    rang = [2*np.pi/np.sqrt(Y.shape[1]), 32*np.pi/np.sqrt(Y.shape[1])]
 
                # G1[i, (k) * p_y + j] = rewards[i] + gamma * basis_x(next_states[i], Y[:, j], C)
    def parallel_compute_F_Fplus(n, p_y, p_v, states, actions, next_states, Y, V, C, pwc):
    # Vectorizing the computation of F_plus and F
    # Define the fully vectorized function for one slice (i.e., one 'i')
        def compute_slice(i):
            if pwc: 
                F_plus_slice = vmap(lambda j: basis_pwc_mp(next_states[i], Y[:, j], rang))(np.arange(p_y))
                F_slice = np.concatenate([
                    vmap(lambda j: basis_pwc_mp(states[i], Y[:, j], rang) + basis_u(actions[i], V[k]))(np.arange(p_y))
                    for k in range(p_v)
                ])
            else:
                F_plus_slice = vmap(lambda j: basis_x(next_states[i], Y[:, j], C))(np.arange(p_y))
                F_slice = np.concatenate([
                    vmap(lambda j: basis_x(states[i], Y[:, j], C) + basis_u(actions[i], V[k]))(np.arange(p_y))
                    for k in range(p_v)
                ])
            return F_plus_slice, F_slice
        # Vectorize compute_slice over all states/actions/next_states
        F_plus_all, F_all = vmap(compute_slice)(np.arange(n))

        return F_plus_all, F_all.reshape(n, -1)
    t = time.perf_counter()

    # for i in range(n):
    #     for j in range(p_y):
    #         F_plus = F_plus.at[i,j].set(basis_x(next_states[i], Y[:, j], C))
    #         for k in range(p_v):
    #             F = F.at[i, (k) * p_y + j].set(basis_x(states[i], Y[:, j], C) + basis_u(actions[i], V[k]))
    F_plus, F = parallel_compute_F_Fplus(n, p_y, p_v, states, actions, next_states, Y, V, C, pwc)

    Rew_s_replicated = np.tile(rewards.reshape(-1,1), (1, p))

# Replicate Fx_Xplus across columns p_v times
    F_plus_replicated = np.tile(F_plus, (1, p_v))

# Calculate G
    G = Rew_s_replicated + gamma * F_plus_replicated
    Fs = max_plus_product_optimized(F.T, F)
    Gs = max_plus_product_optimized(F.T, G)

    run_time = time.perf_counter() - t

    return Fs, Gs, run_time


