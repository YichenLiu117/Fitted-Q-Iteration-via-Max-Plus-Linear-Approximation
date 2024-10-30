import jax.numpy as np


def unif_grid(Delta_x, N_vec):
    X = []
    for i in range(Delta_x.shape[0]):
        X_i = np.linspace(Delta_x[i, 0], Delta_x[i, 1], N_vec[i])
        X.append(X_i.reshape(-1, 1))

    return X

def basis_x(x, y, c):
    return -c * np.dot((x - y).T, (x - y))

# Implementing basis_u using JAX, ensuring compatibility with JAX transformations
def basis_u(u, v):
    # return 0 if u == v else -np.inf
    return np.log(u == v)
def basis_pwc_fqi(x, y, rang):
    check0 = np.float32(np.array(np.abs(x[0] - y[0])) < rang[0]/2)
    check1 = np.float32(np.array(np.abs(x[1] - y[1])) < rang[1]/2)
    return check0 * check1
    
    
def basis_pwc_mp(x, y, rang):
    dist0 = np.array(np.abs(x[0] - y[0]))
    dist1 = np.array(np.abs(x[1] - y[1]))
    check0 = np.float32(dist0 < rang[0]/2)
    check1 = np.float32(dist1 < rang[1]/2)
    flag = check0 * check1
    # return np.log(flag)
    return (flag-1) * np.sqrt(dist0**2 + dist1**2)

def basis_almost_pwc(x, y):
    check = np.array(np.abs(np.abs(x) - np.abs(y)) - RANG/2)
    if np.all(check <= 0):
        return 0
    else:
        return -10000
def basis_rbf(x, y, c):
    # Using squared Euclidean distance in an RBF kernel
    return np.exp(-(1/c) * np.linalg.norm(x - y)**2)

def basis_1(u, v):
    return np.float32(u == v)

def max_plus_product_optimized(A, B):
    # Element-wise addition between rows of A and columns of B
    def max_plus_single_row(a_row, B):
        return np.max(a_row[:, np.newaxis] + B, axis=0)

    # Vectorize across rows of A using vmap
    max_plus_vectorized = vmap(max_plus_single_row, in_axes=(0, None))
    
    return max_plus_vectorized(A, B)

def new_rew_func_diff(state, next_state):
    check1 = (abs(state[0]) - abs(next_state[0])) > 0
    check2 = (abs(state[2]) - abs(next_state[2])) > 0

    return int(check1) + int(check2) - 1

def quadratic_reward(x):
    return - (x[0] ** 2 + (11.5 * x[2]) **2)



def grid_gen_2d(size):
    Delta_y = np.array(RANGE_MAT)

    # Number of data points for each dimension
    p_y_g = np.array([size, size])
    n_vec = p_y_g+np.array([2,2])
    # Generate grid points for each dimension
    Y_g = np.array([np.linspace(Delta_y[dim, 0], Delta_y[dim, 1], p_y_g[dim]) for dim in range(2)])*((size-1)/size)
    # Y_g = np.array([np.linspace(Delta_y[dim, 0], Delta_y[dim, 1], p_y_g[dim]) for dim in range(2)])
    # Product of sizes for total number of points
    p_y = np.prod(p_y_g)

    # Initialize Y with zeros for 2 dimensions
    Y = np.zeros((2, p_y))

    # Fill Y with grid coordinates
    index = 0
    for i in range(p_y_g[0]):
        for j in range(p_y_g[1]):
            Y = Y.at[:, index].set([Y_g[0][i], Y_g[1][j]])
            index += 1

    return Y