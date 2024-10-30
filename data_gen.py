import jax.numpy as np
import jax.random as jnd
from DC_motor_env import *

USE_NEW_REW = False
QUAD_REWARD = False
GET_PARAMS = False
ULT_TEST = False
WILD_INIT = False
SAMPLE_GRID = False
RANGE_MAT = np.array([[-np.pi, np.pi],
             [-16*np.pi, 16*np.pi]])
K_CL = np.array([12.798508208418262, 0.682923885557335])


env = DCMotorEnv()
n_episodes = 5000
max_length = 1
sample_size = n_episodes * max_length
keyarray = jnd.split(jnd.PRNGKey(1), sample_size)


# Collect data
data = []
states = []
actions = []
rewards = []
next_states = []
episode_lengths = []
keys = jnd.split(jnd.PRNGKey(42), 100)


for i in range(sample_size):
    state = env.reset(keyarray[i])
    states.append(state)
    action = env.action_space.sample()  # Random policy
    actions.append(action)
    next_state, reward, done, _, _ = env.step(state, action)
    rewards.append(reward)
    next_states.append(next_state)
    data.append((state, action, reward, next_state))




states_actions = []
for d in data:
    state_action = np.append(d[0], d[1])
    states_actions.append(state_action)
rewards = np.array([d[2] for d in data])
next_states = np.array([d[3] for d in data])

env.close()

