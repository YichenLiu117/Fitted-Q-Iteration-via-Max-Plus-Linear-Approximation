import numpy
from gym import spaces
import jax.random as jnd
import jax.numpy as np

class DCMotorEnv(gym.Env):
    """Custom Environment for a DC motor that follows gym interface"""
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(DCMotorEnv, self).__init__()

        # Define action and observation space
        # Actions are discrete integer values in the set {-10, -5, 0, 5, 10}
        self.action_space = spaces.Discrete(5)
        self.actions = np.array([-10, -5, 0, 5, 10])

        # Observations are the state variables, with each state having a range
        self.observation_space = spaces.Box(low=numpy.array([-np.pi, -16*np.pi]),
                                            high=numpy.array([np.pi, 16*np.pi]),
                                            dtype=numpy.float32)

        # System dynamics matrices
        self.A = numpy.array([[1, 0.0049],
                           [0, 0.9540]])
        self.B = numpy.array([0.0021, 0.8505])
        self.Q = numpy.array([[5, 0], [0, 0.01]])
        self.R = 0.01
    def step(self, state, action, actual_action=False):
        # Map the action to the corresponding value
        if actual_action:
          u = action
        else:
          u = self.actions[action]
        # Update state
        state = self.A @ state + self.B * u
        state = np.clip(state, self.observation_space.low, self.observation_space.high)
        # Calculate reward
        # exponentiated
        # reward = np.expm1(-state.T @ self.Q @ state - self.R * u**2)
        # bimodal concave
        # b = 5
        # r1 = -state.T @ self.Q @ state - self.R * u**2 - b*state
        # r2 = r1 + 2*b*state
        # reward = np.max(np.array([r1, r2]))
        reward = -state.T @ self.Q @ state - self.R * u**2
        # Check if the state is terminal (For this case, let's say it never is)
        done = False
        # Optionally we can pass additional info, not used in this case
        info = {}
        truncated = done
        return state, reward, done, truncated, info

    def reset(self, key):
        # Reset the state to a random initial state within the allowed range
        return jnd.uniform(key,
                           self.observation_space.low.shape,
                           minval=np.array(self.observation_space.low),
                           maxval=np.array(self.observation_space.high))

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError("Only console mode is supported.")
        print(f"Current state: {self.state}")

    def close(self):
        pass