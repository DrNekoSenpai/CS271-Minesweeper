import numpy as np

class Agent:
    """
    Chooses a valid action uniformly at random.
    Only clicks tiles marked -1 (exposed but unrevealed).
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, obs):
        exposed = np.argwhere(obs == -1)
        if len(exposed) == 0:
            return self.action_space.sample()

        z, r, c = exposed[np.random.randint(len(exposed))]

        H, W, D = obs.shape
        index = z * (H * W) + r * W + c
        return index