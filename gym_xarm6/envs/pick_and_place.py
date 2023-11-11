import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env

class xArm6ReachEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, '')
