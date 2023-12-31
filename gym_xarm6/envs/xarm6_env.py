"""
Author: OpenAI
Modified by: Julio Cesar RAMIREZ
Institution: ESTIA

Modified in November 2023 by J.Edgar Hernandez Cancino EStrada
B.S. in Robotics and Digital Systems Engineering (June, 2024)
Tecnologico de Monterrey

"""
import numpy as np
from gym_xarm6.envs import rotations, robot_env, utils
import time
from gym_xarm6.data import dhm
import math
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

var = -math.sqrt(2)/2
eef_qpos = [var, 0, var, 0]
# eef_qpos = [0.5, 0.5, -0.5, 0.5]

class xArm6Env(robot_env.RobotEnv):
    """Superclass for all xArm6 environments.
    """

    def __init__(
            self, model_path, n_substeps, gripper_extra_height, block_gripper,
            target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, initial_qpos, reward_type,
            ):
        """Initializes a new xArm6 environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        super(xArm6Env, self).__init__(
                model_path=model_path, n_substeps=n_substeps, n_actions=4,
                initial_qpos=initial_qpos)

        # GoalEnv methods
    # --------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # --------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:wrist_roll_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.02  # limit maximum change in position
        rot_ctrl = eef_qpos  # fixed rotation of the end effector, expressed as a quaternion

        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper: 
            gripper_ctrl = np.zeros_like(gripper_ctrl) 
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)    

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        
        # object_pos = np.zeros(0)
        # gripper_state = robot_qpos[-2:]
        # gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        achieved_goal = grip_pos.copy()

        obs = np.concatenate([
            grip_pos, grip_velp
            ])

        return {
                'observation': obs.copy(),
                'achieved_goal': achieved_goal.copy(),
                'desired_goal': self.goal.copy(),
            }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:tool_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        self.sim.forward()
        return True

    def _working_space(self):
        pass


    def _sample_goal(self):

        goal = np.array([
            np.random.uniform(0.5, .62, size=1)[0],
            np.random.uniform(-.3, .3, size=1)[0],
            np.random.uniform(.15, .45, size=1)[0]])

        # goal = np.array([
        #     np.random.uniform(-.3, .3, size=1)[0],
        #     np.random.uniform(-0.42, -.62, size=1)[0],
        #     np.random.uniform(.15, .45, size=1)[0]])
        
        return goal.copy()


    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array(eef_qpos)

        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(50):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        

    def render(self, mode='human', width=500, height=500):
        return super(xArm6Env, self).render(mode, width, height)
