""" 
    DDPG + HER based on Baselines Implementation

    Jose Edgar Hernandez Cancino Estrada
    B.S. in Robotics and Digital Systems Engineering (June, 2024)
    Tecnologico de Monterrey

    Utilization of xArm6Reach-v1 gym environment by Julio-Design:
    https://github.com/julio-design/xArm6-Gym-Env

"""

# Import libraries
import gym
import numpy as np
import torch
import sys
import os
import time

from mpi4py import MPI

from baselines_code.util import *
from ddpg_her_agent import DDPG_HER_AGENT

import gym_xarm6

from networks import ActorNetwork
import random


def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args['clip_obs'], args['clip_obs'])
    g_clip = np.clip(g, -args['clip_obs'], args['clip_obs'])
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args['clip_range'], args['clip_range'])
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args['clip_range'], args['clip_range'])
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
    return inputs

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    
    # configure logger, disable logging in child MPI processes (with rank > 0)
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)

    # Create gym environment
    env = gym.make(args.env)
    obs = env.reset()

    # Get algorithm and environment parameters
    kwargs = HER_PARAMS

    env_params = {'obs_shape': obs['observation'].shape[0],
        'goal_shape': obs['desired_goal'].shape[0],
        'action_shape': env.action_space.shape[0],
        'max_action': env.action_space.high[0],
        'min_action': -env.action_space.high[0],
        'max_timesteps': env._max_episode_steps,
        'env_type': "Mujoco",
        'seed': 123,
        'n_actions': 4,
        'clip_range': 5.0,
        'save_dir': "models",
        'env': args.env
        }
    
    kwargs.update(env_params)

    # Set random seeds for reproductibility
    env.seed(kwargs['seed'] + MPI.COMM_WORLD.Get_rank())
    random.seed(kwargs['seed'] + MPI.COMM_WORLD.Get_rank())
    np.random.seed(kwargs['seed'] + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(kwargs['seed'] + MPI.COMM_WORLD.Get_rank())
            
    if args.test:
        if os.path.isfile(args.test):
            o_mean, o_std, g_mean, g_std, model = torch.load(args.test, map_location=lambda storage, loc: storage)
  
            env = gym.make(args.env)
            observation = env.reset()
            # create the actor network
            actor_network = ActorNetwork(alpha = kwargs['pi_lr'], 
                                  state_dims = kwargs['obs_shape'] + kwargs['goal_shape'], 
                                  n_actions=kwargs['n_actions'],
                                  name="Actor",
                                  max_action=kwargs['max_action'])
            
            actor_network.load_state_dict(model)
            actor_network.eval() 

            for i in range(40):
                observation = env.reset_goal()
                # start to do the demo

                obs = observation['observation']
                g = observation['desired_goal']

                position_trajectory = np.zeros((env._max_episode_steps, 3))

                position_trajectory[0] = obs[:3]

                for t in range(env._max_episode_steps):
                    env.render()
                    inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, kwargs)
                    with torch.no_grad():
                        pi = actor_network(inputs.cuda())
                    action = pi.cpu().numpy().squeeze()

                    # put actions into the environment
                    observation_new, reward, _, info = env.step(action)
                    obs = observation_new['observation']
                    position_trajectory[t] = observation_new['observation'][:3]
                
                    if info['is_success']:
                        env.render()
                        for i in range(1000):
                            continue
                        # import time
                        # time.sleep(1)
                        break
                print('the episode is: {}, is success: {}'.format(i, info['is_success']))
                
        else:
            print(f"Test file {args.test} not found")

    elif args.train:
        print('Training {} on {}:{} with algorithm arguments \n{}'.format(
            args.alg, kwargs['env_type'], kwargs['env'], kwargs))
    
        model = DDPG_HER_AGENT(env=env, params=kwargs)
        model.learn()
    
    else:
        env.render()
        time.sleep(5)