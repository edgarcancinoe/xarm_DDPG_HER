HER_PARAMS = {
    # ddpg
    'Q_lr': 0.005,  # critic learning rate
    'pi_lr': 0.005,  # actor learning rate
    'buffer_size': int(1E6),  # for experience replay
    'polyak': 0.95,  # polyak averaging coefficient
    'action_l2': 1.5,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'gamma': 0.98,
    # training
    'n_epochs': 36,
    'n_cycles': 80,  # per epoch
    'rollout_batch_size': 1,  # per mpi thread
    'n_batches': 96,  # training batches per cycle
    'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 20,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    # exploration
    'random_eps': 0.3,  # percentage of time a random action is taken
    'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
}

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    parser.add_argument('--env', help='environment ID', type=str, default='xArm6Reach-v1')
    parser.add_argument('--alg', help='Algorithm', type=str, default='HER')
    
    return parser

def transitions_in_episode_batch(episode_batch):
    """Number of transitions in a given episode batch.
    """
    shape = episode_batch['u'].shape
    return shape[1]