import numpy as np
import torch

class EpisodeWorker:
    def __init__(self, env, args, o_norm, g_norm):
        self.env = env
        self.args = args
        self.o_norm = o_norm
        self.g_norm = g_norm
    
    def _get_action(self, pre_action):
        action = pre_action.cpu().numpy().squeeze()

        # Add gaussian noise
        action += self.args['noise_eps'] * self.args['max_action'] * np.random.randn(*action.shape)
        # Clip actions
        action = np.clip(action, self.args['min_action'], self.args['max_action'])
        # Generate random actions
        random_actions = np.random.uniform(low=-self.args['max_action'], high=self.args['max_action'], size=action.shape)
        
        # Choose to use random actions with a certain probability
        use_random = np.random.binomial(1, self.args['random_eps'], 1)[0]
        if use_random:
            # If selected, blend in a fraction of random actions with the current actions
            action += use_random * (random_actions - action)

        return action

    def _pre_process_input(self, observation, goal):
        observation_normalized = self.o_norm.normalize(observation)
        goal_normalized = self.g_norm.normalize(goal)

        input = np.concatenate([observation_normalized, goal_normalized])
        input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)

        return input
    
    def generate_rollout(self, actor, test=False):
        # Get starting state
        observation = self.env.reset()
        obs = observation['observation']
        goal = observation['desired_goal']
        ag = observation['achieved_goal']

        # Generate episodes
        r_obs, r_achieved_goals, r_acts, r_goals, r_successes = [], [], [], [], []

        for t in range(self.args['max_timesteps']):
            with torch.no_grad():
                input = self._pre_process_input(obs, goal)
                action = actor(input.cuda())
                if not test:
                    action = self._get_action(action)
                else:
                    action = action.detach().cpu().numpy().squeeze()
                    
            # Execute action
            observation_new, reward, done, info = self.env.step(action)

            # Get results of action
            obs_new = observation_new['observation']
            ag_new = observation_new['achieved_goal']

            # Append to rollout arrays
            r_obs.append(obs.copy())
            r_acts.append(action.copy())
            r_achieved_goals.append(ag.copy())
            r_goals.append(goal.copy())
            r_successes.append(info['is_success'].copy())

            if (done):
                break

            # Update environment status
            obs = obs_new
            ag = ag_new

        # Add last observation & achieved goal
        r_obs.append(obs.copy())
        r_achieved_goals.append(ag.copy())

        return [r_obs, r_acts, r_goals, r_achieved_goals, r_successes]
