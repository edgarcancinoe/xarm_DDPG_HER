from networks import ActorNetwork, CriticNetwork
from her import HER_Sampler
import os
from mpi4py import MPI
from baselines_code.replay_buffer import ReplayBuffer
from baselines_code.normalizer import Normalizer
from baselines_code.util import transitions_in_episode_batch
import torch
from episode_worker import EpisodeWorker
from datetime import datetime
import numpy as np
from mpi_utils import sync_networks, sync_grads
import torch.nn.functional as F

class DDPG_HER_AGENT():
    def __init__(self, env, params):
        self.env = env
        self.args = params
        
        # Create networks
        self.actor = ActorNetwork(alpha = self.args['pi_lr'], 
                                  state_dims = self.args['obs_shape'] + self.args['goal_shape'], 
                                  n_actions=self.args['n_actions'],
                                  name="Actor",
                                  max_action=self.args['max_action'])
        
        self.critic = CriticNetwork(beta=self.args['Q_lr'], 
                                    state_dims = self.args['obs_shape'] + self.args['goal_shape'],
                                    n_actions=self.args['n_actions'],
                                    name="Critic",
                                    max_action=self.args['max_action'])

        # Sync networks across cpus 
        sync_networks(self.actor)
        sync_networks(self.critic)

        # Create target networks
        self.target_actor = ActorNetwork(alpha = self.args['pi_lr'], 
                                  state_dims = self.args['obs_shape'] + self.args['goal_shape'], 
                                  n_actions = self.args['n_actions'],
                                  name = "TargetActor",
                                  max_action = self.args['max_action'])
        
        self.target_critic = CriticNetwork(beta = self.args['Q_lr'], 
                                    state_dims = self.args['obs_shape'] + self.args['goal_shape'],
                                    n_actions = self.args['n_actions'],
                                    name = "TargetCritic",
                                    max_action = self.args['max_action'])

        self.actor.cuda()
        self.critic.cuda()
        self.target_actor.cuda()
        self.target_critic.cuda()
    
        # Make target networks be the same as training networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # HER sampler
        self.HER_sampler = HER_Sampler(replay_k=self.args['replay_k'], replay_strategy='future', reward_function=self.env.compute_reward)
        self.sample_transitions = self.HER_sampler.sample_transitions

        # Replay Buffer
        self.T = self.args['max_timesteps']
        buffer_shapes = dict()
        buffer_shapes['o'] = (self.T + 1, self.args['obs_shape'])
        buffer_shapes['g'] = (self.T, self.args['goal_shape'])
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.args['goal_shape'])
        buffer_shapes['u'] = (self.T, self.args['action_shape'])
        buffer_shapes['ag'] = (self.T + 1, self.args['goal_shape'])


        self.buffer = ReplayBuffer(buffer_shapes = buffer_shapes,
                                   size_in_transitions = self.args['buffer_size'],
                                   T = self.args['max_timesteps'], 
                                   sample_transitions = self.sample_transitions)
        
        # MPI normalizer to ensure that observations ara approximately distributed
        # according to a standard Normal distribution (mean zero and variance one).
        self.o_norm = Normalizer(size=self.args['obs_shape'], default_clip_range=self.args['clip_range'])
        self.g_norm = Normalizer(size=self.args['goal_shape'], default_clip_range=self.args['clip_range'])

        # Worker to play rollouts
        self.worker = EpisodeWorker(env=self.env, args=self.args, o_norm=self.o_norm, g_norm=self.g_norm)

        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args['save_dir']):
                os.mkdir(self.args['save_dir'])
            self.model_path = os.path.join(self.args['save_dir'], self.args['env'])
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
    
    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args['clip_obs'], self.args['clip_obs'])
        g = np.clip(g, -self.args['clip_obs'], self.args['clip_obs'])
        return o, g
    
    def _update_normalizer(self, episode_batch):
        episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
        episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]

        num_normalizing_transitions = transitions_in_episode_batch(episode_batch)

        transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)
        o, g, ag = transitions['o'], transitions['g'], transitions['ag']
        transitions['o'], transitions['g'] = self._preproc_og(o, g)
        # No need to preprocess the o_2 and g_2 since this is only used for stats

        self.o_norm.update(transitions['o'])
        self.g_norm.update(transitions['g'])

        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _soft_update(self, target, source):
        for target_weight, weight in zip(target.parameters(), source.parameters()):
            target_weight.data.copy_((1 - self.args['polyak']) * weight.data + self.args['polyak'] * target_weight.data)

    def _evaluate(self):
        success_rate = []
        for rollout in range(self.args["n_test_rollouts"]):
            success = self.worker.generate_rollout(self.actor)[-1][-1]
            success_rate.append(success)
        success_rate = np.mean(np.array(success_rate))
        global_success_rate = MPI.COMM_WORLD.allreduce(success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()    
            
    def _train(self):
        # 1. Get batch sample from buffer
        transitions = self.buffer.sample(batch_size=self.args['batch_size'])

        # Input preprocessing
        state, goal = self._preproc_og(transitions['o'], transitions['g'])
        new_state, new_goal = self._preproc_og(transitions['o_2'], transitions['g'])
        
        # Normalize inputs
        state = self.o_norm.normalize(state)
        goal = self.g_norm.normalize(goal)
        input = np.concatenate([state, goal], axis=1)

        new_state = self.o_norm.normalize(new_state)
        new_goal = self.g_norm.normalize(new_goal)
        input_new = np.concatenate([new_state, new_goal], axis=1)

        # Create tensors
        input = torch.tensor(input, dtype=torch.float32).cuda()
        input_new = torch.tensor(input_new, dtype=torch.float32).cuda()
        actions = torch.tensor(transitions['u'], dtype=torch.float32).cuda()
        rewards = torch.tensor(transitions['r'], dtype=torch.float32).cuda()

        # 2. Set y_i = r_i + γ * Q'(s_{i+1}, u'(s_{i+1} | 0^{u'}) | 0^{Q})

        with torch.no_grad():
            # u'(s_{i+1} | 0^{u'})
            next_action_by_target_actor = self.target_actor(input_new)
            # Q'(s_{i+1}, u'(s_{i+1} | 0^{u'}) | 0^{Q})
            q_next_state = self.target_critic(input_new, next_action_by_target_actor).detach()

        # y_i
        target_q = (rewards + self.args['gamma'] * q_next_state).detach()

        # Clip the q_value used to train the critic to the range [- 1/(1 - γ), 0] as specified in paper
        clip_range = - 1 / (1 - self.args['gamma'])
        target_q = torch.clamp(target_q, clip_range, 0)


        # 3. Update critic by minimizing L = mean((y_i - Q(s_i, q_i | 0^{Q}))^2)

        # Q(s_i, q_i | 0^{Q})
        critic_value = self.critic(input, actions)
        L = F.mse_loss(target_q, critic_value)

        self.critic.optimizer.zero_grad()
        L.backward()
        sync_grads(self.critic)
        self.critic.optimizer.step()
        
        # 4. Update actor policy using sampled policy gradient grad(J) = mean(grad(Q(s_i, u(s_i | 0^{u}) | 0^{Q})))
        
        # u(s_i | 0^{u})
        mu = self.actor(input)
        # Q(s_i, u(s_i | 0^{u}) | 0^{Q}) & Is negative bc gradient ascent
        actor_loss = -self.critic(input, mu).mean()
        # Add quadratic penalty on actions to mantain smoothnes, as in baselines implementation.
        actor_loss += self.args['action_l2'] * (mu / self.args['max_action']).pow(2).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor.optimizer.step()

        
    def learn(self):
        
        for epoch in range(self.args['n_epochs']):
            for cycle in range(self.args['n_cycles']):
                # Generate rollout_batch_size rollouts per mpi
                episode_obs, episode_achieved_goals, episode_goals, episode_actions = [], [], [], []
                for rollout in range(self.args['rollout_batch_size']):
                    # Generate episode
                    episode = self.worker.generate_rollout(self.actor)

                    episode_obs.append(episode[0])
                    episode_actions.append(episode[1])
                    episode_goals.append(episode[2])
                    episode_achieved_goals.append(episode[3])

                # Store episode history
                episode_dict = dict(o=np.array(episode_obs).copy(),
                                    u=np.array(episode_actions).copy(),
                                    g=np.array(episode_goals).copy(),
                                    ag=np.array(episode_achieved_goals).copy())

                # Store episode on buffer
                self.buffer.store_episode(episode_batch=episode_dict)
                # Update normalizer
                self._update_normalizer(episode_batch=episode_dict)
                
                for batch in range(self.args['n_batches']):
                    # Train networks
                    self._train()
                
                # Soft update
                self._soft_update(self.target_actor, self.actor)
                self._soft_update(self.target_critic, self.critic)

            # Start epoch evaluation & save model
            success_rate = self._evaluate()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor.state_dict()], \
                            self.model_path + '/model.pt')

