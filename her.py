from baselines_code.her_sampler import sample_her_transitions

class HER_Sampler:
    def __init__(self, replay_k, replay_strategy='future', reward_function=None) -> None:
        self.k = replay_k
        self.strategy = replay_strategy
        self.reward_function = reward_function
        self.future_p = 1 - (1. / (1 + replay_k)) if self.strategy == 'future' else 0        
    
    def sample_transitions(self, episode_batch, batch_size_in_transitions):
        return sample_her_transitions(episode_batch, batch_size_in_transitions, self.reward_function, self.future_p)
    