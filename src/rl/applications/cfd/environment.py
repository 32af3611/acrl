import torch
from torch import nn

from src.rl.framework.util import MultiLayerPerceptron


class CfdEnvironment(nn.Module):
    def __init__(self,
                 reward_model: MultiLayerPerceptron,
                 reward_type: str,
                 n_actions: int,
                 mean_min: float,
                 mean_max: float,
                 actions: torch.Tensor,
                 actions_size: int,
                 actions_type: str,
                 actions_stop_null: bool,
                 max_steps: int,
                 use_cumulative_reward: bool
                 ):
        super().__init__()

        self.reward_model = reward_model
        self.reward_type = reward_type
        self.n_actions = n_actions
        self.mean_min = mean_min
        self.mean_max = mean_max
        self.mean_dist = torch.distributions.Uniform(mean_min, mean_max)
        self.initial_states_fn = self._state_equal

        self.device = next(self.reward_model.parameters()).device
        self.actions_size = actions_size
        self.actions_type = actions_type
        self.actions_stop_null = torch.tensor(actions_stop_null).to(self.device)
        self.actions_size_half = self.actions_size // 2
        self.max_steps = max_steps
        self.use_cumulative_reward = use_cumulative_reward
        self.steps_decrement = 1 / self.max_steps
        self.actions = actions

        self.steps_left = None
        self.state = None
        self.profile = None
        self.reward_total = None
        self.reward = None
        self.drag = None
        self.baseline = None
        self.current_mean = None

    def _state_equal(self):
        return torch.cat([
            -torch.ones(self.actions_size_half),
            torch.ones(self.actions_size_half),
        ]).to(self.device) * self.current_mean

    def reset(self):
        self.current_mean = self.mean_dist.sample((1,)).to(self.device)
        self.steps_left = torch.Tensor([1]).unsqueeze(0).to(self.device)
        self.state = self.initial_states_fn().unsqueeze(0).to(self.device)

        self.profile = self.initial_states_fn().unsqueeze(0).to(self.device)
        self.profile, _, _ = self.normalize_profile(profile=self.profile, target=self.current_mean)

        self.drag = self.reward_model(self.profile).to(self.device)
        self.baseline = self.reward_model(self.profile).to(self.device)

        self.reward = torch.tensor([0.0]).to(self.device)
        self.reward_total = torch.tensor([0.0]).to(self.device)

        self.state = torch.cat([self.state, self.steps_left], dim=-1)

    def get_valid_actions(self, state):
        return self.actions

    def normalize_profile(self, profile, target):
        ps_coeffs = profile[0, :self.actions_size_half]
        ss_coeffs = profile[0, self.actions_size_half:]

        ps_coeffs = ps_coeffs - ps_coeffs.mean() + (-target)
        ss_coeffs = ss_coeffs - ss_coeffs.mean() + (+target)

        profile = torch.cat([ps_coeffs, ss_coeffs]).unsqueeze(0)

        return profile, ps_coeffs.mean(), ss_coeffs.mean()

    @torch.no_grad()
    def step(self, action):
        self.state = self.state[:, :self.actions_size]

        profile = self.state + action.to(self.device) \
            if self.actions_type == "additive" \
            else self.state * action.to(self.device)

        profile = profile.float()

        no_change = torch.norm(self.state - profile) == 0

        profile, ps_mean, ss_mean = self.normalize_profile(profile=profile, target=self.current_mean)

        drag = self.reward_model(profile)

        d_drag = self.drag - drag
        rewards = dict(
            drag=-drag,
            d_drag=d_drag,
            d_baseline=self.baseline - drag,
            discrete=torch.tensor(1 if d_drag > 0 else -1)
        )

        reward = sum([rewards[r] for r in self.reward_type])

        self.drag = drag
        self.reward = reward
        self.reward_total += reward.view(-1)
        self.profile = profile

        self.steps_left -= self.steps_decrement
        no_steps_left = self.steps_left < 0

        profile = torch.cat([profile, self.steps_left], dim=-1)

        self.state = profile

        done = no_steps_left or (self.actions_stop_null and no_change)

        info = dict(drag=drag, ps_mean=ps_mean, ss_mean=ss_mean)

        return self.state, self.reward_total if self.use_cumulative_reward else self.reward, done, info
