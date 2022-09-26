import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import CyclicLR, ExponentialLR
from torch.optim.optimizer import Optimizer

from src.rl.framework.util import MultiLayerPerceptron, ReplayBuffer


class DoubleDQNAgent(nn.Module):
    def __init__(
            self,
            online_network: MultiLayerPerceptron,
            target_network: MultiLayerPerceptron,
            n_actions: int,
            optimizer: Optimizer,
            scheduler: CyclicLR or ExponentialLR,
            loss: _Loss,
            batch_size: int,
            gamma: float,
            epsilon: float,
            epsilon_threshold: float,
            buffer_size: int
    ):
        super().__init__()

        self.online_network = online_network
        self.target_network = target_network
        self.n_actions = n_actions

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_threshold = epsilon_threshold

        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(size=self.buffer_size)

    @torch.no_grad()
    def action(self, state):
        """Perform action in a given state."""
        random = torch.rand(size=(1,))[0] < self.epsilon

        if random:
            return torch.randint(low=0, high=self.n_actions, size=(1,)), None

        q_values = self.online_network(state)
        _, action = q_values.view(-1).max(dim=0)
        return action, q_values

    def optimize(self):
        states, actions, next_states, rewards, dones = zip(*self.replay())

        state_batch = torch.cat(states)
        next_state_batch = torch.cat(next_states)
        reward_batch = torch.cat(rewards)
        done_batch = torch.cat(dones)

        state_action_values = self.online_network(state_batch)

        # aneb: no need to select the previously chosen action if there is only one
        if self.online_network.output_size > 1:
            action_batch = torch.cat(actions)
            state_action_values = state_action_values.gather(1, action_batch.view(-1, 1))

        with torch.no_grad():
            next_state_values = self.target_network(next_state_batch).max(1)[0].unsqueeze(1)
            next_state_values[done_batch] = 0.0

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return loss

    def reset_buffer(self):
        """ Reset replay buffer. """
        self.replay_buffer = ReplayBuffer(size=self.buffer_size)

    def buffer(self, observation):
        """Put observation into replay buffer."""
        self.replay_buffer.put(observation)

    def replay(self):
        """Retrieve observation of given batch size from replay buffer"""
        return self.replay_buffer.get(n=self.batch_size)

    def sync_networks(self):
        self.target_network.load_state_dict(self.online_network.state_dict())
