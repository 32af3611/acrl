import numpy as np
import torch
import torch.optim as opt
from baselines.deepq import replay_buffer

from src.rl.applications.mol import hyp
from src.rl.applications.mol.environment import Molecule
from src.rl.applications.mol.model.dqn import MolDQN
from src.rl.applications.mol.utils import get_fingerprint

REPLAY_BUFFER_CAPACITY = hyp.replay_buffer_size


class RewardMolecule(Molecule):
    """The molecule whose reward is the QED."""

    def __init__(self, discount_factor, **kwargs):
        """Initializes the class.

    Args:
      discount_factor: Float. The discount factor. We only
        care about the molecule at the end of modification.
        In order to prevent a myopic decision, we discount
        the reward at each step by a factor of
        discount_factor ** num_steps_left,
        this encourages exploration with emphasis on long term rewards.
      **kwargs: The keyword arguments passed to the base class.
    """
        super(RewardMolecule, self).__init__(**kwargs)
        self.discount_factor = discount_factor

    def _reward(self):
        """Reward of a state.
    Returns:
      Whatever the user chooses as reward. Default : 0
    """
        return 0


class Agent(object):
    def __init__(self, input_length, output_length, device):
        self.device = device
        self.dqn, self.target_dqn = (
            MolDQN(input_length, output_length).to(self.device),
            MolDQN(input_length, output_length).to(self.device),
        )
        for p in self.target_dqn.parameters():
            p.requires_grad = False
        self.replay_buffer = replay_buffer.ReplayBuffer(REPLAY_BUFFER_CAPACITY)
        self.optimizer = getattr(opt, hyp.optimizer)(
            self.dqn.parameters(), lr=hyp.learning_rate
        )

    def get_action(self, observations, epsilon_threshold):

        if np.random.uniform() < epsilon_threshold:
            action = np.random.randint(0, observations.shape[0])
        else:
            q_value = self.dqn.forward(observations.to(self.device)).cpu()
            action = torch.argmax(q_value).numpy()

        return action

    def update_params(self, batch_size, gamma, polyak, approx=False, triple=False, approx_reward=None,
                      model=None, model1=None, model2=None, model3=None):
        # update target network

        # sample batch of transitions
        states_smiles, _, rewards, next_states_smiles, dones = self.replay_buffer.sample(batch_size)
        states = [np.append(get_fingerprint(states_smiles[i, 0],
                                            hyp.fingerprint_length, hyp.fingerprint_radius),
                            float(states_smiles[i, 1]))
                  for i in range(len(states_smiles))]
        next_states = []
        for i in range(len(next_states_smiles)):
            arr = [np.append(get_fingerprint(next_states_smiles[i][j, 0], hyp.fingerprint_length,
                                             hyp.fingerprint_radius), float(next_states_smiles[i][j, 1]))
                   for j in range(len(next_states_smiles[i]))]
            next_states.append(arr)
        next_states = np.asarray(next_states)

        if approx:
            rewards = []
            for i in range(len(states_smiles)):
                if triple:
                    rewards.append(approx_reward(states_smiles[i, 0], model1, model2, model3))
                else:
                    rewards.append(approx_reward(states_smiles[i, 0], model))

        q_t = torch.zeros(batch_size, 1, requires_grad=False)
        v_tp1 = torch.zeros(batch_size, 1, requires_grad=False)
        for i in range(batch_size):
            state = (
                torch.FloatTensor(states[i])
                .reshape(-1, hyp.fingerprint_length + 1)
                .to(self.device)
            )
            q_t[i] = self.dqn(state)

            next_state = (
                torch.FloatTensor(next_states[i])
                .reshape(-1, hyp.fingerprint_length + 1)
                .to(self.device)
            )
            v_tp1[i] = torch.max(self.target_dqn(next_state))

        rewards = torch.FloatTensor(rewards).reshape(q_t.shape).to(self.device)
        q_t = q_t.to(self.device)
        v_tp1 = v_tp1.to(self.device)
        dones = torch.FloatTensor(dones).reshape(q_t.shape).to(self.device)

        # # get q values
        q_tp1_masked = (1 - dones) * v_tp1
        q_t_target = rewards + gamma * q_tp1_masked
        td_error = q_t - q_t_target

        q_loss = torch.where(
            torch.abs(td_error) < 1.0,
            0.5 * td_error * td_error,
            1.0 * (torch.abs(td_error) - 0.5),
        )
        q_loss = q_loss.mean()

        # backpropagate
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(self.dqn.parameters(), self.target_dqn.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        return q_loss
