import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import QED
from sklearn.model_selection import train_test_split

from src.rl.applications.mol import utils, hyp
from src.rl.applications.mol.agent import Agent
from src.rl.applications.mol.model.model_definitions import train
from src.rl.applications.mol.utils import penalized_logp


def run(property, model=None, approx_reward=None, model_update=False, frequency=0, column="logP"):
    """Executes the reinforcement learning training using the MolDQN method described in paper.
        Args:
          property: subclass of RewardMolecule, found in agent.py
            where the task reward is defined
          model: instance of class MolDQN (see dqn.py)
            specific to tasks where predictions of previously trained ML models (on given properties)
            are used as task rewards. Only PyTorch trained models (on QM9 dataset) are used for now (model directory)
          approx_reward: function representing how the approximate reward is calculated
          model_update: Boolean. Whether model needs to be retrained to better predict properties used as rewards.
            Stems from the fact that SMILES generated during training generally exceed 9 heavy atoms in QM9 dataset
            (used for training), so model can be retrained on newly generated SMILES with real property value
          frequency: Integer. The frequency with which the model is retrained on QM9 dataset plus newly generated SMILES
          column : String. The name of the column in mol/raw_dataset used as reward in model property estimation task.
        Returns :
            episodes_list : list of episodes of training
            SMILES_list : list of all SMILES generated during training
            rewards_list : list of values of rewards as defined in environment
            losses_list : list of training losses
        """

    episodes = 0  # number of episodes, get updated as training happens
    iterations = 200000  # number of iterations, for final number of episodes = iterations / hyp.max_steps_per_episode
    update_interval = 20  # how often the agent parameters get updated
    batch_size = 128
    num_updates_per_it = 1  # how many updates per iteration

    # initialize lists
    episodes_list, smiles_list, rewards_list, losses_list = [], [], [], []

    # train on cuda if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"The device used is : {device}")

    # DQN Inputs and Outputs:
    # input: appended action (fingerprint_length + 1) .
    # Output size is (1).

    agent = Agent(hyp.fingerprint_length + 1, 1, device)

    environment = property(
        discount_factor=hyp.discount_factor,
        atom_types=set(hyp.atom_types),
        init_mol=hyp.start_molecule,
        allow_removal=hyp.allow_removal,
        allow_no_modification=hyp.allow_no_modification,
        allow_bonds_between_rings=hyp.allow_bonds_between_rings,
        allowed_ring_sizes=set(hyp.allowed_ring_sizes),
        max_steps=hyp.max_steps_per_episode,
    )
    # initialize environment
    environment.initialize()

    eps_threshold = 1.0
    batch_losses = []

    # if in a model estimation task, model needs to be updated, use QM9 data and prepare X and y
    if model_update:
        data = pd.read_csv("data/mol/raw_dataset.csv")
        x_data = [utils.get_fingerprint(smiles, hyp.fingerprint_length, hyp.fingerprint_radius) for smiles in data["SMILES"]]
        y_data = data[column].tolist()

    # main training loop to go over all iterations
    for it in range(iterations):
        steps_left = hyp.max_steps_per_episode - environment.num_steps_taken

        # Compute a list of possible valid actions. (Here valid_actions stores the states after taking possible actions)
        valid_actions = list(environment.get_valid_actions())

        # Append each valid action to steps_left and store in observations.
        observations = np.zeros((len(valid_actions), hyp.fingerprint_length + 1))
        for act_idx, act in enumerate(valid_actions):
            observations[act_idx, :-1] = utils.get_fingerprint(act, hyp.fingerprint_length, hyp.fingerprint_radius)
            observations[act_idx, -1] = steps_left

        observations = torch.Tensor(observations)
        # Get action through epsilon-greedy policy with the following scheduler.
        # eps_threshold = hyp.epsilon_end + (hyp.epsilon_start - hyp.epsilon_end) * \
        #     math.exp(-1. * it / hyp.epsilon_decay)

        a = agent.get_action(observations, eps_threshold)

        # Find out the new state
        action_smiles = valid_actions[a]
        # Take a step based on the action
        result = environment.step(action_smiles)

        action_smiles = np.append(
            action_smiles,
            steps_left,
        )

        next_state_smiles, reward, done = result

        # Compute number of steps left
        steps_left = hyp.max_steps_per_episode - environment.num_steps_taken

        # Append steps_left to the new state and store in next_state
        next_state = utils.get_fingerprint(
            next_state_smiles, hyp.fingerprint_length, hyp.fingerprint_radius
        )  # (fingerprint_length)
        actions_smiles = []
        for act_idx, act in enumerate(environment.get_valid_actions()):
            actions_smiles.append([act, steps_left])
        actions_smiles = np.array(actions_smiles)

        # Update replay buffer

        agent.replay_buffer.add(
            obs_t=action_smiles,  # smiles
            action=0,  # No use
            reward=reward,
            obs_tp1=actions_smiles,  # all possible smiles
            done=float(result.terminated),
        )

        # in original MolDQN_pytorch implementation, fp arrays were stored in replay buffer, but due to memory issues,
        # # replay buffer now contains SMILES, and fingerprints are only generated when needed

        if done:  # end of episode

            if episodes != 0 and len(batch_losses) != 0:
                # add episode number to episodes_list
                episodes_list.append(episodes)
                print(
                    "SMILES generated in episode {} is {}".format(
                        episodes, next_state_smiles
                    )
                )
                # add smiles of final step to SMILES_list
                smiles_list.append(next_state_smiles)
                print(
                    "reward of final molecule at episode {} is {}".format(
                        episodes, reward
                    )
                )
                # update rewards list
                rewards_list.append(reward)
                print(
                    "mean loss in episode {} is {}".format(
                        episodes, np.array(batch_losses).mean()
                    )
                )
                # update losses list
                losses_list.append(np.array(batch_losses).mean())
                print("\n")

            # where model updates happen in model property estimation tasks

            if model_update:
                if episodes != 0 and episodes % frequency == 0:
                    x = x_data
                    y = y_data
                    for smiles in smiles_list:
                        arr = utils.get_fingerprint(smiles, hyp.fingerprint_length, hyp.fingerprint_radius)
                        x.append(arr)
                        if column == "logP":
                            y.append(penalized_logp(Chem.MolFromSmiles(smiles)))
                        elif column == "QED":
                            y.append(QED.qed(Chem.MolFromSmiles(smiles)))
                        else:
                            print("No such column is found in dataset !!")
                    y = pd.Series(y)
                    X_train, X_valtest, y_train, y_valtest = train_test_split(x, y, test_size=0.2, random_state=0)
                    environment.model = train(environment.model, X_train, y_train, X_valtest, y_valtest)

            episodes += 1
            eps_threshold *= 0.99907
            batch_losses = []
            environment.initialize()

        # where agent parameters get updated
        if it % update_interval == 0 and agent.replay_buffer.__len__() >= batch_size:
            for update in range(num_updates_per_it):
                if model_update:
                    loss = agent.update_params(batch_size, hyp.gamma, hyp.polyak, approx=True,
                                               approx_reward=approx_reward, model=environment.model)
                else:
                    loss = agent.update_params(batch_size, hyp.gamma, hyp.polyak)
                loss = loss.item()
                batch_losses.append(loss)

    return episodes_list, smiles_list, rewards_list, losses_list
