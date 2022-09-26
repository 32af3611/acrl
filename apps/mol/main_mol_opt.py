import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import QED

from src.rl.applications.mol import utils, hyp
from src.rl.applications.mol.agent_mol_opt import Agent
from src.rl.applications.mol.model.dqn import MolDQN
from src.rl.applications.mol.model.model_definitions import train, triple_predict, multiple_train
from src.rl.applications.mol.utils import penalized_logp, bin_based_selection, get_coords_from_smiles, xtb_calc


def triple_run(property, chk_dir, start_from_chk=None, model1=None, model2=None, model3=None, approx_reward=None,
               model_update=False, update_mode="random", n_points=400, frequency=0, column="logP"):
    """Executes the reinforcement learning training using the MolDQN method described in paper.
        Args:
          property: subclass of RewardMolecule, found in agent.py
            where the task reward is defined
          chk_dir : the directory where checkpoints will be stored (in general TB_LOG_PATH + "/checkpoints"
          start_from_chk : the checkpoint directory from which to start, default : None (starting from scratch)
          model1 : instance of class MolDQN (see dqn.py)
            specific to tasks where predictions of previously trained ML models (on given properties)
            are used as task rewards. Only PyTorch trained models (on QM9 dataset) are used for now (model/ directory)
          model2 : same as model1
          model3 : same as above
          approx_reward: function representing how the approximate reward is calculated
          model_update: Boolean. Whether model needs to be retrained to better predict properties used as rewards.
            Stems from the fact that SMILES generated during training generally exceed 9 heavy atoms in QM9 dataset
            (used for training), so model can be retrained on newly generated SMILES with real property value
          update_mode: String. Update mode,"random" selects random points from generated smiles to add to train set,
                                       "st_dev" selects points with the highest standard deviation in model predictions,
                                       "full" appends all newly generated SMILES to dataset and retrains (very costly)
                                       "bin" implements a novel strategy to improve from st_dev by selecting points with
                                       higher st_deviations from each bin of the distribution rather than all the points
                                       not available for the mol_opt task, in which points are selected by st_dev
          n_points: number of points to be appended to the train set for model update task.
            Only relevant for update="random" or "st_dev"
            The number of points should never exceed the number of SMILES generated throughout the episodes !
          frequency: Integer. The frequency with which the model is retrained on QM9 dataset plus newly generated SMILES
          column : String. The name of the column in mol/raw_dataset used as reward in model property estimation task.
        Returns :
            episodes_list : list of episodes of training
            SMILES_list : list of all SMILES generated during training
            rewards_list : list of values of rewards as defined in environment
            losses_list : list of training losses
        """

    iterations = 25000  # number of iterations, for final number of episodes = iterations / hyp.max_steps_per_episode
    update_interval = 20  # how often the agent parameters get updated
    batch_size = 128
    num_updates_per_it = 1  # how many updates per iteration
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(f"The device used is : {device}")

    # initialize
    episodes = 0
    episodes_list, original_smiles_list, smiles_list, rewards_list, losses_list = [], [], [], [], []
    new_smiles_list = []
    start_molecule = hyp.start_molecule
    environment = property(
        discount_factor=hyp.discount_factor,
        atom_types=set(hyp.atom_types),
        init_mol=start_molecule,
        allow_removal=hyp.allow_removal,
        allow_no_modification=hyp.allow_no_modification,
        allow_bonds_between_rings=hyp.allow_bonds_between_rings,
        allowed_ring_sizes=set(hyp.allowed_ring_sizes),
        max_steps=hyp.max_steps_per_episode,
    )
    print(f'The start molecule is {hyp.start_molecule}')
    # DQN Inputs and Outputs:
    # input: appended action (fingerprint_length + 1) .
    # Output size is (1).

    agent = Agent(hyp.fingerprint_length + 1, 1, device)
    eps_threshold = 1.0
    batch_losses = []
    environment.model1 = model1
    environment.model2 = model2
    environment.model3 = model3
    environment.ref_homo = hyp.reference_homo
    environment.ref_lumo = hyp.reference_lumo
    if start_from_chk is not None:
        if not os.path.exists(start_from_chk):
            exit("Error: Did not find checkpoint %s" % start_from_chk)
        with open(start_from_chk + 'summary.pkl', 'rb') as f:
            episodes_list = pickle.load(f)
            original_smiles_list = pickle.load(f)
            smiles_list = pickle.load(f)
            rewards_list = pickle.load(f)
            losses_list = pickle.load(f)
            eps_threshold = pickle.load(f)
            batch_losses = pickle.load(f)
        episodes = episodes_list[len(episodes_list) - 1] + 1
        if column == "HOMO_LUMO":
            model1, model2, model3 = MolDQN(2048, 2), MolDQN(2048, 2), MolDQN(2048, 2)
        else:
            model1, model2, model3 = MolDQN(2048, 1), MolDQN(2048, 1), MolDQN(2048, 1)

        #device = torch.device('cpu')
        model1.load_state_dict(torch.load(start_from_chk + 'model1.pth', map_location=device))
        model1.eval()
        model2.load_state_dict(torch.load(start_from_chk + 'model2.pth', map_location=device))
        model2.eval()
        model3.load_state_dict(torch.load(start_from_chk + 'model3.pth', map_location=device))
        model3.eval()
        with open(start_from_chk + 'environment.pkl', 'rb') as f:
            environment = pickle.load(f)
        environment.model1 = model1
        environment.model2 = model2
        environment.model3 = model3
        with open(start_from_chk + 'agent.pkl', 'rb') as f:
            agent = pickle.load(f)

    environment.initialize()

    # if in a model estimation task, model needs to be updated, use QM9 data and prepare X and y
    if model_update:
        new_smiles_list = []
        data = pd.read_csv("data/mol/raw_dataset.csv")
        '''x_data = [utils.get_fingerprint(smiles, hyp.fingerprint_length, hyp.fingerprint_radius)
                  for smiles in data["SMILES"]]'''
        if column == "HOMO_LUMO":
            data.dropna(subset=["HOMO"], inplace=True)
            data.dropna(subset=["HOMO"], inplace=True)
            x_data = data["SMILES"].tolist()
            y_data = data[["HOMO", "LUMO"]].values.tolist()
        else:
            x_data = data["SMILES"].tolist()
            y_data = data[column].tolist()
        train_size = 0.8
        random_state = np.random.RandomState(seed=1)
        train_1 = random_state.choice(range(len(x_data)), int(train_size * len(x_data)), replace=False).tolist()
        train_2 = random_state.choice(range(len(x_data)), int(train_size * len(x_data)), replace=False).tolist()
        train_3 = random_state.choice(range(len(x_data)), int(train_size * len(x_data)), replace=False).tolist()
        total = list(range(len(x_data)))
        test_1 = list(set(total) - set(train_1))
        test_2 = list(set(total) - set(train_2))
        test_3 = list(set(total) - set(train_3))

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
            action=[environment.ref_homo, environment.ref_lumo],  # No use
            reward=reward,
            obs_tp1=actions_smiles,  # all possible smiles
            done=float(result.terminated),
        )

        # in original MolDQN_pytorch implementation, fp arrays were stored in replay buffer, but due to memory issues,
        # # replay buffer now contains SMILES, and fingerprints are only generated when needed

        if done:  # end of episode

            if episodes != 0:
                original_smiles_list.append(start_molecule)
                print(
                    "initial molecule at episode {} is {}, with HOMO {} and LUMO {}".format(
                        episodes, start_molecule, environment.ref_homo, environment.ref_lumo
                    )
                )
                # add episode number to episodes_list
                episodes_list.append(episodes)
                print(
                    "SMILES generated in episode {} is {}".format(
                        episodes, next_state_smiles
                    )
                )
                # add smiles of final step to SMILES_list
                smiles_list.append(next_state_smiles)
                new_smiles_list.append(next_state_smiles)
                print(
                    "reward of final molecule at episode {} is {}".format(
                        episodes, reward
                    )
                )
                # update rewards list
                rewards_list.append(reward)
                #if len(batch_losses) != 0:
                print(
                    "mean loss in episode {} is {}".format(
                        episodes, np.array(batch_losses).mean()
                    )
                )
                # update losses list
                losses_list.append(np.array(batch_losses).mean())
            print("\n")

            # where model updates happen in model property estimation tasks

            df = pd.read_csv("data/mol/raw_dataset.csv")
            df.dropna(subset=["HOMO"], inplace=True)
            mol9 = [df['SMILES'].iloc[i] for i in range(len(df)) if
                    Chem.MolFromSmiles(df['SMILES'].iloc[i]).GetNumHeavyAtoms() == 9]
            start_molecule = random.choice(mol9)
            reference_homo = df[df['SMILES'] == start_molecule]['HOMO'].values[0]
            reference_lumo = df[df['SMILES'] == start_molecule]['LUMO'].values[0]

            environment2 = property(
                discount_factor=hyp.discount_factor,
                atom_types=set(hyp.atom_types),
                init_mol=start_molecule,
                allow_removal=hyp.allow_removal,
                allow_no_modification=hyp.allow_no_modification,
                allow_bonds_between_rings=hyp.allow_bonds_between_rings,
                allowed_ring_sizes=set(hyp.allowed_ring_sizes),
                max_steps=hyp.max_steps_per_episode,
            )
            environment2.ref_homo = reference_homo
            environment2.ref_lumo = reference_lumo
            environment2.model1 = model1
            environment2.model2 = model2
            environment2.model3 = model3

            if model_update:
                if episodes != 0 and episodes % frequency == 0:
                    new_indices = []

                    for smiles in new_smiles_list:
                        # arr = utils.get_fingerprint(smiles, hyp.fingerprint_length, hyp.fingerprint_radius)
                        if column == "logP":
                            y_data.append(penalized_logp(Chem.MolFromSmiles(smiles)))
                            x_data.append(smiles)
                            new_indices.append(len(x_data) - 1)
                        elif column == "QED":
                            y_data.append(QED.qed(Chem.MolFromSmiles(smiles)))
                            x_data.append(smiles)
                            new_indices.append(len(x_data) - 1)
                        elif column == "HOMO_LUMO":
                            suffix = 0
                            conversion_method = 'rdkit'
                            (coords, elements) = get_coords_from_smiles(smiles, suffix, conversion_method)
                            results = xtb_calc(coords, elements, opt=False)
                            if results["HOMO"] is None or results["LUMO"] is None:
                                pass
                            else:
                                y_data.append([results["HOMO"], results["LUMO"]])
                                x_data.append(smiles)
                                new_indices.append(len(x_data) - 1)
                        else:
                            exit("Error : No such column is found in dataset !!")

                    test_1 += new_indices
                    test_2 += new_indices
                    test_3 += new_indices

                    new_selected_indices_1 = []
                    new_selected_indices_2 = []
                    new_selected_indices_3 = []

                    if update_mode == "full" or len(test_1) <= n_points:
                        new_selected_indices_1 = new_indices
                        new_selected_indices_2 = new_indices
                        new_selected_indices_3 = new_indices
                    elif update_mode == "random":
                        new_selected_indices_1 = random.sample(test_1, n_points)
                        new_selected_indices_2 = random.sample(test_2, n_points)
                        new_selected_indices_3 = random.sample(test_3, n_points)
                    elif update_mode == "st_dev":
                        tests = [test_1, test_2, test_3]
                        new_selected_indices = []
                        for test in tests:
                            test_smiles_list = [x_data[index] for index in test]
                            st_dev_list = [np.std(triple_predict(Chem.MolFromSmiles(smiles),
                                                                 environment.model1,
                                                                 environment.model2,
                                                                 environment.model3),
                                                  axis=0) for smiles in test_smiles_list]
                            df = pd.DataFrame()
                            df['index'] = test
                            if column == "HOMO_LUMO":
                                df['st_dev1'] = [st_dev_list[i][0] for i in range(len(st_dev_list))]
                                df['st_dev2'] = [st_dev_list[i][1] for i in range(len(st_dev_list))]
                                df = df.sort_values(['st_dev1', 'st_dev2'], ascending=[False, False]).head(n_points)
                            else:
                                df['st_dev'] = st_dev_list
                                df = df.sort_values('st_dev', ascending=False).head(n_points)
                            new_selected_indices.append(df['index'].values.tolist())
                        new_selected_indices_1, new_selected_indices_2, new_selected_indices_3 = new_selected_indices

                    elif update_mode == "bin":
                        if column == "HOMO_LUMO":
                            exit("Error : no bin update mode for the mol_opt task, "
                                 "please choose the st_dev or random mode")
                        tests = [test_1, test_2, test_3]
                        new_selected_indices = []
                        for test in tests:
                            test_smiles_list = [x_data[index] for index in test]
                            new_selected_indices.append(bin_based_selection(test_smiles_list, test, n_points, 20,
                                                                            environment.model1,
                                                                            environment.model2,
                                                                            environment.model3))
                        new_selected_indices_1, new_selected_indices_2, new_selected_indices_3 = new_selected_indices

                    else:
                        exit("Error: update_mode should be either full, random, st_dev or bin")

                    train_1 += new_selected_indices_1
                    train_2 += new_selected_indices_2
                    train_3 += new_selected_indices_3

                    total = list(range(len(x_data)))
                    test_1 = list(set(total) - set(train_1))
                    test_2 = list(set(total) - set(train_2))
                    test_3 = list(set(total) - set(train_3))

                    X_train_1 = [x_data[index] for index in train_1]
                    X_train_2 = [x_data[index] for index in train_2]
                    X_train_3 = [x_data[index] for index in train_3]
                    X_valtest_1 = [x_data[index] for index in test_1]
                    X_valtest_2 = [x_data[index] for index in test_2]
                    X_valtest_3 = [x_data[index] for index in test_3]
                    smiles_entries = [X_train_1, X_train_2, X_train_3, X_valtest_1, X_valtest_2, X_valtest_3]
                    fps = []
                    for entry in smiles_entries:
                        fp = [utils.get_fingerprint(smiles, hyp.fingerprint_length, hyp.fingerprint_radius)
                              for smiles in entry]
                        fps.append(fp)
                    X_train_1, X_train_2, X_train_3, X_valtest_1, X_valtest_2, X_valtest_3 = fps

                    y_train_1 = [y_data[index] for index in train_1]
                    y_train_2 = [y_data[index] for index in train_2]
                    y_train_3 = [y_data[index] for index in train_3]
                    y_valtest_1 = [y_data[index] for index in test_1]
                    y_valtest_2 = [y_data[index] for index in test_2]
                    y_valtest_3 = [y_data[index] for index in test_3]

                    if column == "HOMO_LUMO":
                        environment2.model1 = multiple_train(environment.model1,
                                                             X_train_1, y_train_1, X_valtest_1, y_valtest_1)
                        environment2.model2 = multiple_train(environment.model2,
                                                             X_train_2, y_train_2, X_valtest_2, y_valtest_2)
                        environment2.model3 = multiple_train(environment.model3,
                                                             X_train_3, y_train_3, X_valtest_3, y_valtest_3)
                    else:
                        environment2.model1 = train(environment.model1, X_train_1, y_train_1, X_valtest_1, y_valtest_1)
                        environment2.model2 = train(environment.model2, X_train_2, y_train_2, X_valtest_2, y_valtest_2)
                        environment2.model3 = train(environment.model3, X_train_3, y_train_3, X_valtest_3, y_valtest_3)

                    new_smiles_list = []

            episodes += 1
            eps_threshold *= 0.99907
            batch_losses = []
            environment = environment2
            environment.initialize()

        # where agent parameters get updated
        if it % update_interval == 0 and agent.replay_buffer.__len__() >= batch_size:
            for update in range(num_updates_per_it):
                if model_update:
                    loss = agent.update_params(batch_size, hyp.gamma, hyp.polyak, approx=True, triple=True,
                                               approx_reward=approx_reward,
                                               model1=environment.model1,
                                               model2=environment.model2,
                                               model3=environment.model3)
                    print(loss)
                else:
                    loss = agent.update_params(batch_size, hyp.gamma, hyp.polyak)
                loss = loss.item()
                batch_losses.append(loss)

        if episodes % 500 == 0:
            chk_dir_here = chk_dir + "/checkpoint_%i/" % episodes
            if not os.path.exists(chk_dir_here):
                os.makedirs(chk_dir_here)

            with open(chk_dir_here + 'summary.pkl', 'wb') as output:
                pickle.dump(episodes_list, output)
                pickle.dump(original_smiles_list, output)
                pickle.dump(smiles_list, output)
                pickle.dump(rewards_list, output)
                pickle.dump(losses_list, output)
                pickle.dump(eps_threshold, output)
                pickle.dump(batch_losses, output)

            torch.save(model1.state_dict(), chk_dir_here + 'model1.pth')
            torch.save(model2.state_dict(), chk_dir_here + 'model2.pth')
            torch.save(model3.state_dict(), chk_dir_here + 'model3.pth')

            with open(chk_dir_here + 'environment.pkl', 'wb') as output:
                pickle.dump(environment, output)

            with open(chk_dir_here + 'agent.pkl', 'wb') as output:
                pickle.dump(agent, output)

    return episodes_list, original_smiles_list, smiles_list, rewards_list
