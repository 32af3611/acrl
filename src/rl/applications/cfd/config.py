from collections import namedtuple

import numpy as np

training_conf = namedtuple("TrainingConfig", [
    "DATA_SOURCE_DIRECTORIES",

    "TRAINING_LABEL",
    "TRAINING_REWARD_MODEL_ID",
    "TRAINING_REWARD_HIDDEN_SIZES",
    "TRAINING_REWARD_BATCH_SIZE",
    "TRAINING_REWARD_LR",
    "TRAINING_REWARD_GAMMA",
    "TRAINING_REWARD_TYPE",
    "TRAINING_REWARD_USE_CUMULATIVE_REWARD",

    "TRAINING_AGENT_MODEL_ID",
    "TRAINING_AGENT_N_TOTAL_EPISODES",
    "TRAINING_AGENT_N_EXPLORATION_EPISODES",
    "TRAINING_AGENT_N_STEPS_PER_EPISODE",
    "TRAINING_AGENT_WARMUP_STEPS",
    "TRAINING_AGENT_LOG_FREQUENCY",
    "TRAINING_AGENT_EPSILON_START",
    "TRAINING_AGENT_EPSILON_END",
    "TRAINING_AGENT_GAMMA",
    "TRAINING_AGENT_BATCH_SIZE",
    "TRAINING_AGENT_SYNC_FREQUENCY",
    "TRAINING_AGENT_REPLAY_BUFFER_SIZE",
    "TRAINING_AGENT_HIDDEN_SIZES",
    "TRAINING_AGENT_OPTIMIZER",
    "TRAINING_AGENT_LR_MIN",
    "TRAINING_AGENT_LR_MAX",
    "TRAINING_AGENT_LR_CYCLE",
    "TRAINING_AGENT_LR_DECAY",
    "TRAINING_AGENT_MEAN_CONFIG",
    "TRAINING_AGENT_ACTION_CONFIG",
    "TRAINING_AGENT_ACTION_STOP_NULL",

    "TRAINING_RETRAINING_FREQUENCY",
    "TRAINING_RESAMPLING_FREQUENCY",
    "TRAINING_RESAMPLING_OUTPUT_PREFIX"
])

resampling_conf = namedtuple("ResamplingConf", [
    "RESAMPLING_LABEL",
    "RESAMPLING_PROFILES",
    "RESAMPLING_OUTPUT_DIRECTORY",
    "RESAMPLING_SIMULATION_DIRECTORY",
])


def default_locations(n_coeffs):
    start, end = 200, 900
    step = int((end - start) / (n_coeffs + 1))
    return (np.arange(start, end, step) / 1000).tolist()[:n_coeffs + 1]


def default_actions(state_size, action_config):
    if action_config["type"] not in ["additive", "multiplicative"]:
        raise ValueError(f"invalid action config")

    ps_actions, ss_actions = [], []

    magnitudes, use_null = [action_config[k] for k in ["magnitudes", "use_null_action"]]

    if action_config["type"] == "additive":
        single_per_side = np.concatenate([np.eye(state_size // 2) * step for step in magnitudes])
        ps_actions += (-single_per_side).tolist()
        ss_actions += single_per_side.tolist()

        ps_actions = np.concatenate([ps_actions, np.zeros_like(ps_actions)], axis=1)
        ss_actions = np.concatenate([np.zeros_like(ss_actions), ss_actions], axis=1)

        actions = np.concatenate([ps_actions, ss_actions], axis=0)

        if action_config["use_null_action"]:
            null_action = np.zeros((1, state_size))
            actions = np.concatenate([null_action, actions], axis=0)
        return actions

    if action_config["type"] == "multiplicative":
        single_per_side = np.concatenate([np.eye(state_size // 2) * step for step in magnitudes])
        single_per_side[single_per_side == 0] = 1

        ps_actions = np.concatenate([single_per_side, np.ones_like(single_per_side)], axis=1)
        ss_actions = np.concatenate([np.ones_like(single_per_side), single_per_side], axis=1)

        actions = np.concatenate([ps_actions, ss_actions], axis=0)

        if action_config["use_null_action"]:
            null_action = np.ones((1, state_size))
            actions = np.concatenate([null_action, actions], axis=0)

        return actions

    # pyplot.figure(figsize=(6, 10))
    # pyplot.imshow(actions, cmap="Blues")
    # pyplot.colorbar()
    # pyplot.tight_layout()
    # pyplot.show()
    raise ValueError(f"invalid action config")
