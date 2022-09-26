import json
import os
import time
import uuid

import numpy as np
import torch

from src.rl.applications.cfd.config import default_actions, training_conf
from src.rl.applications.cfd.environment import CfdEnvironment
from src.rl.applications.cfd.loading import load_datasets
from src.rl.applications.cfd.plot import plot_action_matrix, plot_data_recalculations, plot_data_scatter
from src.rl.applications.cfd.simulation import SimulationWrapper
from src.rl.applications.cfd.util import get_agent, get_reward_model, agent_log_episode, openfoam_available
from src.xutils.io import write_json
from src.xutils.logging import get_logger
from src.xutils.slurm.submission import parse_config

if not openfoam_available():
    raise RuntimeError("OpenFOAM not loaded!")

logger = get_logger(__file__)
config = parse_config()
logger.info(json.dumps(config, indent=2))

config = training_conf(**config)
job_id = f"{config.TRAINING_LABEL}-{uuid.uuid4()}" if config.TRAINING_LABEL else os.environ.get("SLURM_JOB_ID", None)

job_directory = os.path.abspath(f"outputs/cfd/{job_id}")
models_directory = os.path.abspath(f"models")
resampling_directory = f"{config.TRAINING_RESAMPLING_OUTPUT_PREFIX}/{job_id}"
plot_directory = os.path.join(job_directory, "plots")

profiles_directory = f"{job_directory}/profiles"
actions_directory = f"{job_directory}/actions"

os.makedirs(job_directory, exist_ok=True)
os.makedirs(models_directory, exist_ok=True)
os.makedirs(resampling_directory, exist_ok=True)
os.makedirs(plot_directory, exist_ok=True)
os.makedirs(profiles_directory, exist_ok=True)
os.makedirs(actions_directory, exist_ok=True)

config.DATA_SOURCE_DIRECTORIES.append(resampling_directory)

write_json(f"{job_directory}/config.json", config._asdict(), indent=2)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"training running on {device}")

reward_model_version = 0
reward_model_file = config.TRAINING_REWARD_MODEL_ID or os.path.join("models", f"reward-{job_id}-{reward_model_version}")
agent_model_file = config.TRAINING_AGENT_MODEL_ID or os.path.join("models", f"agent-{job_id}")

train_dataset, val_dataset = load_datasets(config, device)
state_size = train_dataset.dataset.tensors[0].shape[-1]

actions = torch.tensor(default_actions(state_size=state_size, action_config=config.TRAINING_AGENT_ACTION_CONFIG))
reward_model, reward_logs = get_reward_model(train_dataset, val_dataset, reward_model_file, config, device, state_size)

# x_train, y_train = train_dataset.dataset.tensors
# plot_data_distribution_hist(x_train.cpu().numpy(), y_train.cpu().numpy())
# plot_best_profiles(x_train.numpy(), y_train.numpy(), n=10, reverse=False)
# plot_profiles_with_same_drag(config.DATA_SOURCE_DIRECTORIES[0])
# config.DATA_SOURCE_DIRECTORIES.extend(glob.glob("resampling/test-1727020"))
# compare_profiles(glob.glob("resampling/**/*.json"), reward_model)

# plot_results()

env = CfdEnvironment(
    reward_model=reward_model,
    reward_type=config.TRAINING_REWARD_TYPE,
    n_actions=len(actions),
    mean_min=config.TRAINING_AGENT_MEAN_CONFIG["min"],
    mean_max=config.TRAINING_AGENT_MEAN_CONFIG["max"],
    actions=actions,
    actions_size=state_size,
    actions_type=config.TRAINING_AGENT_ACTION_CONFIG["type"],
    actions_stop_null=config.TRAINING_AGENT_ACTION_STOP_NULL,
    use_cumulative_reward=config.TRAINING_REWARD_USE_CUMULATIVE_REWARD,
    max_steps=config.TRAINING_AGENT_N_STEPS_PER_EPISODE
)

torch.save(env.reward_model, reward_model_file)
plot_data_scatter(model=env.reward_model, train_dataset=train_dataset, val_dataset=val_dataset, episode=0, file=os.path.join(plot_directory, "predictions-0"))

agent = get_agent(
    agent_model_file=agent_model_file,
    config=config, device=device,
    environment_n_actions=len(env.actions),
    state_size=state_size + 1
)

global_action_matrix = np.zeros((len(env.actions), config.TRAINING_AGENT_N_STEPS_PER_EPISODE))

logger.info(f"reward model has {env.reward_model.n_params:,} params")
logger.info(f"online network has {agent.online_network.n_params:,} params")
logger.info(f"environment using {len(env.actions)} actions")
logger.info("agent training")

training_logs, profiles_logging, profiles_sampling = [], [], []
n_episodes = config.TRAINING_AGENT_N_TOTAL_EPISODES
epsilon_step = (config.TRAINING_AGENT_EPSILON_START - config.TRAINING_AGENT_EPSILON_END) / config.TRAINING_AGENT_N_EXPLORATION_EPISODES

os.sched_setaffinity(os.getpid(), list(range(4)))
skip_blocks = 1
sim = SimulationWrapper(
    output_data_directory=resampling_directory,
    simulation_tmp_directory=os.path.abspath("tmp"),
    drag_validation_fn=lambda x: x is not None and x < 0.02,
    n_workers=12 + skip_blocks,
    n_cores_per_worker=4,
    use_hyperthreading_cores=False,
    skip_blocks=skip_blocks,
    verbose=False,
    delete=True
)

logger.info("training agent")
for episode in range(n_episodes):
    episode_start = time.time()

    agent.epsilon = max(agent.epsilon - epsilon_step, agent.epsilon_threshold)
    env.reset()

    env.state = env.state.to(device)
    state = env.state
    device = state.device
    valid_actions = env.get_valid_actions(state)

    episode_action_matrix = np.zeros_like(global_action_matrix)

    episode_logs = []

    logs = None
    for step in range(config.TRAINING_AGENT_N_STEPS_PER_EPISODE):
        action_idx, action_q_values = agent.action(state)
        action_idx = action_idx.to(device)
        action = valid_actions[action_idx]

        next_state, reward, done, info = env.step(action)

        observation = (
            state.view(1, -1),
            action_idx.view(1, -1),
            next_state.view(1, -1),
            reward.view(1, -1),
            done.view(1, -1),
        )
        agent.buffer(observation)

        episode_action_matrix[action_idx, step] += 1

        profile = state.view(-1).tolist()[:env.actions_size]
        profile = np.around(profile, 6)

        loss = agent.optimize().item()

        logs = dict(
            loss=loss,
            episode=episode,
            step=step + 1,
            action=action_idx.item(),
            reward=reward.item(),
            profile=profile,
            drag=info["drag"].item(),
            ps_mean=info["ps_mean"].item(),
            ss_mean=info["ss_mean"].item(),
            lr=agent.optimizer.param_groups[0]["lr"],
            mean_q=0.0 if action_q_values is None else action_q_values.mean().item(),
            max_q=0.0 if action_q_values is None else action_q_values.max().item()
        )
        episode_logs += [logs]
        state = next_state
        if done: break  # noqa

    log_entry = episode_logs[-1]

    if not config.TRAINING_REWARD_USE_CUMULATIVE_REWARD:
        log_entry["reward"] = sum([l["reward"] for l in episode_logs])

    log_entry["duration"] = round(1000 * (time.time() - episode_start))

    # aneb: cyclic LR scheduler can be applied all the time
    agent.scheduler.step()

    # aneb: sync every episode during warmup to avoid bootstrapping on random values
    if episode % config.TRAINING_AGENT_SYNC_FREQUENCY == 0 or episode < config.TRAINING_AGENT_WARMUP_STEPS:
        agent.sync_networks()

    if not episode:
        continue

    profiles_logging.append(log_entry)
    profiles_sampling += episode_logs
    training_logs.append(log_entry)

    global_action_matrix = 0.999 * global_action_matrix + episode_action_matrix

    if episode % config.TRAINING_AGENT_LOG_FREQUENCY == 0:
        profiles_logging.sort(key=lambda x: x["reward"])

        agent_log_episode(
            current_episode=episode,
            n_episodes=n_episodes,
            config=config,
            logs=training_logs,
            smooth=100,
            warmup=config.TRAINING_AGENT_WARMUP_STEPS,
            best_result=profiles_logging[-1],
            job_directory=job_directory,
            profiles_directory=profiles_directory,
            additional_logs=dict(eps=agent.epsilon, lr=agent.optimizer.param_groups[0]['lr'])
        )
        plot_action_matrix(
            matrix=global_action_matrix,
            actions=env.actions,
            title=f"episode {episode}",
            file=f"{actions_directory}/{episode}.png"
        )

        profiles_logging = []

    if episode % config.TRAINING_RESAMPLING_FREQUENCY == 0:
        profiles_sampling.sort(key=lambda x: x["reward"])
        sample = profiles_sampling[-1]

        sim.queue(profile=sample)
        profiles_sampling = []

    if episode % config.TRAINING_RETRAINING_FREQUENCY == 0:
        logger.info(f"retraining reward model on episode {episode}")
        reward_model_version += 1
        reward_model_file = os.path.join("models", f"reward-{job_id}-{reward_model_version}")
        train_dataset, val_dataset = load_datasets(config, device)

        new_reward_model, new_reward_logs = get_reward_model(train_dataset, val_dataset, reward_model_file, config, device, state_size)
        env.reward_model = new_reward_model
        torch.save(env.reward_model, reward_model_file)

        plot_data_scatter(model=env.reward_model, train_dataset=train_dataset, val_dataset=val_dataset, episode=episode, file=os.path.join(plot_directory, f"predictions-{episode}"))
        plot_data_recalculations(resampling_directory=resampling_directory, file=os.path.join(plot_directory, f"recalculations-{episode}"))

torch.save(agent, agent_model_file)

# aneb: no need to wait for remaining profiles
sim.done(cancel_futures=True)

logger.info("agent training done")
logger.info("done")
