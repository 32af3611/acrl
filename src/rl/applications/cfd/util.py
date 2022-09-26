import os
import os.path
import sys
import time
from copy import deepcopy

import pandas as pd
import torch
from torch import nn

from src.rl.applications.cfd.config import default_locations
from src.rl.applications.cfd.config import training_conf
from src.rl.applications.cfd.plot import plot_profile, plot_training
from src.rl.framework.agent import DoubleDQNAgent
from src.rl.framework.util import MultiLayerPerceptron
from src.rl.training.metrics import r2
from src.xutils.logging import get_logger

logger = get_logger(__file__)


def get_reward_model(train_dataset, val_dataset, reward_model_file, config, device, state_size, verbose=1):
    if os.path.exists(reward_model_file):
        logger.info(f"reward model at {reward_model_file} exists. loading it.")
        for _ in range(12):
            try:
                return torch.load(reward_model_file, map_location=device), None
            except Exception as e:
                logger.error(e)
            time.sleep(5)

        raise RuntimeError(f"could not load model at {reward_model_file}")

    reward_model_args = dict(
        input_size=state_size,
        output_size=1,
        hidden_sizes=config.TRAINING_REWARD_HIDDEN_SIZES,
        hidden_activation=nn.ReLU(),
        bias=False,  # not using bias somehow allows for larger models
        output_activation=None
    )

    reward_model = MultiLayerPerceptron(**reward_model_args).to(device)

    logger.info(f"model at {reward_model_file} does not exist. training it.")
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=config.TRAINING_REWARD_LR, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.TRAINING_REWARD_GAMMA)

    reward_model, logs = train_supervised_model(
        model=reward_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        metrics=dict(r2=r2),
        loss_fn=torch.nn.L1Loss(),
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=50000,
        start_save=100,
        patience_max=1000,
        min_delta=1e-6,
        monitor="val_loss",
        mode="min",
        verbose=verbose
    )

    logger.info(f"model training statistics: {logs}")
    return reward_model, logs


def get_agent(agent_model_file, config: training_conf, device, environment_n_actions, state_size):
    if os.path.exists(agent_model_file):
        logger.info(f"agent model at {agent_model_file} exists. loading it.")
        return torch.load(agent_model_file, map_location=device)

    agent_model_args = dict(
        input_size=state_size,
        output_size=environment_n_actions,
        hidden_sizes=config.TRAINING_AGENT_HIDDEN_SIZES,
        hidden_activation=nn.ReLU(),
        output_activation=None,
        bias=False
    )

    online_network = MultiLayerPerceptron(**agent_model_args).to(device).train()
    target_network = MultiLayerPerceptron(**agent_model_args).to(device).eval()
    target_network.load_state_dict(online_network.state_dict())

    optimizers = dict(
        adam=torch.optim.Adam,
        rmsprop=torch.optim.RMSprop,
        sgd=torch.optim.SGD
    )

    optimizer_class = optimizers[config.TRAINING_AGENT_OPTIMIZER["name"]]
    optimizer_args = {k: v for k, v in config.TRAINING_AGENT_OPTIMIZER.items() if k != "name"}

    agent_optimizer = optimizer_class(online_network.parameters(), lr=config.TRAINING_AGENT_LR_MIN, **optimizer_args)

    if config.TRAINING_AGENT_LR_MIN == config.TRAINING_AGENT_LR_MAX or not config.TRAINING_AGENT_LR_MAX:
        agent_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            agent_optimizer,
            gamma=config.TRAINING_AGENT_LR_DECAY
        )
    else:
        agent_scheduler = torch.optim.lr_scheduler.CyclicLR(
            agent_optimizer,
            base_lr=config.TRAINING_AGENT_LR_MIN,
            max_lr=config.TRAINING_AGENT_LR_MAX or config.TRAINING_AGENT_LR_MIN,
            step_size_up=config.TRAINING_AGENT_LR_CYCLE // 2,
            mode='exp_range',
            gamma=config.TRAINING_AGENT_LR_DECAY,
            scale_fn=None,
            scale_mode='iterations',
            cycle_momentum=False
        )

    agent = DoubleDQNAgent(
        online_network=online_network,
        target_network=target_network,
        n_actions=environment_n_actions,
        optimizer=agent_optimizer,
        scheduler=agent_scheduler,
        loss=nn.L1Loss(),
        batch_size=config.TRAINING_AGENT_BATCH_SIZE,
        gamma=config.TRAINING_AGENT_GAMMA,
        epsilon=config.TRAINING_AGENT_EPSILON_START,
        epsilon_threshold=config.TRAINING_AGENT_EPSILON_END,
        buffer_size=config.TRAINING_AGENT_REPLAY_BUFFER_SIZE
    )

    return agent


def openfoam_available():
    foam_loaded = bool(os.environ.get("FOAM_RUN", None))
    return foam_loaded


def iterate(model, optimizer, loss_fn, dataset, metrics, training):
    iteration_loss = 0.0
    iteration_steps = 0

    iteration_metrics = {name: 0.0 for name in metrics.keys()}

    for i, (x, y) in enumerate(dataset):
        if training:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(x)
                loss = loss_fn(outputs, y)

        iteration_loss += loss.item()
        for name, fn in metrics.items():
            iteration_metrics[name] += fn(outputs, y)

        iteration_steps += 1

    iteration_metrics = {name: float(value / iteration_steps) for name, value in iteration_metrics.items()}

    return dict(
        loss=iteration_loss / iteration_steps,
        **iteration_metrics
    )


def train_supervised_model(
        model,
        train_dataset,
        val_dataset,
        metrics,
        optimizer,
        scheduler,
        loss_fn,
        n_epochs=1,
        start_save=1,
        patience_max=1000,
        min_delta=1e-4,
        verbose=1,
        monitor="val_r2",
        mode="max"
):
    patience_left = patience_max
    best_logs = None
    best_weights = deepcopy(model.state_dict())

    best = -sys.maxsize if mode == "max" else sys.maxsize

    for epoch in range(n_epochs):
        patience_left -= 1

        if not patience_left and verbose:
            logger.info(f"{monitor} did not improve from {best:.4f} by {min_delta} for {patience_max} epochs.")
            break

        model.train()
        epoch_train_logs = iterate(model, optimizer, loss_fn, train_dataset, metrics, training=True)

        model.eval()
        epoch_val_logs = iterate(model, None, loss_fn, val_dataset, metrics, training=False)
        epoch_val_logs = {f"val_{k}": v for k, v in epoch_val_logs.items()}

        scheduler.step()
        epoch_logs = {**epoch_train_logs, **epoch_val_logs}
        current = epoch_val_logs[monitor]
        improvement = current - best if mode == "max" else best - current

        train_log = [f"{k}: {v:.8f}" for k, v in epoch_train_logs.items()]
        val_log = [f"{k}: {v:.8f}" for k, v in epoch_val_logs.items()]
        other_logs = [
            f"lr: {optimizer.param_groups[0]['lr']:.6f}",
            f"patience: {patience_left}"
        ]

        epoch_info = ", ".join(train_log + val_log + other_logs)

        if improvement > min_delta:
            if verbose:
                logger.info(f"epoch {epoch} - {monitor} improved by {improvement:.8f} from {best:.8f} to {current:.8f}.")
                logger.info(f"epoch {epoch} - {epoch_info}")

            patience_left = patience_max
            best = current
            best_logs = epoch_logs
            if epoch >= start_save:
                best_weights = deepcopy(model.state_dict())

        if not verbose or epoch % 100 != 0:
            continue

        logger.info(f"epoch {epoch} - {epoch_info}")

    model.load_state_dict(best_weights)
    return model, best_logs


def agent_log_episode(current_episode, n_episodes, config, logs, smooth, warmup, best_result, job_directory, profiles_directory, additional_logs):
    skip = ["action", "profile", "episode"]
    data = pd.DataFrame(logs)
    data.drop(labels=skip, axis=1, inplace=True)

    plot_training(
        losses=data["loss"],
        rewards=data["reward"],
        drags=data["drag"],
        ps_means=data["ps_mean"],
        ss_means=data["ss_mean"],
        mean_q_values=data["mean_q"],
        max_q_values=data["max_q"],
        lrs=data["lr"],
        steps=data["step"],
        exploration_end=config.TRAINING_AGENT_N_EXPLORATION_EPISODES,
        smooth=smooth,
        warmup=warmup,
        file=f"{job_directory}/result.png"
    )

    data.to_pickle(f"{job_directory}/result.pkl")

    train_logs = data.mean().to_dict()
    logs = ", ".join([f"{k}: {v:.8f}" if not isinstance(v, str) else f"{k}: {v[:6]}" for k, v in {**train_logs, **additional_logs}.items()])
    logger.info(f"episode {current_episode}/{n_episodes} - {logs}")

    profile_episode, profile_step, profile_reward, profile_drag, profile_coeffs = \
        best_result["episode"], best_result["step"], best_result["reward"], best_result["drag"], best_result["profile"]

    n_side = len(profile_coeffs) // 2
    locations = default_locations(n_side)

    plot_profile(
        file_name=f"{profiles_directory}/profile_episode_{profile_episode}.png",
        title=f"episode {profile_episode}, step {profile_step} - predicted drag: {profile_drag:.6f}, reward: {profile_reward:.6f}",
        locations=locations[:-1],
        coeff_ps=profile_coeffs[:n_side],
        coeff_ss=profile_coeffs[n_side:]
    )
