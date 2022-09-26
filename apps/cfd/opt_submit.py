import os.path
from copy import deepcopy

from src.rl.slurm.config import slurm_logs_dir, slurm_submissions_dir, slurm_pre_run_commands
from src.xutils.logging import get_logger
from src.xutils.slurm.submission import parse_args, SlurmSubmission

logger = get_logger(__file__)

cwd = os.getcwd()

training_time = [10 * 60]
setup_time = 15

slurm_config = {
    "PYTHONPATH": [cwd],
    "--partition": ["accelerated"],
    "--gres": ["gpu:4"],
    "--mem": [32000],
    "--nodes": [1],
    "--ntasks": [152],
    "--time": [t + setup_time for t in training_time],
    "--output": [f"{slurm_logs_dir}/slurm_%x_%j.out"],
    "--job-name": ["cfd-opt"]
}

script_config = {
    "DATA_SOURCE_DIRECTORIES": [
        [f"data/cfd"],
    ],

    "TRAINING_LABEL": ["training-large-range"],
    "TRAINING_REWARD_MODEL_ID": [None],
    "TRAINING_REWARD_HIDDEN_SIZES": [
        [1024, 1024]
    ],
    "TRAINING_REWARD_BATCH_SIZE": [64],
    "TRAINING_REWARD_LR": [1e-3],
    "TRAINING_REWARD_GAMMA": [0.9999],
    "TRAINING_REWARD_TYPE": [
        ["d_drag"]
    ],
    "TRAINING_REWARD_USE_CUMULATIVE_REWARD": [True],

    "TRAINING_AGENT_MODEL_ID": [None],
    "TRAINING_AGENT_N_TOTAL_EPISODES": [300000],
    "TRAINING_AGENT_N_EXPLORATION_EPISODES": [50000],
    "TRAINING_AGENT_N_STEPS_PER_EPISODE": [30],
    "TRAINING_AGENT_WARMUP_STEPS": [2500],
    "TRAINING_AGENT_LOG_FREQUENCY": [5000],
    "TRAINING_AGENT_EPSILON_START": [0.5],
    "TRAINING_AGENT_EPSILON_END": [0.0],
    "TRAINING_AGENT_BATCH_SIZE": [128],
    "TRAINING_AGENT_GAMMA": [1.0],
    "TRAINING_AGENT_SYNC_FREQUENCY": [10],
    "TRAINING_AGENT_REPLAY_BUFFER_SIZE": [50000],
    "TRAINING_AGENT_HIDDEN_SIZES": [
        [512, 512, 512]
    ],
    "TRAINING_AGENT_OPTIMIZER": [
        dict(name="rmsprop", momentum=0.0)
    ],
    "TRAINING_AGENT_LR_MIN": [1e-5],
    "TRAINING_AGENT_LR_MAX": [None],
    "TRAINING_AGENT_LR_CYCLE": [5000],
    "TRAINING_AGENT_LR_DECAY": [0.99999],
    "TRAINING_AGENT_ACTION_STOP_NULL": [False],
    "TRAINING_AGENT_ACTION_CONFIG": [
        dict(type="additive", magnitudes=[1e-3], use_null_action=True)
    ],

    "TRAINING_AGENT_MEAN_CONFIG": [
        dict(min=1.50e-3, max=2.50e-3),
        # dict(min=1.90e-3, max=2.10e-3)
    ],

    "TRAINING_RETRAINING_FREQUENCY": [10000, 10000000],
    "TRAINING_RESAMPLING_FREQUENCY": [100],
    "TRAINING_RESAMPLING_OUTPUT_PREFIX": ["resampling"]
}


def filter_fn(job_config, script_config):
    script_config = deepcopy(script_config)
    prefix = script_config["TRAINING_LABEL"]
    retrain = "no-retrain" if script_config["TRAINING_RETRAINING_FREQUENCY"] > 10000 else "retrain"
    n_steps = script_config["TRAINING_AGENT_N_STEPS_PER_EPISODE"]
    script_config["TRAINING_LABEL"] = f"{prefix}-{retrain}-{n_steps}"
    return job_config, script_config


yes, force = parse_args()
submission = SlurmSubmission(
    directory=slurm_submissions_dir,
    slurm_config_batch=slurm_config,
    script_config_batch=script_config,
    pre_run_commands=slurm_pre_run_commands,
    sbatch_flags=None,
    filter_fn=filter_fn,
    yes=yes, force=force
)

script = os.path.join(cwd, "apps/cfd/opt_run.py")
submission.submit_batch(script_name=script)
