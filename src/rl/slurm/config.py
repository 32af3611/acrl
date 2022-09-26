import os

slurm_pre_run_commands = [
    'echo `basename "$0"`',
    'eval "$(conda shell.bash hook)"',

    'module load cae/openfoam/7',
    'foamInit',
    'conda activate python >/dev/null 2>&1'
]

cwd = os.getcwd()
slurm_root_dir = os.path.join(cwd, "slurm")
slurm_logs_dir = os.path.join(slurm_root_dir, "logs")
slurm_submissions_dir = os.path.join(slurm_root_dir, "submissions")

os.makedirs(slurm_root_dir, exist_ok=True)
os.makedirs(slurm_logs_dir, exist_ok=True)
os.makedirs(slurm_submissions_dir, exist_ok=True)
