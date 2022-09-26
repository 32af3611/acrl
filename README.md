# Actively learning complex reward functions for Reinforcement Learning

# Setting up the repository

Due to file size limitations, code and data are split up into different archives.
To set up the repository, you need to perform the following steps:

1. Download all files from the supplementary information section and place them into
some directory.
2. Run ``bash unpack.sh`` 

Everything should now be placed under an ``acrl`` directory, which is assumed
to be the project's root directory in the following.
To reproduce the results in the manuscript, please consult the task-specific 
setup description below.

Now ``cd`` into ``acrl``.

# Setting up Python
First set up the conda environment by running
```
conda create -y -n acrl python=3.7
conda init
conda activate acrl
```

Run ``bash setup.sh`` to install all necessary dependencies.

# Reproducing molecular tasks
To reproduce the benchmark *logP/QED* experiments:

- **Oracle experiment**

```PYTHONPATH=. python apps/mol/run_real_reward.py --task "logP" --file 0```

- **Non-updated model**

```PYTHONPATH=. python apps/mol/run_one_model.py --task "logP" --file 0```

- **ACRL**

```PYTHONPATH=. python apps/mol/run_three_models.py --task "logP" --file 0 --points 400 --frequency 500 --mode "st_dev" ```

To reproduce the molecular improvement task:

```PYTHONPATH=. python apps/mol/run_mol_opt.py --file 0 --points 400 --frequency 500 --mode "random" ```

List of arguments (can be directly modified when using the above commands):

`--task`: the benchmark task at hand, can either be "logP" or "QED".

`--file`: csv output file name, by default a '0.csv' file will be created in the output directory.

`--points`: number of points to sample in ACRL, default 400 points.

`--frequency`: frequency of the Active Learning component, default every 500 episodes.

`--mode`: uncertainty sampling strategy, use "st_dev", "random" or "bin" respectively for standard
deviation based,
random or bin-based selection. See manuscript SI for more details.

# Reproducing drag optimization

- **OpenFOAM setup**. To reproduce the drag optimization results, you first need to install
  OpenFOAM v7 for [Ubuntu](https://openfoam.org/download/7-ubuntu/) or other
  [Linux](https://openfoam.org/download/7-linux/) distributions.
  You also need to install the [Swak4FOAM](https://openfoamwiki.net/index.php/Contrib/swak4Foam)
  extension.
- **Python setup.** Create conda environment from [requirements_cfd.txt](requirements_cfd.txt).

You can then run

```
PYTHONPATH=. python apps/cfd/opt_run.py --config apps/cfd/opt_conf.json
```

from the project's root directory.
Under [outputs](outputs), you can watch your training progress as well
the results of our two experiments.
Note that the script starts several background jobs for CFD simulations.
The exact number depends on the number of cores on your machine.
We ran training on a machine with 76 physical cores.
Due to high computational cost of each simulation, we suggest to use
a similar hardware setup for reproduction.

To reproduce results with both constraint variations, update the
[configuration](apps/cfd/opt_conf.json) sections to include either

```
"TRAINING_AGENT_MEAN_CONFIG": {
    "min": 1.9e-3,
    "max": 2.1e-3
  }
```

or

```
"TRAINING_AGENT_MEAN_CONFIG": {
    "min": 1.5e-3,
    "max": 2.5e-3
  }
```
