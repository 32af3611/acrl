import itertools
import json
import os
import subprocess
import sys
from hashlib import blake2b


def parse_config():
    args = sys.argv

    if "--config" not in args:
        raise ValueError("missing '--config' option")

    config_arg_idx = args.index("--config")
    config_file = args[config_arg_idx + 1]

    if not os.path.exists(config_file):
        raise ValueError(f"config path {config_file} does not exist")

    with open(config_file, "r") as f:
        return json.load(f)


def parse_args():
    args = [arg[1:] for arg in sys.argv if arg.startswith("-")]
    yes = "y" in args
    force = "f" in args

    return yes, force


def get_config_product(config):
    config = list(itertools.product(*[[(k, single) for single in multi] for k, multi in config.items()]))
    config = [dict(p) for p in config]
    return config


class SlurmSubmission:
    def __init__(self, directory, slurm_config_batch, script_config_batch, pre_run_commands, sbatch_flags=None, filter_fn=None, yes=False, force=False):
        self.directory = directory
        self.slurm_config_batch = slurm_config_batch
        self.script_config_batch = script_config_batch
        self.pre_run_commands = pre_run_commands
        self.sbatch_flags = sbatch_flags or ''
        self.filter_fn = filter_fn

        self.yes = yes
        self.force = force

        os.makedirs(self.directory, exist_ok=True)

    def _check_config(self, config):
        invalid = [(k, v) for k, v in config.items() if not isinstance(v, list)]

        if invalid:
            messages = [f"value for key '{k}' must be a list, got {v} ({type(v).__name__}): " for k, v in invalid]
            messages = "\n".join(messages)
            raise TypeError(messages)

    def hash(self, value):
        if type(value) not in [dict, str]:
            raise NotImplemented(f"hashing for type {type(value)} not implemented")

        if isinstance(value, dict):
            value = json.dumps(value, sort_keys=True, indent=False)

        b = blake2b(digest_size=4)
        b.update(value.encode("utf8"))
        return b.hexdigest()

    def _exec_cmd(self, command, raise_error=True):
        code, result = subprocess.getstatusoutput(command)

        if code != 0 and raise_error:
            raise TypeError(result)

        if code != 0 and not raise_error:
            print(result)

        return result

    def _ask_for_yes(self, message):
        user_input = input(message + " (y/n?)\n")

        if user_input not in ["", "y", "n"]:
            print(f"received invalid input '{user_input}'. skipping.")
            return False

        return user_input != "n"

    def submit_single(self, job_config, script_config, script_name, pre_run_commands, return_script_path=False):
        submission_hash = "".join([self.hash(v) for v in [job_config, script_config, script_name]])

        script_path = os.path.abspath(f"{self.directory}/submission_{submission_hash}.sh")
        config_path = os.path.abspath(f"{self.directory}/submission_{submission_hash}.json")

        if os.path.exists(script_path) and not self.force:
            print(f"script with hash {submission_hash} already exists! not submitting it a second time.")
            return

        head_lines = ["#!/bin/sh"]
        sbatch_lines = [f"#SBATCH {k}=\"{v}\"" for k, v in job_config.items() if k.startswith("--")]
        env_lines = [f"export {k}=\"{v}\"" for k, v in job_config.items() if not k.startswith("--")]

        command_lines = [
            f'python -u {script_name} --config {config_path}',
            '\n'
        ]

        blocks = [head_lines, sbatch_lines, env_lines, pre_run_commands, command_lines]

        blocks = ["\n".join(block) for block in blocks]
        script = "\n\n".join(blocks)

        with open(script_path, "w", encoding="utf8") as f:
            f.write(script)

        with open(config_path, "w", encoding="utf8") as f:
            json.dump(script_config, f, indent=2)

        if return_script_path:
            return script_path

        cmd = f"sbatch {self.sbatch_flags} \"{script_path}\""

        print(cmd)
        result = self._exec_cmd(cmd, raise_error=False)
        print(result)
        return result

    def submit_batch(self, script_name):
        submission_params = get_config_product(self.slurm_config_batch)
        model_params = get_config_product(self.script_config_batch)

        configs = list(itertools.product(submission_params, model_params))

        if self.filter_fn:
            len_before = len(configs)
            configs = [self.filter_fn(job_config, script_config) for job_config, script_config in configs]
            configs = [p for p in configs if p is not None]
            len_after = len(configs)
            print(f"removed {len_before - len_after} config entries")

        config_ok = self.yes or self._ask_for_yes(f"current parameters yielded {len(configs)} configuration(s). continue?")

        if not config_ok:
            print(f"config not ok. bye.")
            return

        os.makedirs(self.directory, exist_ok=True)
        job_ids = [self.submit_single(job_config, script_config, script_name, self.pre_run_commands) for job_config, script_config in configs]
        return job_ids
