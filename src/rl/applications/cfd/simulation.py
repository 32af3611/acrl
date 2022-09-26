import datetime
import json
import os.path
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from uuid import uuid4

import numpy as np

from src.rl.applications.cfd.config import default_locations
from src.rl.applications.cfd.lib.cfdControl import cfd_blackbox
from src.xutils.logging import get_logger


def run_simulation(
        output_data_directory, output_simulation_directory,
        description, locations,
        coeff_ss, coeff_ps,
        cores, episode, drag_validation_fn, drag_prediction, verbose=True, label=None
):
    t_start = time.time()
    case_id = label or str(uuid4())

    args = dict(
        root_dir=output_simulation_directory,
        case_desc=description,
        blc_loc_ss=list(locations),
        blc_loc_ps=list(locations),
        blc_coef_ss=list(coeff_ss),
        blc_coef_ps=list(coeff_ps),
        cores=cores,
        verbose=verbose
    )

    drag, iterations = None, 0

    try:
        drag, iterations = cfd_blackbox(**args)
    except Exception as e:
        if verbose:
            print(f"simulation failed", str(e))

    t_end = time.time()
    duration = round(t_end - t_start)

    result = case_id, duration, iterations, drag

    if drag is None or iterations == 0:
        return result

    if not drag_validation_fn(drag):
        return result

    info = dict(**args, label=label, episode=episode, drag=float(drag), prediction=drag_prediction, iterations=iterations, duration=duration, date=str(datetime.datetime.now()))
    with open(f"{output_data_directory}/{case_id}.json", "w") as f:
        json.dump(info, f, indent=2)

    return result


class SimulationWrapper:
    def __init__(
            self,
            output_data_directory,
            simulation_tmp_directory,
            drag_validation_fn,
            n_workers,
            n_cores_per_worker,
            use_hyperthreading_cores,
            skip_blocks=0,
            delete=True,
            verbose=False
    ):
        self.logger = get_logger("Simulation")

        self.output_data_directory = output_data_directory
        self.simulation_tmp_directory = simulation_tmp_directory
        self.drag_validation_fn = drag_validation_fn
        self.n_workers = n_workers
        self.n_cores_per_worker = n_cores_per_worker
        self.use_hyperthreading_cores = use_hyperthreading_cores
        self.skip_blocks = skip_blocks
        self.delete = delete
        self.verbose = verbose

        self.affinities = None
        self.affinity_queue = None
        self.executor = None

        self.setup()

    def get_ht_cores(self):
        cmd = "cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list"
        info = [x.split(",") for x in os.popen(cmd).read().split("\n") if x]
        return {int(x): int(y) for x, y in info}

    def setup(self):
        total_cpus = min(self.n_workers * self.n_cores_per_worker, os.cpu_count() // 2)
        if self.n_workers > total_cpus:
            self.logger.warning(f"setting number of workers from {self.n_workers} to {total_cpus}")
            self.n_workers = total_cpus

        self.affinities = np.array(np.split(np.arange(total_cpus), self.n_workers)).tolist()[self.skip_blocks:]

        if self.use_hyperthreading_cores:
            ht_cores = self.get_ht_cores()
            # noinspection PyTypeChecker
            ht_cores = np.array([[ht_cores[b] for b in a] for a in self.affinities])
            self.affinities = np.concatenate((self.affinities, ht_cores), axis=1).tolist()

        # noinspection PyTypeChecker
        self.affinities = [",".join(list(map(str, a))) for a in self.affinities]

        self.logger.info(json.dumps(self.affinities))

        self.affinity_queue = Queue()
        for a in self.affinities:
            self.affinity_queue.put(a)

        self.executor = ThreadPoolExecutor(max_workers=self.n_workers, thread_name_prefix='worker')

        hyperthreaded = "hyperthreaded" if self.use_hyperthreading_cores else "non-hyperthreaded"
        self.logger.info(f"simulations running on {self.n_workers} workers with {self.n_cores_per_worker} {hyperthreaded} cores per worker")

    def run(self, profile):
        thread = threading.currentThread().getName()

        cores = self.affinity_queue.get()

        episode, coeffs, drag, label = profile["episode"], profile["profile"], profile["drag"], profile.get("label", None)
        n_coeffs_per_side = len(coeffs) // 2
        coeff_ps, coeff_ss = coeffs[:n_coeffs_per_side], coeffs[n_coeffs_per_side:]

        profile_dir = os.path.join(self.simulation_tmp_directory, label or str(uuid.uuid4()))

        _, duration, iterations, drag = run_simulation(
            output_data_directory=self.output_data_directory,
            output_simulation_directory=profile_dir,
            drag_validation_fn=self.drag_validation_fn,
            drag_prediction=drag,
            description="data sampling",
            locations=default_locations(len(coeff_ps)),
            coeff_ss=coeff_ss,
            coeff_ps=coeff_ps,
            cores=cores,
            episode=episode,
            verbose=self.verbose,
            label=label
        )

        self.affinity_queue.put(cores)

        if drag is None:
            self.logger.error(f"{thread} - calculation failed, duration {duration}s")
        else:
            self.logger.info(f"{thread} - profile returned drag {drag} after {duration}s")

        if self.delete:
            shutil.rmtree(profile_dir)

    def queue(self, profile):
        self.executor.submit(self.run, profile)

    def done(self, cancel_futures):
        self.executor.shutdown(wait=True, cancel_futures=cancel_futures)

    def wait(self):
        self.executor.shutdown(wait=True, cancel_futures=False)
        self.executor = ThreadPoolExecutor(max_workers=self.n_workers, thread_name_prefix='worker')
