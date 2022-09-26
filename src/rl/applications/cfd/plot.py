import json
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import seaborn
import seaborn as sns
import torch
from matplotlib import pyplot
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from src.rl.applications.cfd.config import default_locations
from src.rl.applications.cfd.loading import load_samples_from_directory
from src.rl.applications.cfd.simulation import SimulationWrapper
from src.rl.training.metrics import r2
from src.xutils.io import read_json, read_pickle, write_pickle


def plot_profile(file_name, title, locations, coeff_ss, coeff_ps, show=False, limit=0.01):
    fig, axes = pyplot.subplots(ncols=2, sharey="all", figsize=(12, 6))

    axes[0].plot(locations, np.zeros_like(coeff_ps), color="gray")
    axes[0].plot(locations, [np.mean(coeff_ps)] * len(locations), color="limegreen")
    axes[0].bar(locations, coeff_ps, width=0.0125, color="C0")
    axes[0].set_title("pressure side")
    axes[0].set_xlabel("locations")
    axes[0].set_ylabel("coefficients")

    axes[1].plot(locations, np.zeros_like(coeff_ss), color="gray")
    axes[1].plot(locations, [np.mean(coeff_ss)] * len(locations), color="limegreen")
    axes[1].bar(locations, coeff_ss, width=0.0125, color="C1")
    axes[1].set_title("suction side")
    axes[1].set_xlabel("locations")
    axes[1].set_ylabel("coefficients")

    pyplot.suptitle(title)
    pyplot.tight_layout()
    pyplot.ylim(-limit, limit)

    if show:
        pyplot.show()

    if file_name is not None:
        fig.savefig(file_name)

    pyplot.close("all")


def plot_best_profiles(x, y, n=5, reverse=False):
    sorted_y = np.argsort(y.reshape(-1)).tolist()

    n_coeffs = x.shape[-1] // 2
    locations = default_locations(n_coeffs)
    if reverse:
        sorted_y = list(reversed(sorted_y))  # noqa: type inference is wrong

    for i in sorted_y[:n]:
        coeff_ps, coeff_ss = x[i][:n_coeffs], x[i][n_coeffs:]
        drag = y[i]
        plot_profile(
            file_name=None,
            title=f"profile with drag {float(drag):.6f}",
            locations=locations[:-1],
            coeff_ps=coeff_ps,
            coeff_ss=coeff_ss,
            show=True
        )


def plot_profiles_with_same_drag(directory):
    files = load_samples_from_directory(directory)

    groups = {}
    for f in files:
        drag = round(f["drag"], 5)
        f["profile"] = f["blc_coef_ps"] + f["blc_coef_ss"]
        groups.setdefault(drag, [])
        groups[drag].append(f)

    groups = dict(sorted(groups.items()))
    for drag_group, profiles in groups.items():
        profiles.sort(key=lambda x: x["drag"])
        for profile in profiles[:10]:
            n_coeffs = len(profile["profile"]) // 2
            drag = profile["drag"]
            plot_profile(
                file_name=None,
                title=f"profile in drag group {drag_group:.6f} (exact {drag:.6f})",
                locations=default_locations(n_coeffs)[:-1],
                coeff_ss=profile["blc_coef_ss"],
                coeff_ps=profile["blc_coef_ps"],
                show=True,
                limit=0.01
            )

    mean_profiles = {k: np.array(v).mean(0).tolist() for k, v in groups.items()}
    for drag, profile in mean_profiles.items():
        n_coeffs = len(profile["profile"]) // 2
        plot_profile(
            file_name=None,
            title=f"mean profile of drag {drag:.6f}",
            locations=default_locations(n_coeffs)[:-1],
            coeff_ss=profile["blc_coef_ss"],
            coeff_ps=profile["blc_coef_ps"],
            show=True,
            limit=0.01
        )


def compare_profiles(files, model):
    def predict(ps, ss):
        profile = torch.cat([ps, ss], -1)
        drag = model(profile)
        return profile.tolist(), drag.item()

    def change_profile(profile, indices, value):
        profile = profile.clone()
        old_mean = profile.mean()

        other_indices = list(set(list(range(len(profile)))) - set(indices.tolist()))

        old_idx_mean = profile[indices].mean()
        profile[indices] = value
        new_idx_mean = profile[indices].mean()
        mean_diff = old_idx_mean - new_idx_mean
        scale_factor = len(indices) / len(other_indices)
        profile[other_indices] += mean_diff * scale_factor

        new_mean = profile.mean()
        assert abs(new_mean - old_mean) < 1e-9, "means do not match"
        return profile

    cache_file = os.path.abspath("eval-cache-file.pkl")
    if os.path.exists(cache_file):
        data = read_pickle(cache_file)
    else:
        n = 100000
        np.random.shuffle(files)
        with ThreadPool(os.cpu_count() // 2) as p:
            data = p.map(read_json, files[:n])
        write_pickle(cache_file, data)

    ps, ss, drags, predictions = zip(*[(f["blc_coef_ps"], f["blc_coef_ss"], f["drag"], f["prediction"]) for f in data])
    ps, ss, drags = np.array(ps), np.array(ss), np.array(drags)
    locations = np.array(default_locations(len(ps[0]))[:-1]).round(2)

    idx = drags < np.quantile(drags, 0.001)
    ps_q, ss_q, drag_q = ps[idx], ss[idx], drags[idx]

    ps_mean, ss_mean = ps_q.mean(0), ss_q.mean(0)
    ps_mean, ss_mean = torch.tensor(ps_mean).float(), torch.tensor(ss_mean).float()

    results = {}

    data_eval_dir = os.path.abspath("data-evaluation")

    json_data = [read_json(os.path.join(data_eval_dir, f)) for f in os.listdir(data_eval_dir)]
    df = pd.DataFrame(json_data)
    df["error"] = (df["drag"] - df["prediction"]).abs()
    with open("eval.html", "w") as f:
        f.write(df.sort_values("drag").to_html(columns=["label", "drag", "prediction", "error"]))

    os.makedirs(data_eval_dir, exist_ok=True)
    sim = SimulationWrapper(
        output_data_directory=data_eval_dir,
        simulation_tmp_directory=os.path.abspath("tmp"),
        drag_validation_fn=lambda x: x is not None and x < 0.02,
        n_workers=19,
        n_cores_per_worker=2,
        use_hyperthreading_cores=True,
        verbose=True,
        delete=False
    )

    default_profile, default_drag = predict(ps_mean, ss_mean)
    results["default"] = default_drag
    sim.queue(dict(profile=default_profile, drag=default_drag, label="default"))

    small_indices = np.array([2, 4, 5, 7, 9, 10, 12, 14, 15]) - 1
    ps_1 = change_profile(ps_mean, indices=small_indices, value=0.0)
    zero_small_profile, zero_small_drag = predict(ps_1, ss_mean)
    results["zero_small"] = zero_small_drag
    sim.queue(dict(profile=zero_small_profile, drag=zero_small_drag, label="zero_small"))

    for roll in range(1, len(ps_mean)):
        roll_profile, roll_drag = predict(torch.roll(ps_mean, roll), ss_mean)
        results[f"roll_{roll}"] = roll_drag
        sim.queue(dict(profile=roll_profile, drag=roll_drag, label=f"roll_{roll}"))

    for p in range(25):
        perm = torch.randperm(len(ps_mean))
        perm_profile, perm_drag = predict(ps_mean[perm], ss_mean)
        results[f"perm_{p}"] = perm_drag
        sim.queue(dict(profile=perm_profile, drag=perm_drag, label=f"perm_{p}"))

    poly = np.polynomial.polynomial.Polynomial.fit(locations, ps_mean, 2)

    poly_all_ps = torch.tensor(poly(locations)).float()
    poly_zero_ps = change_profile(poly_all_ps, small_indices, 0.0)

    poly_all_profile, poly_all_drag = predict(poly_all_ps, ss_mean)
    poly_zero_profile, poly_zero_drag = predict(poly_zero_ps, ss_mean)

    results["poly_all"] = poly_all_drag
    results["poly_zero"] = poly_zero_drag

    plot_profile(file_name=None, title=f"drag: {poly_all_drag}", locations=locations.tolist(), coeff_ss=ss_mean.tolist(), coeff_ps=poly_all_ps.tolist(), show=True, limit=0.01)
    plot_profile(file_name=None, title=f"drag: {poly_zero_drag}", locations=locations.tolist(), coeff_ss=ss_mean.tolist(), coeff_ps=poly_zero_ps.tolist(), show=True, limit=0.01)

    sim.queue(dict(profile=poly_all_profile, drag=poly_all_drag, label="poly_all"))
    sim.queue(dict(profile=poly_zero_profile, drag=poly_zero_drag, label="poly_zero"))

    baseline = results["default"]
    result = {f"{k}-default": round(v, 8) - baseline for k, v in results.items()}
    result = dict(sorted(result.items(), key=lambda x: x[1]))
    print(json.dumps(result, indent=2))

    print("waiting for simulations to finish")
    # sim.done(cancel_futures=False)
    print("simulations finished")
    return


def plot_action_matrix(matrix, actions, title, file):
    annot = False
    fig, axes = pyplot.subplots(nrows=1, ncols=3, sharex="none", sharey="all", figsize=(24, 8))
    axes = axes.ravel()

    sns.heatmap((matrix / matrix.sum()).round(2), cmap="Greens", ax=axes[0], annot=annot)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("action")
    axes[0].set_title("actions/global")

    sns.heatmap((matrix / matrix.sum(axis=0)).round(2), cmap="Greens", ax=axes[1], annot=annot)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("action")
    axes[1].set_title("actions/step")

    sns.heatmap(actions, cmap="Greens", ax=axes[2])
    axes[2].set_title("actions")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(file)


def plot_data_recalculations(resampling_directory, file):
    new_files = [read_json(os.path.join(resampling_directory, d)) for d in os.listdir(resampling_directory)]

    if not new_files:
        return

    df = pd.DataFrame(new_files)
    y_true, y_pred = df["drag"], df["prediction"]
    r2_value = round(float(r2(y_pred, y_true)), 6)
    sns.regplot(x=y_true, y=y_pred, line_kws=dict(color="red"), label="Training Data")
    pyplot.title(f"predictions of {len(df)} data points, $R^2$ {r2_value}")
    pyplot.savefig(file)


def plot_data_distribution_scatter(x_samples, y_samples):
    x_projected = PCA(n_components=2).fit_transform(x_samples)
    y_projected = PCA(n_components=2).fit_transform(y_samples)

    n_clusters = min(len(x_samples), 10)
    kmeans_x = MiniBatchKMeans(n_clusters=n_clusters, random_state=0).fit(x_projected)
    kmeans_y = MiniBatchKMeans(n_clusters=n_clusters, random_state=0).fit(y_projected)

    x_centers = kmeans_x.cluster_centers_
    y_centers = kmeans_y.cluster_centers_

    fig, axes = pyplot.subplots(ncols=2, sharex="all", sharey="all", figsize=(12, 6))
    pyplot.setp(axes, xlim=(-0.15, 0.15), ylim=(-0.15, 0.15))

    axes[0].scatter(x_projected[:, 0], x_projected[:, 1], c=kmeans_x.labels_, s=10)
    axes[0].scatter(x_centers[:, 0], x_centers[:, 1], color="red", s=25)
    axes[0].set_title(f"data distribution of {len(x_samples)} samples - gradient-based")

    axes[1].scatter(y_projected[:, 0], y_projected[:, 1], c=kmeans_y.labels_, s=10)
    axes[1].scatter(y_centers[:, 0], y_centers[:, 1], color="red", s=25)
    axes[1].set_title(f"data distribution of {len(y_samples)} samples - prediction-based")

    pyplot.tight_layout()
    pyplot.set_cmap("tab20c")
    pyplot.show()

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return fig, image


def plot_data_distribution_hist(x, y):
    coeffs_per_side = x.shape[-1] // 2
    pressure_side = x[:, :coeffs_per_side]
    suction_side = x[:, coeffs_per_side:]

    n_bins = 250
    fig, axes = pyplot.subplots(nrows=2, ncols=2, sharex="none", sharey="none", figsize=(20, 8))
    axes = axes.ravel()

    axes[0].set_title("pressure side mean")
    axes[0].hist(pressure_side.mean(axis=1), bins=n_bins)

    axes[1].set_title("suction side mean")
    axes[1].hist(suction_side.mean(axis=1), bins=n_bins)

    axes[2].set_title("full range of coefficients (both sides)")
    axes[2].hist(x.reshape(-1), bins=n_bins)

    axes[3].set_title("drag coefficients")
    axes[3].hist(y, bins=n_bins)

    pyplot.tight_layout()
    pyplot.suptitle(f"value distributions of {len(x)} items")
    pyplot.show()

    projected = PCA(n_components=2).fit_transform(x)
    clusters = MiniBatchKMeans(n_clusters=4, random_state=0).fit(projected).labels_
    pyplot.scatter(projected[:, 0], projected[:, 1], c=clusters)
    pyplot.title("data distribution via 2D PCA")
    pyplot.show()


def plot_data_scatter(model, train_dataset, val_dataset, episode, file=None):
    x_train, y_train = train_dataset.dataset.tensors
    y_train = y_train.detach().cpu().numpy().reshape(-1)
    y_pred_train = model(x_train).detach().cpu().numpy().reshape(-1)

    x_val, y_val = val_dataset.dataset.tensors
    y_val = y_val.detach().cpu().numpy().reshape(-1)
    y_pred_val = model(x_val).detach().cpu().numpy().reshape(-1)

    fig, axes = pyplot.subplots(nrows=1, ncols=2, sharex="all", sharey="all", figsize=(20, 8))
    pyplot.figure(figsize=(10, 10))

    seaborn.regplot(x=y_train, y=y_pred_train, line_kws=dict(color="red"), label="Training Data", ax=axes[0], color="C0")
    seaborn.regplot(x=y_val, y=y_pred_val, line_kws=dict(color="red"), label="Validation Data", ax=axes[1], color="C2")

    r2_train = round(float(r2(y_pred_train, y_train)), 6)
    r2_val = round(float(r2(y_pred_val, y_val)), 6)

    axes[0].set_title(f"$R^2$: {r2_train}")
    axes[1].set_title(f"$R^2$: {r2_val}")

    for ax in axes:
        ax.set_xlabel("actual")
        ax.set_ylabel("prediction")
        ax.legend()

    fig.suptitle(f"predictions on episode {episode}")

    if file is not None:
        fig.savefig(file)
        return

    fig.show()


def plot_training(losses, rewards, drags, ps_means, ss_means, mean_q_values, max_q_values, lrs, steps, exploration_end, file, smooth=100, warmup=0, show=False):
    losses_, rewards_, drags_, ps_means_, ss_means_, mean_q_values_, max_q_values_, lrs_, steps_ = [
        pd.Series(x).rolling(smooth).mean()[smooth + warmup:].tolist()
        for x in [losses, rewards, drags, ps_means, ss_means, mean_q_values, max_q_values, lrs, steps]
    ]

    if not losses_:
        return

    x_axis = np.arange(smooth, len(losses_) + smooth) + warmup

    fig, axes = pyplot.subplots(nrows=4, ncols=2, sharex="none", sharey="none", figsize=(20, 14))
    axes = axes.ravel()
    axes[0].plot(x_axis, losses_, color="C0", label="Loss")

    axes[2].plot(x_axis, max_q_values_, color="C1", label="Max Q-Values")
    axes[2].plot(x_axis, mean_q_values_, color="C2", label="Mean Q-Values")

    axes[4].plot(x_axis, lrs_, color="C3", label="LR")
    axes[6].plot(x_axis, steps_, color="C4", label="Steps")

    axes[1].plot(x_axis, rewards_, color="C3", label="Reward")

    axes[3].plot(x_axis, ps_means_, color="C4", label="PS Mean")
    axes[3].plot(x_axis, ss_means_, color="C5", label="SS Mean")
    axes[3].plot(x_axis, np.zeros_like(ss_means_), color="gray")

    axes[5].plot(x_axis, drags_, color="C6", label="Drag")

    axes[7].plot(x_axis, np.zeros_like(x_axis), color="gray", label="empty")
    axes[7].set_visible(False)

    for ax in axes:
        ax.legend()
        ax.set_xlabel("Episode")

        if len(losses) > exploration_end:
            ax.axvline(x=exploration_end, color="gray")

    fig.suptitle(f"running means smoothed over {smooth} episodes ({warmup} warmup steps omitted)")

    if show:
        pyplot.show()

    pyplot.savefig(file)


def plot_actions_gif(directory, file):
    action_files = [os.path.join(directory, d) for d in os.listdir(directory)]
    action_files = [(f, int(Path(f).stem)) for f in action_files]
    action_files.sort(key=lambda x: x[1])
    action_images = [pyplot.imread(f) for f, episode in action_files if episode % 500 == 0]
    imageio.mimsave(file, action_images, fps=1.5)


def plot_results():
    df_retrain = pd.read_pickle("outputs/cfd/retrain-0-079ef6b6-aeec-40c1-a1ed-43395b0cabba/result.pkl")
    df_no_retrain = pd.read_pickle("outputs/cfd/no-retrain-0-a7ae54e4-8744-49d8-a9c5-495c7f6b6c66/result.pkl")
    min_epochs = min(len(df_retrain), len(df_no_retrain))
    rolling = 500
    pyplot.plot(df_retrain.rolling(rolling).mean()["reward"][rolling:min_epochs], label="retraining")
    pyplot.plot(df_no_retrain.rolling(rolling).mean()["reward"][rolling:min_epochs], label="no retraining")
    pyplot.legend()
    pyplot.xlabel("episode")
    pyplot.ylabel("reward")
    pyplot.tight_layout()
    pyplot.show()
