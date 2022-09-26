import glob
import os.path
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
from matplotlib import pyplot, pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from src.rl.applications.cfd.config import default_locations
from src.xutils.io import read_json


def bin_averages(data, bin_width, field):
    series = data[field]
    bins = np.array_split(series, len(series) // bin_width)
    y_axis = np.array([x.mean() for x in bins])
    x_axis = np.cumsum([len(x) for x in bins]) - len(bins[0]) // 2

    return x_axis, y_axis


def plot(filename, dataframes, field, xlabel, bins=None, loc=None, ncol=None, anchor=None, font_size=None, window=100, plot_zero=True):
    font = dict(size=font_size)
    pyplot.rc("font", **font)
    pyplot.figure(figsize=figsize)

    for df, label, color in dataframes:
        series = df if field is None else df[field]
        rolling = series.rolling(window)

        y_axis = rolling.mean()
        x_axis = np.arange(len(y_axis))
        fill = rolling.std()[window:]

        if plot_zero:
            plt.plot(x_axis, np.zeros_like(x_axis), color="grey", linewidth=3)
        plt.plot(x_axis, y_axis, label=label, color=color, linewidth=5)
        plt.fill_between(
            x_axis,
            y_axis - fill,
            y_axis + fill,
            alpha=0.1,
            color=color
        )

        pyplot.xlabel("Episode", fontdict=font)
        pyplot.ylabel(xlabel, fontdict=font)

    if bins is not None:
        for x, y, label, color in bins:
            pyplot.scatter(x, y, color=color, marker="D", s=500, label=label)

    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.5f}'))
    pyplot.xlim((0.0, None))
    pyplot.legend(loc=loc, ncol=ncol, fontsize=0.8 * font_size, bbox_to_anchor=anchor)
    pyplot.grid()
    pyplot.tight_layout()
    if filename is not None:
        pyplot.savefig(filename)
    pyplot.show()


def plot_profile_distribution(filename, profiles, font_size=30):
    font = dict(size=font_size)

    ps, ss, drags, predictions = zip(*[(f["blc_coef_ps"], f["blc_coef_ss"], f["drag"], f["prediction"]) for f in profiles])
    ps, ss, drags = np.array(ps), np.array(ss), np.array(drags)
    locations = np.array(default_locations(len(ps[0]))[:-1]).round(2)
    x_n = locations.shape[-1] + 1
    for q in [0.01]:
        pyplot.rc("font", **font)
        pyplot.figure(figsize=figsize)

        idx = drags < np.quantile(drags, q)
        ps_q, ss_q, drag_q = ps[idx], ss[idx], drags[idx]

        pyplot.boxplot(ps_q, showfliers=False)
        pyplot.boxplot(ss_q, showfliers=False)
        pyplot.plot(np.zeros((x_n,)), color="gray")
        pyplot.xlabel(f"Coefficient number", fontdict=font)
        pyplot.ylabel(f"Coefficient value\n(ps < 0, ss > 0)", fontdict=font)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}'))
        pyplot.xlim((0.5, None))

        pyplot.tight_layout()
        pyplot.savefig(filename)
        pyplot.show()


target_dir = os.path.abspath("figures")
source_dir = os.path.abspath("plots")

figsize = (22, 14)
acrl = "ACRL"
static = "Static\nreward model"
oracle = "Oracle"
full = "Fully updated\nreward model"

os.makedirs(target_dir, exist_ok=True)


def plot_logp():
    logp_files = [
        (os.path.join(source_dir, "logp", "logp_triple_approx_st_dev_400.csv"), acrl, "C0"),
        (os.path.join(source_dir, "logp", "logp_approx_no_update.csv"), static, "C1"),
        (os.path.join(source_dir, "logp", "logp_approx_update.csv"), full, "C2"),
        (os.path.join(source_dir, "logp", "logp_real.csv"), oracle, "C3")
    ]

    logp = [(pd.read_csv(f), l, c) for f, l, c in logp_files]
    plot(
        filename=os.path.join(target_dir, "main_logp"), dataframes=logp, field="reward",
        xlabel="logP", loc="lower right", ncol=1, font_size=45
    )


def plot_qed():
    qed_files = [
        (os.path.join(source_dir, "qed", "qed_triple_approx_st_dev_400.csv"), acrl, "C0"),
        (os.path.join(source_dir, "qed", "qed_approx_no_update.csv"), static, "C1"),
        (os.path.join(source_dir, "qed", "qed_approx_update.csv"), full, "C2"),
        (os.path.join(source_dir, "qed", "qed_real.csv"), oracle, "C3")
    ]

    qed = [(pd.read_csv(f), l, c) for f, l, c in qed_files]
    plot(
        filename=os.path.join(target_dir, "main_qed"), dataframes=qed, field="reward",
        xlabel="QED", loc="center right", ncol=1, font_size=45
    )


def plot_opt():
    opt_model_files = [
        (os.path.join(source_dir, "mol_opt", "mol_opt.csv"), acrl, "C0"),
        (os.path.join(source_dir, "mol_opt", "mol_opt_no_retrain.csv"), static, "C1")
    ]

    opt_real_files = [
        (os.path.join(source_dir, "mol_opt", "full_reward_retrain.csv"), "Oracle with updates", "C0"),
        (os.path.join(source_dir, "mol_opt", "full_reward_no_retrain.csv"), "Oracle without updates", "C1")
    ]

    opt_model = [(pd.read_csv(f), l, c) for f, l, c in opt_model_files]
    opt_real = [(pd.read_csv(f), l, c) for f, l, c in opt_real_files]
    opt_bins = [(*bin_averages(data, bin_width=250, field="reward"), label, color) for data, label, color in opt_real]

    plot(
        filename=os.path.join(target_dir, "main_mol_opt"), dataframes=opt_model, field="reward",
        xlabel="Reward", bins=opt_bins, loc="lower right", ncol=1, font_size=45
    )


def plot_fig3():
    tasks = ["logp", "qed"]
    xlabels = ["logP", "QED"]
    locs = ["lower right", "center right"]
    anchors = [None, (1.0, 0.4)]
    fig_files = [[
        (os.path.join(source_dir, f"fig3_{task}", f"{task}_approx_no_update.csv"), static, "C1"),
        (os.path.join(source_dir, f"fig3_{task}", f"{task}_approx_update.csv"), full, "C2"),
        (os.path.join(source_dir, f"fig3_{task}", f"{task}_real.csv"), oracle, "C3"),
        (os.path.join(source_dir, f"fig3_{task}", f"{task}_800_stdev.csv"), f"{acrl} (std)", "C0"),
        (os.path.join(source_dir, f"fig3_{task}", f"{task}_800_random.csv"), f"{acrl} (random)", "C4"),
        (os.path.join(source_dir, f"fig3_{task}", f"{task}_800_bin.csv"), f"{acrl} (bins)", "C5"),

    ] for task in tasks]

    for fig, xlabel, loc, anchor in zip(fig_files, xlabels, locs, anchors):
        dataframes = [(pd.read_csv(f), l, c) for f, l, c in fig]
        plot(
            filename=os.path.join(target_dir, f"si_fig3_{xlabel}"), dataframes=dataframes, field="reward", bins=None,
            xlabel=xlabel, window=100, font_size=45, plot_zero=False, loc=loc, ncol=1, anchor=anchor
        )


def plot_fig4():
    tasks = ["logp", "qed"]
    xlabels = ["logP", "QED"]
    anchors = [None, (1.0, 0.2)]

    fig_files = [[
        (os.path.join(source_dir, f"fig4_{task}", f"{task}_approx_no_update.csv"), static, "C1"),
        (os.path.join(source_dir, f"fig4_{task}", f"{task}_approx_update.csv"), full, "C2"),
        (os.path.join(source_dir, f"fig4_{task}", f"{task}_real.csv"), oracle, "C3"),
        (os.path.join(source_dir, f"fig4_{task}", f"{task}_stdev_400.csv"), f"{acrl} (400)", "C0"),
        (os.path.join(source_dir, f"fig4_{task}", f"{task}_stdev_100.csv"), f"{acrl} (100)", "C4"),
        (os.path.join(source_dir, f"fig4_{task}", f"{task}_stdev_200.csv"), f"{acrl} (200)", "C5"),
        (os.path.join(source_dir, f"fig4_{task}", f"{task}_stdev_600.csv"), f"{acrl} (600)", "C6"),
        (os.path.join(source_dir, f"fig4_{task}", f"{task}_stdev_800.csv"), f"{acrl} (800)", "C7"),
        (os.path.join(source_dir, f"fig4_{task}", f"{task}_stdev_1600.csv"), f"{acrl} (1600)", "C8"),
        (os.path.join(source_dir, f"fig4_{task}", f"{task}_stdev_2000.csv"), f"{acrl} (2000)", "C9"),

    ] for task in tasks]

    for fig, xlabel, anchor in zip(fig_files, xlabels, anchors):
        dataframes = [(pd.read_csv(f), l, c) for f, l, c in fig]
        plot(
            filename=os.path.join(target_dir, f"si_fig4_{xlabel}"), dataframes=dataframes, field="reward", bins=None,
            xlabel=xlabel, window=100, font_size=45, plot_zero=False, loc="lower right", ncol=2, anchor=anchor
        )


def plot_fig5():
    fn = pd.read_csv(os.path.join(source_dir, "fig5_qed", "function.csv"))
    fn_dataframes = [
        (fn["1"], "$\\epsilon=0.01$", "C4"),
        (fn["3"], "$\\epsilon=0.03$", "C5"),
        (fn["10"], "$\\epsilon=0.1$", "C6"),
        (fn["20"], "$\\epsilon=0.2$", "C7")
    ]

    plot(
        filename=os.path.join(target_dir, "si_fig5_fn"), dataframes=fn_dataframes, field=None, bins=None,
        xlabel="$\\epsilon$", window=100, font_size=45, plot_zero=False, loc="upper right", ncol=1
    )

    fig5_files = [
        (os.path.join(source_dir, "fig5_qed", "qed_approx_no_update.csv"), static, "C1"),
        (os.path.join(source_dir, "fig5_qed", "qed_approx_update.csv"), full, "C2"),
        (os.path.join(source_dir, "fig5_qed", "qed_real.csv"), oracle, "C3"),
        (os.path.join(source_dir, "fig5_qed", "qed_4800_1.csv"), "$\\epsilon=0.01$", "C4"),
        (os.path.join(source_dir, "fig5_qed", "qed_4800_3.csv"), "$\\epsilon=0.03$", "C5"),
        (os.path.join(source_dir, "fig5_qed", "qed_4800_10.csv"), "$\\epsilon=0.1$", "C6"),
        (os.path.join(source_dir, "fig5_qed", "qed_4800_20.csv"), "$\\epsilon=0.2$", "C7")
    ]

    dataframes = [(pd.read_csv(f), l, c) for f, l, c in fig5_files]
    plot(
        filename=os.path.join(target_dir, "si_fig5_qed"), dataframes=dataframes, field="reward", bins=None,
        xlabel="QED", window=100, font_size=45, plot_zero=False, loc="best", ncol=2, anchor=(0.45, 0.4)
    )


def plot_fig6():
    fn = pd.read_csv(os.path.join(source_dir, "fig6_qed", "function.csv"))
    fn_dataframes = [
        (fn["full_exponential"], "Full exponential", "C4"),
        (fn["linear_20"], "$\\lambda=0.2$", "C5"),
        (fn["linear_50"], "$\\lambda=0.5$", "C6"),
        (fn["linear_70"], "$\\lambda=0.7$", "C7"),
        (fn["full_linear"], "Full linear", "C8")
    ]

    plot(
        filename=os.path.join(target_dir, "si_fig6_fn"), dataframes=fn_dataframes, field=None, bins=None,
        xlabel="$\\lambda$", window=100, font_size=45, plot_zero=False, loc="upper right", ncol=1
    )

    fig5_files = [
        (os.path.join(source_dir, "fig6_qed", "qed_approx_no_update.csv"), static, "C1"),
        (os.path.join(source_dir, "fig6_qed", "qed_approx_update.csv"), full, "C2"),
        (os.path.join(source_dir, "fig6_qed", "qed_real.csv"), oracle, "C3"),
        (os.path.join(source_dir, "fig6_qed", "qed_lambda_0.0.csv"), "Full exponential", "C4"),
        (os.path.join(source_dir, "fig6_qed", "qed_lambda_0.2.csv"), "$\\lambda=0.2$", "C5"),
        (os.path.join(source_dir, "fig6_qed", "qed_lambda_0.5.csv"), "$\\lambda=0.5$", "C6"),
        (os.path.join(source_dir, "fig6_qed", "qed_lambda_0.7.csv"), "$\\lambda=0.7$", "C7"),
        (os.path.join(source_dir, "fig6_qed", "qed_lambda_1.0.csv"), "Full linear", "C8")

    ]

    dataframes = [(pd.read_csv(f), l, c) for f, l, c in fig5_files]
    plot(
        filename=os.path.join(target_dir, "si_fig6_qed"), dataframes=dataframes, field="reward", bins=None,
        xlabel="QED", window=100, font_size=45, plot_zero=False, loc="center right", ncol=2, anchor=(1.0, 0.375)
    )


def plot_cfd():
    prefixes = [
        "training-small-range",
        "training-large-range"
    ]

    for prefix in prefixes:
        cfd_files = [
            (glob.glob(f"outputs/cfd/{prefix}-retrain-*/result.pkl")[0], acrl, "C0"),
            (glob.glob(f"outputs/cfd/{prefix}-no-retrain-*/result.pkl")[0], static, "C1")
        ]
        cfd = [(pd.read_pickle(f), l, c) for f, l, c in cfd_files]
        shortest = min([len(df) for df, _, _ in cfd])
        cfd = [(df[:shortest], l, c) for df, l, c in cfd]
        cfd_bins = [(*bin_averages(data, bin_width=20000, field="drag"), label, color) for i, (data, label, color) in enumerate(cfd)]

        plot(
            filename=os.path.join(target_dir, f"main_cfd_{prefix}_training"), dataframes=cfd, field="drag", bins=cfd_bins,
            xlabel="Drag", window=1000, font_size=45, plot_zero=False, ncol=2, loc="lower left"
        )

        plot(
            filename=os.path.join(target_dir, f"main_cfd_{prefix}_reward"), dataframes=cfd, field="reward", bins=None,
            xlabel="Reward", window=1000, font_size=45, plot_zero=False, ncol=2, loc="lower right"
        )


def plot_profiles():
    prefixes = [
        "training-small-range",
        "training-large-range"
    ]

    for prefix in prefixes:
        with ThreadPool(os.cpu_count() // 2) as p:
            profiles_retrain = p.map(read_json, glob.glob(f"resampling/{prefix}-retrain-*/*.json"))
            profiles_no_retrain = p.map(read_json, glob.glob(f"resampling/{prefix}-no-retrain-*/*.json"))
        profiles_retrain.sort(key=lambda x: x["episode"])
        profiles_no_retrain.sort(key=lambda x: x["episode"])

        profiles_retrain = pd.DataFrame(profiles_retrain)
        plot_profile_distribution(filename=os.path.join(target_dir, f"main_cfd_{prefix}_profiles"), profiles=list(profiles_retrain.T.to_dict().values()), font_size=45)


plot_logp()
plot_qed()
plot_opt()

plot_fig3()
plot_fig4()
plot_fig5()
plot_fig6()

plot_cfd()
plot_profiles()
