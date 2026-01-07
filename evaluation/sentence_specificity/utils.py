from typing import Tuple, List, Literal

import numpy as np
from matplotlib import pyplot as plt
from pygam import LinearGAM, s

from granuscore import GranuScore


def score_dataset(
    model: GranuScore,
    dataset: List[dict],
    invert_rating: bool = False,
    pooling: Literal["sum", "mean", "lower_quantile_mean", "min"] = "mean",
    pooling_scope: Literal["document", "sentence"] = "document",
    scope_pooling_method: Literal["sum", "mean", "lower_quantile_mean", "min"] = "mean",
    tail_q: float = 0.1,
    scope_tail_q: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    sentences = [b['sentence'] for b in dataset]
    scores = model.predict(sentences, pooling=pooling, pooling_scope=pooling_scope, scope_pooling_method=scope_pooling_method, encoding_batch_size=128, batch_size=128, show_progress_bar=True, percentile_output=True, percentile_before_pooling=True, tail_q=tail_q, scope_tail_q=scope_tail_q)

    if invert_rating:
        ratings = [-b["rating"] for b in dataset]
    else:
        ratings = [b["rating"] for b in dataset]
    return np.asarray(scores), np.asarray(ratings)


def analyze_length_granularity_gam(
    data,
    ratings,
    granularity_scores,
    grid_size=200,
    n_splines=20,
):
    """
    Primary GAM-based analysis of length and granularity.

    Reports (paper-ready):
    - explained deviance (length-only, length+granularity, delta)
    - p-values for length and granularity smooths
    - granularity turning point (peak of inverted-U)
    - partial dependence curves (for figures / appendix)
    """

    # -----------------------
    # Prepare data
    # -----------------------
    lengths = np.array([len(d["sentence"].split()) for d in data])
    gran = np.array(granularity_scores)

    X_len = lengths.reshape(-1, 1)
    X_both = np.column_stack([lengths, gran])

    # -----------------------
    # Fit GAMs
    # -----------------------
    gam_len = LinearGAM(
        s(0, n_splines=n_splines)
    ).fit(X_len, ratings)

    gam_both = LinearGAM(
        s(0, n_splines=n_splines) +   # length
        s(1, n_splines=n_splines)     # granularity
    ).fit(X_both, ratings)

    # -----------------------
    # Explained deviance
    # -----------------------
    R_len = gam_len.statistics_["pseudo_r2"]["explained_deviance"]
    R_both = gam_both.statistics_["pseudo_r2"]["explained_deviance"]

    # -----------------------
    # p-values (smooth terms)
    # -----------------------
    p_length = gam_both.statistics_["p_values"][0]
    p_granularity = gam_both.statistics_["p_values"][1]

    # -----------------------
    # Partial dependence grids
    # -----------------------

    low_gran, high_gran = np.percentile(gran, [1, 99])
    gran_grid = np.linspace(low_gran, high_gran, grid_size)

    low_len, high_len = np.percentile(lengths, [1, 99])
    length_grid = np.linspace(low_len, high_len, grid_size)

    X_len_grid = np.column_stack([
        length_grid,
        np.full_like(length_grid, gran.mean())
    ])

    X_gran_grid = np.column_stack([
        np.full_like(gran_grid, lengths.mean()),
        gran_grid
    ])

    length_effect = gam_both.partial_dependence(term=0, X=X_len_grid)
    gran_effect = gam_both.partial_dependence(term=1, X=X_gran_grid)

    peak_idx = int(np.argmax(gran_effect))
    granularity_peak = float(gran_grid[peak_idx])
    peak_effect_value = float(gran_effect[peak_idx])

    return {
        "explained_deviance": {
            "length_only": float(R_len),
            "length_plus_granularity": float(R_both),
            "delta": float(R_both - R_len),
        },
        "p_values": {
            "length": float(p_length),
            "granularity": float(p_granularity),
        },
        "granularity_peak": {
            "value": granularity_peak,
            "effect": peak_effect_value,
        },
        "gam_curves": {
            "length": {
                "x": length_grid.tolist(),
                "y": length_effect.tolist(),
            },
            "granularity": {
                "x": gran_grid.tolist(),
                "y": gran_effect.tolist(),
            },
        },
    }


def plot_gam(res, output_file):
    plt.figure()
    plt.plot(res["length"]["x"],
             res["length"]["y"])
    plt.xlabel("Sentence length (tokens)")
    plt.ylabel("GAM effect on rating")
    plt.title("GAM effect of sentence length")
    plt.tight_layout()
    plt.savefig(output_file.format(type="lenght"))
    plt.close()

    # Granularity effect
    plt.figure()
    plt.plot(res["granularity"]["x"],
             res["granularity"]["y"])
    plt.xlabel("Granularity score")
    plt.ylabel("GAM effect on rating")
    plt.title("GAM effect of granularity")
    plt.tight_layout()
    plt.savefig(output_file.format(type="granularity"))
    plt.close()


def plot_multiple_gams(
    gam_curves,
    output_file="gams_granularity.pdf",
    xlabel="Granuscore",
    ylabel="GAM effect on Rating",
    sharey=True,
    fig_height=2.5,
):
    n = len(gam_curves)
    fig_width = 4.5 * n  # adjust to taste

    fig, axs = plt.subplots(1, n, figsize=(fig_width, fig_height), sharey=sharey)

    if n == 1:
        axs = [axs]

    for i, (ax, title, res) in enumerate(zip(axs, gam_curves.keys(), gam_curves.values())):
        x = np.asarray(res["x"])
        y = np.asarray(res["y"])

        ax.plot(x, y, color='#1f78b4')
        ax.set_title(title)

        ax.set_xlabel(xlabel)

        # only left subplot gets y label
        if i == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")
            if sharey:
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_gam_overlay(
    gam_curves,
    output_file="gam_overlay.pdf",
    xlabel="Granuscore",
    ylabel="GAM effect on specificity",
):
    plt.figure(figsize=(6, 4))
    colors = ['#a6611a','#dfc27d','#80cdc1','#018571']
    plt.axhline(0, color="grey", linestyle="--", linewidth=1.5, alpha=0.7)

    for i, (label, res) in enumerate(gam_curves.items()):
        x = np.asarray(res["x"])
        y = np.asarray(res["y"])

        plt.plot(
            x, y,
            linewidth=2,
            alpha=0.85,
            color=colors[i],
            label=label,
        )

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
