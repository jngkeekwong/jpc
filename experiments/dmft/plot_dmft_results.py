import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _to_numpy(arr):
    if isinstance(arr, (list, tuple)):
        return [_to_numpy(x) for x in arr]
    return np.asarray(arr)


def _warn_if_nonfinite(name, arr):
    arr = np.asarray(arr)
    n_bad = np.size(arr) - np.count_nonzero(np.isfinite(arr))
    if n_bad:
        print(
            f"Warning: {name} has {n_bad}/{np.size(arr)} non-finite values; "
            "plots may appear empty."
        )


def plot_dmft_loss(dmft_loss, plots_dir, gamma_0=None):
    """Plot DMFT loss over training iterations."""
    dmft_loss = np.asarray(dmft_loss).flatten()
    _warn_if_nonfinite("dmft_loss", dmft_loss)
    iterations = np.arange(1, len(dmft_loss) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(iterations, dmft_loss, color="black", linewidth=2)
    plt.xlabel("$t$")
    plt.ylabel("DMFT loss")
    if gamma_0 is not None:
        plt.title(f"$\\gamma_0 = {gamma_0}$")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "dmft_loss.png"), bbox_inches="tight")
    plt.close()


def plot_H_kernels(all_H, plots_dir, gamma_0=None):
    """Plot final-time slice of H kernels for each layer."""
    all_H = _to_numpy(all_H)
    n_layers = len(all_H)
    _warn_if_nonfinite("all_H", np.stack([np.asarray(H) for H in all_H]))

    fig, axes = plt.subplots(
        n_layers, 1, figsize=(6, 2 * n_layers), squeeze=False
    )
    for l, H_l in enumerate(all_H):
        ax = axes[l, 0]
        kernel = np.asarray(H_l[-1, :, -1, :])
        ax.imshow(kernel, cmap="coolwarm")
        if l == 0:
            ax.set_ylabel("Theory")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Layer {l}")

    if gamma_0 is not None:
        fig.suptitle(f"$\\gamma_0 = {gamma_0}$", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "all_H_kernels.png"), bbox_inches="tight")
    plt.close(fig)


def plot_G_kernels(all_G, plots_dir, gamma_0=None):
    """Plot G kernels for each layer.

    Linear DMFT stores ``G`` as ``(T, T)``; nonlinear DMFT stores
    ``(T, P, T, P)``, in which case the final-time sample-sample block is shown.
    """
    all_G = _to_numpy(all_G)
    n_layers = len(all_G)
    _warn_if_nonfinite("all_G", np.stack([np.asarray(G) for G in all_G]))

    fig, axes = plt.subplots(
        n_layers, 1, figsize=(6, 2 * n_layers), squeeze=False
    )
    for l, G_l in enumerate(all_G):
        ax = axes[l, 0]
        G_arr = np.asarray(G_l)
        if G_arr.ndim == 4:
            G_arr = G_arr[-1, :, -1, :]
        ax.imshow(G_arr, cmap="coolwarm")
        if l == 0:
            ax.set_ylabel("Theory")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Layer {l}")

    if gamma_0 is not None:
        fig.suptitle(f"$\\gamma_0 = {gamma_0}$", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "all_G_kernels.png"), bbox_inches="tight")
    plt.close(fig)


def plot_dmft_kernels_and_loss(
    all_H,
    all_G,
    dmft_loss,
    plots_dir,
    gamma_0=None,
    n_hidden=None,
):
    """Plot DMFT loss and kernel matrices under the BP plots subdirectory."""
    if n_hidden is not None:
        plots_dir = os.path.join(plots_dir, f"{n_hidden}_n_hidden")
    if gamma_0 is not None:
        plots_dir = os.path.join(plots_dir, f"gamma_{gamma_0}")
    plots_dir = os.path.join(plots_dir, "bp")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dmft_loss(dmft_loss, plots_dir, gamma_0=gamma_0)
    plot_H_kernels(all_H, plots_dir, gamma_0=gamma_0)
    plot_G_kernels(all_G, plots_dir, gamma_0=gamma_0)
    return plots_dir


def _initial_sample_kernel(cov, num_inference_steps, num_training_steps, num_samples):
    """
    Extract the sample-sample block at the first inference step (k=0) and
    last training time from a flattened ((K+1)*T*P, (K+1)*T*P) covariance.

    The PC kernels store the full inference trajectory k=0,...,K, so the
    state dimension per (t, mu) block is K+1, not K.
    """
    K1 = num_inference_steps + 1
    T = num_training_steps
    P = num_samples
    tensor = np.asarray(cov).reshape(K1, T, P, K1, T, P)
    return tensor[0, -1, :, 0, -1, :]


def plot_pc_layer_kernels(
    kernels,
    plots_dir,
    filename,
    num_inference_steps,
    num_training_steps,
    num_samples,
    gamma_0=None,
    ylabel="Theory",
):
    """Plot initial sample-sample kernels for each PC layer."""
    kernels = _to_numpy(kernels)
    n_layers = len(kernels)
    _warn_if_nonfinite(
        filename,
        np.stack([np.asarray(k) for k in kernels]),
    )

    fig, axes = plt.subplots(
        n_layers, 1, figsize=(6, 2 * n_layers), squeeze=False
    )
    for l, cov_l in enumerate(kernels):
        ax = axes[l, 0]
        kernel = _initial_sample_kernel(
            cov_l,
            num_inference_steps=num_inference_steps,
            num_training_steps=num_training_steps,
            num_samples=num_samples,
        )
        ax.imshow(kernel, cmap="coolwarm")
        if l == 0:
            ax.set_ylabel(ylabel)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Layer {l + 1}")

    if gamma_0 is not None:
        fig.suptitle(f"$\\gamma_0 = {gamma_0}$", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, filename), bbox_inches="tight")
    plt.close(fig)


def plot_pc_dmft_kernels_and_loss(
    all_Ch,
    all_Cdelta,
    pc_dmft_loss,
    plots_dir,
    num_inference_steps,
    num_training_steps,
    num_samples,
    gamma_0=None,
    n_hidden=None,
    activity_lr=None,
):
    """Plot PC DMFT loss and initial sample-sample Ch / Cdelta kernels."""
    if n_hidden is not None:
        plots_dir = os.path.join(plots_dir, f"{n_hidden}_n_hidden")
    if gamma_0 is not None:
        plots_dir = os.path.join(plots_dir, f"gamma_{gamma_0}")
    if activity_lr is not None:
        plots_dir = os.path.join(plots_dir, f"activity_lr_{activity_lr}")
    plots_dir = os.path.join(plots_dir, "pc")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dmft_loss(pc_dmft_loss, plots_dir, gamma_0=gamma_0)
    # Rename the generic loss file for clarity.
    generic_loss = os.path.join(plots_dir, "dmft_loss.png")
    pc_loss = os.path.join(plots_dir, "pc_dmft_loss.png")
    if os.path.exists(generic_loss):
        os.replace(generic_loss, pc_loss)

    plot_pc_layer_kernels(
        kernels=all_Ch,
        plots_dir=plots_dir,
        filename="all_Ch_kernels.png",
        num_inference_steps=num_inference_steps,
        num_training_steps=num_training_steps,
        num_samples=num_samples,
        gamma_0=gamma_0,
        ylabel=r"$C^h$",
    )
    plot_pc_layer_kernels(
        kernels=all_Cdelta,
        plots_dir=plots_dir,
        filename="all_Cdelta_kernels.png",
        num_inference_steps=num_inference_steps,
        num_training_steps=num_training_steps,
        num_samples=num_samples,
        gamma_0=gamma_0,
        ylabel=r"$C^\Delta$",
    )
    return plots_dir


def plot_pc_theory_vs_finite_loss(
    pc_dmft_loss,
    finite_df,
    plots_dir,
    gamma_0=None,
    n_hidden=None,
    activity_lr=None,
    n_infer_iters=None,
    update_mode=None,
    skip_theory=False,
    skip_finite=False,
):
    """Overlay the PC DMFT theory loss curve with finite-size empirical losses.

    ``finite_df`` is expected to have columns ``width``, ``t`` and ``loss``,
    as produced by ``test_coord_check.get_coord_data(..., stats=["loss"])``
    run with the same hyperparameters used for ``pc_dmft_loss``. This lets us
    visually check that the finite-size PC networks converge to the DMFT
    (infinite-width) prediction as width grows.

    ``update_mode`` (e.g. ``"infer"`` / ``"theory"``) is optional metadata used
    in the plot title and output filename so multiple finite simulations can be
    compared without overwriting each other.

    If ``skip_theory`` is True (or ``pc_dmft_loss`` is None / all zeros), only
    finite overlays are drawn and the title/filename use ``pc_finite_loss``.
    If ``skip_finite`` is True (or ``finite_df`` is empty), only the DMFT
    theory curve is drawn and the title/filename use ``pc_theory_loss``.
    """
    if n_hidden is not None:
        plots_dir = os.path.join(plots_dir, f"{n_hidden}_n_hidden")
    if gamma_0 is not None:
        plots_dir = os.path.join(plots_dir, f"gamma_{gamma_0}")
    if activity_lr is not None:
        plots_dir = os.path.join(plots_dir, f"activity_lr_{activity_lr}")
    if n_infer_iters is not None:
        plots_dir = os.path.join(plots_dir, f"{n_infer_iters}_n_infer_iters")
    plots_dir = os.path.join(plots_dir, "pc")
    os.makedirs(plots_dir, exist_ok=True)

    # None / all-zeros placeholder: plot finite overlays only (e.g. --skip_theory).
    plot_theory = (not skip_theory) and pc_dmft_loss is not None
    if plot_theory:
        pc_dmft_loss = np.asarray(pc_dmft_loss).flatten()
        if np.allclose(pc_dmft_loss, 0.0):
            plot_theory = False
        else:
            _warn_if_nonfinite("pc_dmft_loss", pc_dmft_loss)

    plot_finite = (
        (not skip_finite)
        and finite_df is not None
        and len(finite_df)
        and "width" in finite_df.columns
    )
    widths = (
        sorted(finite_df["width"].unique()) if plot_finite else []
    )
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(widths) - 1)) for i in range(len(widths))]

    plt.figure(figsize=(8, 6))
    if plot_finite:
        for width, color in zip(widths, colors):
            sub = finite_df[finite_df["width"] == width].sort_values("t")
            plt.plot(
                sub["t"],
                sub["loss"],
                marker="o",
                color=color,
                alpha=0.8,
                label=f"width={width}",
            )
    if plot_theory:
        theory_t = np.arange(1, len(pc_dmft_loss) + 1)
        plt.plot(
            theory_t,
            pc_dmft_loss,
            color="black",
            linewidth=2.5,
            linestyle="--",
            label="DMFT theory",
        )
    plt.xlabel("$t$")
    plt.ylabel("PC training loss (MSE)")
    if skip_finite or (plot_theory and not plot_finite):
        title = "PC DMFT theory"
        filename = "pc_theory_loss"
    elif skip_theory or not plot_theory:
        title = "PC finite-size simulation"
        filename = "pc_finite_loss"
    else:
        title = "PC theory vs finite-size simulation"
        filename = "pc_theory_vs_finite_loss"
    if update_mode is not None:
        title += f" ({update_mode})"
        filename += f"_{update_mode}"
    if gamma_0 is not None:
        title += f", $\\gamma_0={gamma_0}$"
    if activity_lr is not None:
        title += f", activity lr$={activity_lr}$"
    if n_infer_iters is not None:
        title += f", $K={n_infer_iters}$"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    save_path = os.path.join(plots_dir, f"{filename}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"PC loss plot saved to {save_path}")
    return save_path


_SWEEP_META_COLS = (
    "n_hidden",
    "gamma_0",
    "activity_lr",
    "n_infer_iters",
    "param_type",
    "use_skips",
)

_SWEEP_VALUE_LABEL = {
    "n_hidden": lambda v: f"$H={int(v)}$",
    "gamma_0": lambda v: f"$\\gamma_0={v}$",
    "n_infer_iters": lambda v: f"$K={int(v)}$",
}

_SWEEP_AXIS_TITLE = {
    "n_hidden": r"$H$",
    "gamma_0": r"$\gamma_0$",
    "n_infer_iters": r"$K$",
}

_SWEEP_FILENAME = {
    "n_hidden": "pc_loss_vs_n_hidden",
    "gamma_0": "pc_loss_vs_gamma_0",
    "n_infer_iters": "pc_loss_vs_n_infer_iters",
}


def _pc_loss_plots_dir(
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
):
    if n_hidden is not None:
        plots_dir = os.path.join(plots_dir, f"{n_hidden}_n_hidden")
    if gamma_0 is not None:
        plots_dir = os.path.join(plots_dir, f"gamma_{gamma_0}")
    if activity_lr is not None:
        plots_dir = os.path.join(plots_dir, f"activity_lr_{activity_lr}")
    if n_infer_iters is not None:
        plots_dir = os.path.join(plots_dir, f"{n_infer_iters}_n_infer_iters")
    plots_dir = os.path.join(plots_dir, "pc")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def _mask_equal(df, col, value):
    """Boolean mask comparing ``df[col]`` to ``value``, with NA-safe equality."""
    series = df[col]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return series.isna()
    return series == value


def plot_pc_param_sweep_loss(
    theory_df,
    finite_df,
    plots_dir,
    swept_col,
    skip_theory=False,
    skip_finite=False,
    plot_closed_form=False,
):
    """Overlay theory and finite-size losses for every value of ``swept_col``.

    Finite curves use only the largest recorded width. Theory is dashed; finite
    infer is solid. If ``plot_closed_form`` is True, closed-form finite updates
    are added: one curve per swept value, except for ``n_infer_iters`` where
    closed-form is independent of ``K`` so a single extra curve is drawn.

    For ``n_infer_iters``, the DMFT theory curve is drawn only for the
    smallest ``K`` (larger ``K`` are finite overlays only).

    If ``skip_theory`` is True, only finite overlays are drawn. If
    ``skip_finite`` is True, only DMFT theory curves are drawn.
    """
    if swept_col not in _SWEEP_VALUE_LABEL:
        raise ValueError(
            f"swept_col must be one of {list(_SWEEP_VALUE_LABEL)}, got {swept_col!r}"
        )

    group_cols = [c for c in _SWEEP_META_COLS if c != swept_col]
    frames = []
    if (not skip_theory) and theory_df is not None and len(theory_df):
        frames.append(theory_df[group_cols])
    if (not skip_finite) and finite_df is not None and len(finite_df):
        infer_df = finite_df[finite_df["infer_mode"] == "infer"]
        if len(infer_df):
            frames.append(infer_df[group_cols])
        elif len(finite_df):
            frames.append(finite_df[group_cols])
    if not frames:
        return []

    groups = pd.concat(frames, ignore_index=True).drop_duplicates()
    save_paths = []
    value_label = _SWEEP_VALUE_LABEL[swept_col]
    axis_title = _SWEEP_AXIS_TITLE[swept_col]
    filename = _SWEEP_FILENAME[swept_col]

    for _, group in groups.iterrows():
        def _in_group(df):
            mask = pd.Series(True, index=df.index)
            for col in group_cols:
                mask &= _mask_equal(df, col, group[col])
            return df.loc[mask]

        g_theory = (
            _in_group(theory_df)
            if theory_df is not None and len(theory_df)
            else theory_df
        )
        g_finite = (
            _in_group(finite_df)
            if (not skip_finite)
            and finite_df is not None
            and len(finite_df)
            else None
        )

        infer_finite = (
            g_finite[g_finite["infer_mode"] == "infer"]
            if g_finite is not None and len(g_finite)
            else g_finite
        )
        closed_finite = (
            g_finite[g_finite["infer_mode"] == "closed_form"]
            if g_finite is not None and len(g_finite)
            else g_finite
        )

        swept_values = []
        for source in (g_theory, infer_finite):
            if source is not None and len(source) and swept_col in source.columns:
                swept_values.extend(source[swept_col].dropna().unique().tolist())
        # Preserve numeric order while keeping first-seen type.
        swept_values = sorted(set(swept_values), key=lambda v: (float(v), str(v)))
        if not swept_values:
            continue

        max_width = None
        if infer_finite is not None and len(infer_finite):
            max_width = int(infer_finite["width"].max())
        elif closed_finite is not None and len(closed_finite):
            max_width = int(closed_finite["width"].max())

        cmap = plt.get_cmap("viridis")
        colors = [
            cmap(i / max(1, len(swept_values) - 1))
            for i in range(len(swept_values))
        ]

        plt.figure(figsize=(8, 6))
        plot_theory = (
            (not skip_theory)
            and g_theory is not None
            and len(g_theory)
        )
        # K-sweep: DMFT is shown only for the smallest inference-step count.
        min_swept = min(swept_values, key=lambda v: float(v))
        for value, color in zip(swept_values, colors):
            label = value_label(value)
            plot_theory_this = plot_theory and (
                swept_col != "n_infer_iters"
                or float(value) == float(min_swept)
            )
            if plot_theory_this:
                sub_th = g_theory[g_theory[swept_col] == value].sort_values("t")
                if len(sub_th):
                    y = np.asarray(sub_th["loss"])
                    if not np.allclose(y, 0.0):
                        _warn_if_nonfinite(f"pc_dmft_loss[{swept_col}={value}]", y)
                        plt.plot(
                            sub_th["t"],
                            y,
                            color=color,
                            linestyle="--",
                            linewidth=2.0,
                            label=f"theory, {label}",
                        )
            if infer_finite is not None and len(infer_finite):
                sub_fi = infer_finite[
                    (infer_finite[swept_col] == value)
                    & (infer_finite["width"] == max_width)
                ].sort_values("t")
                if len(sub_fi):
                    plt.plot(
                        sub_fi["t"],
                        sub_fi["loss"],
                        color=color,
                        linestyle="-",
                        marker="o",
                        markersize=4,
                        alpha=0.85,
                        label=f"finite infer, {label}",
                    )
            if (
                plot_closed_form
                and swept_col != "n_infer_iters"
                and closed_finite is not None
                and len(closed_finite)
            ):
                sub_cf = closed_finite[
                    (closed_finite[swept_col] == value)
                    & (closed_finite["width"] == max_width)
                ].sort_values("t")
                if len(sub_cf):
                    plt.plot(
                        sub_cf["t"],
                        sub_cf["loss"],
                        color=color,
                        linestyle=":",
                        marker="s",
                        markersize=4,
                        alpha=0.85,
                        label=f"finite closed-form, {label}",
                    )

        if (
            plot_closed_form
            and swept_col == "n_infer_iters"
            and closed_finite is not None
            and len(closed_finite)
        ):
            cf_width = (
                int(closed_finite["width"].max())
                if max_width is None
                else max_width
            )
            sub_cf = closed_finite[closed_finite["width"] == cf_width].sort_values(
                "t"
            )
            # Closed-form does not depend on K; keep a single curve.
            sub_cf = sub_cf.drop_duplicates(subset=["t"], keep="first")
            if len(sub_cf):
                plt.plot(
                    sub_cf["t"],
                    sub_cf["loss"],
                    color="black",
                    linestyle="-.",
                    linewidth=2.2,
                    label=f"finite closed-form ($N={cf_width}$)",
                )

        plt.xlabel("$t$")
        plt.ylabel("PC training loss (MSE)")
        if skip_finite or (
            (infer_finite is None or not len(infer_finite))
            and (closed_finite is None or not len(closed_finite))
        ):
            title = f"PC DMFT theory vs {axis_title}"
        elif skip_theory or not plot_theory:
            title = f"PC finite-size vs {axis_title}"
        else:
            title = f"PC theory vs finite-size vs {axis_title}"
        if max_width is not None:
            title += f" ($N={max_width}$)"
        plt.title(title)
        plt.legend(fontsize=8, ncol=1)
        plt.grid(True, alpha=0.4)
        plt.tight_layout()

        out_dir = _pc_loss_plots_dir(
            plots_dir,
            n_hidden=group["n_hidden"] if swept_col != "n_hidden" else None,
            gamma_0=group["gamma_0"] if swept_col != "gamma_0" else None,
            activity_lr=group["activity_lr"],
            n_infer_iters=(
                group["n_infer_iters"] if swept_col != "n_infer_iters" else None
            ),
        )
        save_path = os.path.join(out_dir, f"{filename}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"PC sweep loss plot saved to {save_path}")
        save_paths.append(save_path)

    return save_paths


def plot_bp_theory_vs_finite_loss(
    dmft_loss,
    finite_df,
    plots_dir,
    gamma_0=None,
    n_hidden=None,
    skip_theory=False,
):
    """Overlay the BP DMFT theory loss curve with finite-size empirical losses.

    ``finite_df`` is expected to have columns ``width``, ``t`` and ``loss``,
    as produced by ``train_bpn`` over the same hyperparameters used for
    ``dmft_loss``. This lets us visually check that the finite-size BP
    networks converge to the DMFT (infinite-width) prediction as width grows.

    If ``skip_theory`` is True (or ``dmft_loss`` is None / all zeros), only
    finite overlays are drawn and the title/filename use ``bp_finite_loss``.
    """
    if n_hidden is not None:
        plots_dir = os.path.join(plots_dir, f"{n_hidden}_n_hidden")
    if gamma_0 is not None:
        plots_dir = os.path.join(plots_dir, f"gamma_{gamma_0}")
    plots_dir = os.path.join(plots_dir, "bp")
    os.makedirs(plots_dir, exist_ok=True)

    # None / all-zeros placeholder: plot finite overlays only (e.g. --skip_theory).
    plot_theory = (not skip_theory) and dmft_loss is not None
    if plot_theory:
        dmft_loss = np.asarray(dmft_loss).flatten()
        if np.allclose(dmft_loss, 0.0):
            plot_theory = False
        else:
            _warn_if_nonfinite("dmft_loss", dmft_loss)

    widths = sorted(finite_df["width"].unique())
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(widths) - 1)) for i in range(len(widths))]

    plt.figure(figsize=(8, 6))
    for width, color in zip(widths, colors):
        sub = finite_df[finite_df["width"] == width].sort_values("t")
        plt.plot(
            sub["t"],
            sub["loss"],
            marker="o",
            color=color,
            alpha=0.8,
            label=f"width={width}",
        )
    if plot_theory:
        theory_t = np.arange(1, len(dmft_loss) + 1)
        plt.plot(
            theory_t,
            dmft_loss,
            color="black",
            linewidth=2.5,
            linestyle="--",
            label="DMFT theory",
        )
    plt.xlabel("$t$")
    plt.ylabel("BP training loss (MSE)")
    if skip_theory or not plot_theory:
        title = "BP finite-size simulation"
        filename = "bp_finite_loss"
    else:
        title = "BP theory vs finite-size simulation"
        filename = "bp_theory_vs_finite_loss"
    if gamma_0 is not None:
        title += f", $\\gamma_0={gamma_0}$"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    save_path = os.path.join(plots_dir, f"{filename}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"BP loss plot saved to {save_path}")
    return save_path

def plot_grad_cosine_similarities(
    similarities_by_width,
    plots_dir,
    gamma_0=None,
    n_hidden=None,
    activity_lr=None,
):
    """Plot PC–BP gradient cosine similarity over training time.

    ``similarities_by_width`` maps width -> 1D array of cosine similarities
    indexed by training step. Saved under the plots tree (not under
    ``*_input_dim``), so ``--cleanup_npy`` will not delete it.
    """
    if n_hidden is not None:
        plots_dir = os.path.join(plots_dir, f"{n_hidden}_n_hidden")
    if gamma_0 is not None:
        plots_dir = os.path.join(plots_dir, f"gamma_{gamma_0}")
    if activity_lr is not None:
        plots_dir = os.path.join(plots_dir, f"activity_lr_{activity_lr}")
    os.makedirs(plots_dir, exist_ok=True)

    widths = sorted(similarities_by_width.keys())
    if not widths:
        print("No cosine similarity curves to plot.")
        return None

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(widths) - 1)) for i in range(len(widths))]

    plt.figure(figsize=(8, 6))
    for width, color in zip(widths, colors):
        values = np.asarray(similarities_by_width[width]).flatten()
        _warn_if_nonfinite(f"cos_sim width={width}", values)
        t = np.arange(1, len(values) + 1)
        plt.plot(
            t,
            values,
            marker="o",
            color=color,
            alpha=0.8,
            label=f"width={width}",
        )
    plt.xlabel("$t$")
    plt.ylabel(r"$\cos(\nabla_{\theta}\mathcal{L}_{\mathrm{BP}}, "
               r"\nabla_{\theta}\mathcal{F}^*_{\mathrm{PC}})$")
    title = "PC–BP gradient cosine similarity"
    if gamma_0 is not None:
        title += f", $\\gamma_0={gamma_0}$"
    if activity_lr is not None:
        title += f", activity lr$={activity_lr}$"
    plt.title(title)
    plt.ylim(-1.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "grad_cosine_similarities.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"PC–BP gradient cosine similarity plot saved to {save_path}")
    return save_path


def plot_pc_kernel_width_alignment(
    align_df,
    plots_dir,
    gamma_0=None,
    n_hidden=None,
    activity_lr=None,
    n_infer_iters=None,
):
    """Plot finite-vs-theory kernel alignment vs width for every layer.

    ``align_df`` must have columns ``width``, ``layer``, ``kernel``
    (``"h"`` or ``"delta"``), ``alignment``, and optionally ``seed``.
    Mean ± std over seeds is shown when multiple trials are present.
    Compares last-training-time hidden-layer kernels (``C^h`` at ``k=0``,
    ``C^Δ`` at the last inference step ``k=K``) for all ``H`` hidden layers
    (readout omitted).
    """
    plots_dir = _pc_loss_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
    )
    if align_df is None or len(align_df) == 0:
        print("No kernel-alignment records to plot.")
        return []

    specs = (
        (
            "h",
            r"$A(C^{h,\ell}_{\mathrm{DMFT}}, C^{h,\ell}_{\mathrm{NN}})$",
            "pc_kernel_alignment_Ch_vs_width.png",
        ),
        (
            "delta",
            r"$A(C^{\Delta,\ell}_{\mathrm{DMFT}}, C^{\Delta,\ell}_{\mathrm{NN}})$",
            "pc_kernel_alignment_Cdelta_vs_width.png",
        ),
    )
    saved = []
    for kernel_id, ylabel, filename in specs:
        sub = align_df[align_df["kernel"] == kernel_id]
        if sub.empty:
            continue

        plt.figure(figsize=(8, 5))
        layers = sorted(sub["layer"].unique())
        for layer in layers:
            sl = sub[sub["layer"] == layer]
            widths = sorted(sl["width"].unique())
            means = []
            stds = []
            for width in widths:
                vals = sl.loc[sl["width"] == width, "alignment"].to_numpy(
                    dtype=float
                )
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=0))
            means = np.asarray(means)
            stds = np.asarray(stds)
            _warn_if_nonfinite(
                f"kernel alignment {kernel_id} layer={int(layer)}", means
            )
            plt.errorbar(
                widths,
                means,
                stds,
                marker="o",
                label=rf"$\ell = {int(layer) + 1}$",
            )
        plt.xscale("log")
        plt.xlabel(r"$N$", fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        title = "PC kernel alignment vs width"
        if n_hidden is not None:
            title += f", $H={int(n_hidden)}$"
        if gamma_0 is not None:
            title += f", $\\gamma_0={gamma_0}$"
        if activity_lr is not None:
            title += f", activity lr$={activity_lr}$"
        if n_infer_iters is not None:
            title += f", $K={n_infer_iters}$"
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        save_path = os.path.join(plots_dir, filename)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"PC kernel alignment plot saved to {save_path}")
        saved.append(save_path)
    return saved


def _alignment_plots_dir(
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
):
    if n_hidden is not None:
        plots_dir = os.path.join(plots_dir, f"{n_hidden}_n_hidden")
    if gamma_0 is not None:
        plots_dir = os.path.join(plots_dir, f"gamma_{gamma_0}")
    if activity_lr is not None:
        plots_dir = os.path.join(plots_dir, f"activity_lr_{activity_lr}")
    if n_infer_iters is not None:
        plots_dir = os.path.join(plots_dir, f"{n_infer_iters}_n_infer_iters")
    plots_dir = os.path.join(plots_dir, "alignment")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def plot_kernel_displacement(
    displacement_df,
    plots_dir,
    n_hidden=None,
    activity_lr=None,
):
    """Cosine similarity between the initial (``t=0``) and final (last
    training step) feature kernel at every layer, at ``k=0`` for PC --
    i.e. how much each layer's feature kernel moved during training.

    ``displacement_df`` must have columns ``layer``, ``method`` (``"pc"``
    or ``"bp"``), ``displacement``, ``gamma_0`` and ``n_infer_iters``.

    If a single ``(gamma_0, n_infer_iters)`` combination is present, PC
    and BP are overlaid as two curves. If several combinations are
    present (``gamma_0s`` and/or ``n_infer_iters`` swept), only PC is
    drawn, with one curve per combination (BP does not depend on
    ``n_infer_iters`` and is dropped in this case).
    """
    if displacement_df is None or len(displacement_df) == 0:
        print("No kernel displacement records to plot.")
        return None

    combos = (
        displacement_df[["gamma_0", "n_infer_iters"]]
        .drop_duplicates()
        .sort_values(["gamma_0", "n_infer_iters"])
    )
    overlay = len(combos) > 1

    plt.figure(figsize=(8, 6))
    if not overlay:
        # Per-column access (not row-wise .iloc) to avoid pandas upcasting
        # gamma_0 (float) and n_infer_iters (int) to a common dtype.
        gamma_0 = combos["gamma_0"].iloc[0]
        n_infer_iters = int(combos["n_infer_iters"].iloc[0])

        pc_sub = displacement_df[
            displacement_df["method"] == "pc"
        ].sort_values("layer")
        layers = np.asarray(pc_sub["layer"], dtype=float) + 1
        values = np.asarray(pc_sub["displacement"], dtype=float)
        _warn_if_nonfinite("pc displacement", values)
        plt.plot(layers, values, marker="o", color="tab:blue", label="PC")

        bp_sub = displacement_df[
            displacement_df["method"] == "bp"
        ].sort_values("layer")
        if len(bp_sub):
            bp_layers = np.asarray(bp_sub["layer"], dtype=float) + 1
            bp_values = np.asarray(bp_sub["displacement"], dtype=float)
            _warn_if_nonfinite("bp displacement", bp_values)
            plt.plot(
                bp_layers, bp_values,
                marker="s", color="tab:orange", label="Backprop",
            )

        out_dir = _alignment_plots_dir(
            plots_dir,
            n_hidden=n_hidden,
            gamma_0=gamma_0,
            activity_lr=activity_lr,
            n_infer_iters=n_infer_iters,
        )
    else:
        cmap = plt.get_cmap("viridis")
        colors = [
            cmap(i / max(1, len(combos) - 1)) for i in range(len(combos))
        ]
        pc_df = displacement_df[displacement_df["method"] == "pc"]
        # itertuples (not iterrows) keeps each field's own dtype, since
        # iterrows would coerce a mixed float/int row to a common dtype.
        for row, color in zip(combos.itertuples(index=False), colors):
            sub = pc_df[
                (pc_df["gamma_0"] == row.gamma_0)
                & (pc_df["n_infer_iters"] == row.n_infer_iters)
            ].sort_values("layer")
            if not len(sub):
                continue
            layers = np.asarray(sub["layer"], dtype=float) + 1
            values = np.asarray(sub["displacement"], dtype=float)
            _warn_if_nonfinite(
                f"pc displacement (gamma_0={row.gamma_0}, "
                f"K={row.n_infer_iters})",
                values,
            )
            label = (
                rf"PC, $\gamma_0={row.gamma_0}$, "
                rf"$K={int(row.n_infer_iters)}$"
            )
            plt.plot(layers, values, marker="o", color=color, label=label)

        out_dir = _alignment_plots_dir(
            plots_dir, n_hidden=n_hidden, activity_lr=activity_lr
        )

    plt.xlabel(r"layer $\ell$")
    plt.ylabel(r"$\cos(C^{\cdot,\ell}_{t=0}, C^{\cdot,\ell}_{t=T})$")
    title = "Feature-kernel displacement across training"
    if n_hidden is not None:
        title += f", $H={int(n_hidden)}$"
    if activity_lr is not None:
        title += f", activity lr$={activity_lr}$"
    plt.title(title)
    plt.ylim(-1.05, 1.05)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    save_path = os.path.join(out_dir, "kernel_displacement_vs_layer.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Kernel displacement plot saved to {save_path}")
    return save_path


def plot_pc_k_sweep_displacement(
    displacement_df,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    width=None,
):
    """Overlay per-layer feature-kernel displacement for a ``K`` sweep.

    ``displacement_df`` must have columns ``layer``, ``displacement``,
    ``kind`` (``"dmft"``, ``"infer"``, or ``"closed_form"``), and
    ``n_infer_iters``. One curve per series: DMFT (smallest ``K``,
    dashed), finite-size infer (solid, increasing ``K``), and
    closed-form (linear case, dash-dot).
    """
    if displacement_df is None or len(displacement_df) == 0:
        print("No K-sweep kernel displacement records to plot.")
        return None

    plt.figure(figsize=(8, 6))
    infer_ks = sorted(
        {
            int(k)
            for k in displacement_df.loc[
                displacement_df["kind"] == "infer", "n_infer_iters"
            ].dropna().unique()
        }
    )
    cmap = plt.get_cmap("viridis")
    k_colors = {
        k: cmap(i / max(1, len(infer_ks) - 1))
        for i, k in enumerate(infer_ks)
    }

    dmft_sub = displacement_df[displacement_df["kind"] == "dmft"]
    if len(dmft_sub):
        dmft_k = int(dmft_sub["n_infer_iters"].iloc[0])
        sub = dmft_sub.sort_values("layer")
        layers = np.asarray(sub["layer"], dtype=float) + 1
        values = np.asarray(sub["displacement"], dtype=float)
        _warn_if_nonfinite(f"dmft displacement (K={dmft_k})", values)
        plt.plot(
            layers,
            values,
            color=k_colors.get(dmft_k, "black"),
            linestyle="--",
            linewidth=2.0,
            marker="o",
            label=rf"theory, $K={dmft_k}$",
        )

    for k in infer_ks:
        sub = displacement_df[
            (displacement_df["kind"] == "infer")
            & (displacement_df["n_infer_iters"].astype(int) == int(k))
        ].sort_values("layer")
        if not len(sub):
            continue
        layers = np.asarray(sub["layer"], dtype=float) + 1
        values = np.asarray(sub["displacement"], dtype=float)
        _warn_if_nonfinite(f"infer displacement (K={k})", values)
        plt.plot(
            layers,
            values,
            color=k_colors[k],
            linestyle="-",
            marker="o",
            markersize=4,
            alpha=0.85,
            label=rf"finite infer, $K={k}$",
        )

    cf_sub = displacement_df[displacement_df["kind"] == "closed_form"]
    if len(cf_sub):
        sub = cf_sub.sort_values("layer")
        layers = np.asarray(sub["layer"], dtype=float) + 1
        values = np.asarray(sub["displacement"], dtype=float)
        _warn_if_nonfinite("closed-form displacement", values)
        plt.plot(
            layers,
            values,
            color="black",
            linestyle="-.",
            linewidth=2.2,
            marker="s",
            markersize=4,
            label="finite closed-form",
        )

    plt.xlabel(r"layer $\ell$")
    plt.ylabel(r"$\cos(C^{\cdot,\ell}_{t=0}, C^{\cdot,\ell}_{t=T})$")
    title = "Feature-kernel displacement across training"
    if n_hidden is not None:
        title += f", $H={int(n_hidden)}$"
    if width is not None:
        title += f", $N={int(width)}$"
    if gamma_0 is not None:
        title += f", $\\gamma_0={gamma_0}$"
    if activity_lr is not None:
        title += f", activity lr$={activity_lr}$"
    plt.title(title)
    plt.ylim(-1.05, 1.05)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    out_dir = _alignment_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
    )
    save_path = os.path.join(out_dir, "kernel_displacement_vs_layer.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"K-sweep kernel displacement plot saved to {save_path}")
    return save_path


def plot_pc_bp_kernel_alignment(
    alignment_df,
    plots_dir,
    n_hidden=None,
    activity_lr=None,
):
    """Cosine similarity between the final PC and final BP feature kernels
    at every layer (both at the last training step; PC at ``k=0``).

    ``alignment_df`` must have columns ``layer``, ``alignment``,
    ``gamma_0`` and ``n_infer_iters``. One curve is drawn per
    ``(gamma_0, n_infer_iters)`` combination present in the data.
    """
    if alignment_df is None or len(alignment_df) == 0:
        print("No PC-BP kernel alignment records to plot.")
        return None

    combos = (
        alignment_df[["gamma_0", "n_infer_iters"]]
        .drop_duplicates()
        .sort_values(["gamma_0", "n_infer_iters"])
    )
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(combos) - 1)) for i in range(len(combos))]

    plt.figure(figsize=(8, 6))
    # itertuples (not iterrows) keeps each field's own dtype, since
    # iterrows would coerce a mixed float/int row to a common dtype.
    for row, color in zip(combos.itertuples(index=False), colors):
        sub = alignment_df[
            (alignment_df["gamma_0"] == row.gamma_0)
            & (alignment_df["n_infer_iters"] == row.n_infer_iters)
        ].sort_values("layer")
        if not len(sub):
            continue
        layers = np.asarray(sub["layer"], dtype=float) + 1
        values = np.asarray(sub["alignment"], dtype=float)
        _warn_if_nonfinite(
            f"pc-bp alignment (gamma_0={row.gamma_0}, "
            f"K={row.n_infer_iters})",
            values,
        )
        label = rf"$\gamma_0={row.gamma_0}$, $K={int(row.n_infer_iters)}$"
        plt.plot(layers, values, marker="o", color=color, label=label)

    if len(combos) == 1:
        # Per-column access (not row-wise .iloc) to avoid pandas upcasting
        # gamma_0 (float) and n_infer_iters (int) to a common dtype.
        gamma_0 = combos["gamma_0"].iloc[0]
        n_infer_iters = int(combos["n_infer_iters"].iloc[0])
        out_dir = _alignment_plots_dir(
            plots_dir,
            n_hidden=n_hidden,
            gamma_0=gamma_0,
            activity_lr=activity_lr,
            n_infer_iters=n_infer_iters,
        )
    else:
        out_dir = _alignment_plots_dir(
            plots_dir, n_hidden=n_hidden, activity_lr=activity_lr
        )

    plt.xlabel(r"layer $\ell$")
    plt.ylabel(r"$\cos(C^{h,\ell}_{\mathrm{PC}}, H^{\ell}_{\mathrm{BP}})$")
    title = "PC vs backprop final feature-kernel alignment"
    if n_hidden is not None:
        title += f", $H={int(n_hidden)}$"
    if activity_lr is not None:
        title += f", activity lr$={activity_lr}$"
    plt.title(title)
    plt.ylim(-1.05, 1.05)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    save_path = os.path.join(out_dir, "pc_bp_kernel_alignment_vs_layer.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"PC-BP kernel alignment plot saved to {save_path}")
    return save_path


def plot_final_kernel_grid(
    kernel_rows,
    plots_dir,
    gamma_0=None,
    n_hidden=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    filename="final_kernels_grid.png",
    share_clim=False,
):
    """Grid of final-time sample-sample (``P x P``) feature kernels.

    ``kernel_rows`` is a list of ``(ylabel, kernels)`` pairs. Each
    ``kernels`` is a sequence of ``P x P`` arrays, one per layer (column).
    When ``share_clim`` is True, every panel uses the same colour limits.
    """
    if not kernel_rows:
        raise ValueError("kernel_rows must contain at least one row.")

    rows = []
    n_layers = None
    for label, kernels in kernel_rows:
        kernels = _to_numpy(kernels)
        if n_layers is None:
            n_layers = len(kernels)
        elif len(kernels) != n_layers:
            raise ValueError(
                f"row '{label}' has {len(kernels)} layers but the first "
                f"row has {n_layers}."
            )
        _warn_if_nonfinite(
            f"{label} final kernels",
            np.stack([np.asarray(k) for k in kernels]),
        )
        rows.append((label, kernels))

    clim_kw = {}
    if share_clim:
        stacked = np.stack(
            [np.asarray(k) for _, kernels in rows for k in kernels]
        )
        finite = stacked[np.isfinite(stacked)]
        if finite.size:
            clim_kw = dict(vmin=float(finite.min()), vmax=float(finite.max()))

    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, n_layers, figsize=(2 * n_layers, 2 * n_rows), squeeze=False
    )
    for r, (label, kernels) in enumerate(rows):
        for l, kernel in enumerate(kernels):
            ax = axes[r, l]
            ax.imshow(np.asarray(kernel), cmap="coolwarm", **clim_kw)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(rf"$\ell = {l + 1}$")
            if l == 0:
                ax.set_ylabel(label)

    title = "Final feature kernels"
    if n_hidden is not None:
        title += f", $H={int(n_hidden)}$"
    if width is not None:
        title += f", $N={int(width)}$"
    if gamma_0 is not None:
        title += f", $\\gamma_0={gamma_0}$"
    if activity_lr is not None:
        title += f", activity lr$={activity_lr}$"
    if n_infer_iters is not None:
        title += f", $K={int(n_infer_iters)}$"
    fig.suptitle(title, y=1.02)
    fig.tight_layout()

    out_dir = _alignment_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
    )
    save_path = os.path.join(out_dir, filename)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Final feature kernel grid saved to {save_path}")
    return save_path


def load_and_plot(results_dir, gamma_0, plots_dir=None, n_hidden=None):
    """Load saved DMFT results from results_dir and generate plots."""
    suffix = f"{gamma_0}_gamma_0"
    all_H = np.load(
        os.path.join(results_dir, f"all_H_{suffix}.npy"), allow_pickle=True
    )
    all_G = np.load(
        os.path.join(results_dir, f"all_G_{suffix}.npy"), allow_pickle=True
    )
    dmft_loss = np.load(
        os.path.join(results_dir, f"dmft_loss_{suffix}.npy"), allow_pickle=True
    )

    # object arrays from saving lists
    if all_H.dtype == object:
        all_H = list(all_H)
    if all_G.dtype == object:
        all_G = list(all_G)

    if plots_dir is None:
        plots_dir = os.path.join(results_dir, "plots")

    plot_dmft_kernels_and_loss(
        all_H=all_H,
        all_G=all_G,
        dmft_loss=dmft_loss,
        plots_dir=plots_dir,
        gamma_0=gamma_0,
        n_hidden=n_hidden,
    )
    return plots_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--plots_dir", type=str, default=None)
    parser.add_argument("--gamma_0", type=int, default=1)
    parser.add_argument("--n_hidden", type=int, default=None)
    args = parser.parse_args()

    out_dir = load_and_plot(
        results_dir=args.results_dir,
        gamma_0=args.gamma_0,
        plots_dir=args.plots_dir,
        n_hidden=args.n_hidden,
    )
    print(f"Plots saved to {out_dir}")

# python plot_dmft_results.py --results_dir "results" --plots_dir "plots" --gamma_0 1