"""Finite-size PC vs backprop feature-kernel alignment analysis.

Trains one finite-width PC network and one finite-width BP network on the
same data (for a single, given set of hyperparameters — no DMFT theory,
no sweeps), then compares their hidden-layer feature kernels across
training. Linear nets use ``C^{h,\\ell}`` (``phi`` is the identity);
nonlinear nets use ``C^{\\phi,\\ell} = \\phi(h^\\ell)\\phi(h^\\ell)^T / N``.
PC inference is selected via ``--pc_infer_mode``: ``infer`` runs iterative
activity optimisation (``--n_infer_iters`` gradient steps per training
step); ``closed_form`` solves the linear equilibrium activities directly
(requires ``--act_fn linear``).

For ``--n_timepoints`` equally spaced training steps ``t`` (including
``t=0``, the untrained feedforward network, and ``t=T-1``, the last
training step), this produces feature-kernel heatmap grids. Displacement
and alignment line plots use a denser time grid: every training step if
``T <= 31``, otherwise every 2 steps (always including the endpoints).

- PC vs BP training-loss curves
  (``alignment/pc_bp_loss.png``).
- Feature-kernel grid: one figure per heatmap timepoint, a ``P x P``
  heatmap per hidden layer, top row PC / bottom row backprop, kernels
  converted to correlations ``R_{ij} = C_{ij} / sqrt(C_{ii} C_{jj})``
  (``alignment/feature_kernels_grid_t{t}.png``).
- Kernel displacement vs time: one subplot per layer, cosine similarity
  of each layer's feature kernel at ``t`` vs. at ``t=0``, PC and BP
  curves (``alignment/kernel_displacement_vs_time.png``), plus a matching
  figure of relative Frobenius displacement
  ``||C_t - C_0||_F / ||C_0||_F``
  (``alignment/kernel_displacement_rel_vs_time.png``).
- Kernel-target alignment vs time: same per-layer layout, linear CKA of
  each PC / BP feature kernel with the target kernel ``C^y = Y Y^T``
  (``alignment/kernel_target_alignment_vs_time.png``), plus CKA of the
  kernel *change* ``C(t) - C(0)`` with ``C^y``, omitting ``t=0``
  (``alignment/kernel_target_change_alignment_vs_time.png``).
- Kernel-input alignment vs time: CKA with the input kernel
  ``C^x = X X^T`` (``alignment/kernel_input_alignment_vs_time.png``).
- Leading-eigenvector overlap with the labels: absolute cosine of the
  leading eigenvector of the *centered* feature kernel with ``y``
  (subspace cosine if ``Y`` has several columns)
  (``alignment/evec_leading_overlap_label_vs_time.png``).
- Feature-kernel effective rank vs time: participation ratio
  ``(sum λ)^2 / sum λ^2`` of the uncentered ``P x P`` kernels, PC and BP
  (``alignment/kernel_effective_rank_vs_time.png``).
- PC-BP alignment vs. time: a single figure, centered kernel alignment
  (CKA) between the PC and backprop feature kernels at each layer,
  x-axis is training time ``t``, one curve per layer
  (``alignment/pc_bp_kernel_alignment_vs_time.png``), plus a matching
  figure of CKA between the kernel *changes* ``C(t) - C(0)``, omitting
  ``t=0`` (``alignment/pc_bp_kernel_change_alignment_vs_time.png``), plus
  top-``k`` eigenspace overlap (mean of ``cos^2 θ_i`` for ``k=1,3,5``),
  one subplot per layer (``alignment/pc_bp_subspace_overlap_vs_time.png``).
- Final-kernel spectrum: uncentered eigenvalues of the last-step
  ``P x P`` kernels, one subplot per layer, log y-scale, PC vs BP
  (``alignment/kernel_spectrum_final.png``).
- Sample-traced temporal kernels: a single figure using the same grid
  scheme as the feature-kernel grid (top row PC / bottom row backprop,
  one column per layer), but with ``T x T`` kernels traced over samples
  and spanning the whole training trajectory at once, displayed as
  correlations (``alignment/temporal_kernels_grid.png``).
- Temporal-kernel spectrum and effective rank: log-y eigenspectrum of
  the ``T x T`` sample-traced kernels with ``R_eff`` annotated on each
  panel (``alignment/temporal_kernel_spectrum.png``), plus
  participation ratio vs layer
  (``alignment/temporal_kernel_effective_rank_vs_layer.png``), plus
  PC–BP CKA of those temporal kernels vs layer
  (``alignment/temporal_pc_bp_kernel_alignment_vs_layer.png``).
- If ``--n_seeds > 1``: kernel concentration across weight-init seeds
  (mean pairwise CKA of PC kernels and of BP kernels), vs layer and vs
  time (``alignment/kernel_concentration_vs_layer_grid.png``,
  ``alignment/kernel_concentration_vs_time.png``). Skipped when
  ``--n_seeds`` is 1 (the default).

``--seed`` draws two independent RNG streams (dataset, weight init). PC
is initialized from the weight-init stream; BP copies those Linear
weights so both networks start from the same parameters. At ``t=0`` the
script asserts that PC–BP kernel CKA is close to 1 (the mismatch
``1 - CKA`` is small). PC and BP always train on the same ``(X, y)``.
Additional seeds (``--n_seeds``) vary only the weight-init stream.

Use the ``PC_dmft_env`` conda environment:
    /data/ndcn-computational-neuroscience/mert5001/envs/PC_dmft_env/bin/python analyse_alignment.py
"""

import os
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from experiments.datasets import get_dataloaders
from experiments.mupc_paper.utils import set_seed
from experiments.limits_paper.utils import setup_bp_experiment
from experiments.dmft.utils import (
    CIFAR_GRAY_DIM,
    MLP,
    centered_kernel_alignment,
    cleanup_experiment_dirs,
    copy_mlp_linear_params,
    cosine_similarity,
    create_tiny_cifar10_dataset,
    create_toy_dataset,
    kernel_eigs,
    kernels_to_correlations,
    leading_evec_label_overlap,
    participation_ratio,
    participation_ratio_from_eigs,
    relative_frobenius_displacement,
    subspace_overlap,
    train_bpn,
)
from theory_pc_nonlin_utils import get_nonlinearity
from analyse_convergence import (
    _train_finite_pc,
    _feature_kernels_from_h,
    _sample_traced_feature_kernels_from_h_traj,
)
from plot_dmft_results import (
    feature_kernel_symbol,
    plot_final_kernel_grid,
    plot_temporal_kernel_grid,
    plot_kernel_displacement_per_timepoint,
    plot_kernel_concentration_per_timepoint,
    plot_kernel_concentration_vs_time,
    plot_kernel_effective_rank_vs_time,
    plot_kernel_input_alignment_vs_time,
    plot_kernel_spectrum,
    plot_kernel_target_alignment_vs_time,
    plot_kernel_target_change_alignment_vs_time,
    plot_leading_evec_label_overlap_vs_time,
    plot_pc_bp_alignment_vs_time,
    plot_pc_bp_loss,
    plot_pc_bp_subspace_overlap_vs_time,
    plot_temporal_kernel_effective_rank_vs_layer,
    plot_temporal_pc_bp_alignment_vs_layer,
)


def _select_timepoints(n_train_iters, n_timepoints):
    """``n_timepoints`` equally spaced training-step indices.

    Always includes both endpoints (``t=0`` and ``t=n_train_iters - 1``).
    If ``n_timepoints >= n_train_iters``, every training step is used.
    Used for feature-kernel heatmap grids.
    """
    n_timepoints = max(2, min(n_timepoints, n_train_iters))
    idx = np.round(
        np.linspace(0, n_train_iters - 1, n_timepoints)
    ).astype(int)
    return np.unique(idx).tolist()


def _select_curve_timepoints(n_train_iters, stride=2, every_t_max=31):
    """Dense time grid for displacement and alignment line plots.

    Every training step if ``T <= every_t_max``, otherwise every
    ``stride`` steps (always including ``t=0`` and ``t=T-1``).
    """
    if n_train_iters <= every_t_max:
        return list(range(n_train_iters))
    idx = list(range(0, n_train_iters, stride))
    last = n_train_iters - 1
    if idx[-1] != last:
        idx.append(last)
    return idx


_INIT_CKA_ATOL = 1e-3
_SUBSPACE_KS = (1, 3, 5)


def _sample_gram(M):
    """Gram over samples: rows of ``M`` are samples, ``C = M M^T``."""
    M = np.asarray(M, dtype=np.float64)
    if M.ndim == 1:
        M = M[:, None]
    return M @ M.T


def _target_kernel(Y_target):
    """Label Gram ``C^y = Y Y^T`` with sample axis first."""
    return _sample_gram(Y_target)


def _input_kernel(X_input):
    """Input Gram ``C^x = X X^T`` with sample axis first."""
    return _sample_gram(X_input)


def _spectrum_records(kernels, method):
    """One row per descending eigenvalue of each layer's kernel."""
    records = []
    for l, C in enumerate(kernels):
        eigs = kernel_eigs(C)
        r_eff = participation_ratio_from_eigs(eigs)
        for i, lam in enumerate(eigs):
            records.append({
                "layer": l,
                "method": method,
                "index": i + 1,
                "eigenvalue": float(lam),
                "effective_rank": r_eff,
            })
    return records


def _assert_init_pc_bp_cka(pc_kernels, bp_kernels, *, seed, atol=_INIT_CKA_ATOL):
    """Assert that PC and BP feature kernels match at initialisation.

    After copying PC weights onto BP, ``1 - CKA`` should be small at
    ``t=0`` for every hidden layer.
    """
    if len(pc_kernels) != len(bp_kernels):
        raise AssertionError(
            "PC/BP init kernel counts differ: "
            f"{len(pc_kernels)} vs {len(bp_kernels)}"
        )
    for l, (C_pc, C_bp) in enumerate(zip(pc_kernels, bp_kernels)):
        cka = float(centered_kernel_alignment(C_pc, C_bp, eps=1e-30))
        gap = abs(1.0 - cka)
        print(
            f"  init CKA (seed={seed}, layer={l + 1}): {cka:.6f} "
            f"(1-CKA={gap:.2e})"
        )
        if gap > atol:
            raise AssertionError(
                f"PC vs BP init kernels differ at layer {l + 1} "
                f"(seed={seed}): CKA={cka:.6f}, expected "
                f"1-CKA <= {atol}"
            )


def _mean_pairwise_cka(kernels):
    """Mean and std of pairwise CKA over a list of kernels (one per seed)."""
    vals = []
    for i in range(len(kernels)):
        for j in range(i + 1, len(kernels)):
            vals.append(
                centered_kernel_alignment(kernels[i], kernels[j], eps=1e-30)
            )
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan")
    std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    return float(vals.mean()), std


def _train_finite_bp(
    key,
    *,
    results_dir,
    input_dim,
    output_dim,
    n_samples,
    n_hidden,
    use_skips,
    act_fn,
    param_type,
    param_lr,
    gamma_0,
    param_optim_id,
    n_train_iters,
    width,
    loss_id,
    seed,
    X_input,
    Y_target,
    collect_h_k0=True,
    init_from=None,
):
    """Run one finite-width BP training job.

    Returns ``(losses, h_k0_traj)``. ``h_k0_traj`` has shape
    ``(n_hidden, T, P, N)`` — the hidden pre-activations ``h^l`` (see
    ``bp_hidden_preactivations`` in ``experiments.dmft.utils``) at every
    training step, before that step's parameter update — or ``None`` if
    ``collect_h_k0`` is False.

    If ``init_from`` is given (a ``jpc.make_mlp`` layer list or BP
    ``MLP``), Linear weights are copied onto the BP network after
    construction so PC and BP share the same initial parameters.
    """
    save_dir = setup_bp_experiment(
        results_dir=results_dir,
        input_dim=input_dim,
        n_samples=n_samples,
        n_hidden=n_hidden,
        use_skips=use_skips,
        act_fn=act_fn,
        param_type=param_type,
        optim_id=param_optim_id,
        param_lr=param_lr,
        gamma_0=gamma_0,
        n_train_iters=n_train_iters,
        width=width,
        loss_id=loss_id,
        seed=seed,
    )
    model = MLP(
        key=key,
        d_in=input_dim,
        N=width,
        L=n_hidden + 1,
        d_out=output_dim,
        act_fn=act_fn,
        param_type=param_type,
        gamma=gamma_0,
        use_bias=False,
        use_skips=use_skips,
    )
    if init_from is not None:
        model = copy_mlp_linear_params(init_from, model)
        print("Copied PC initial Linear weights onto the BP network.")
    h_k0_steps = [] if collect_h_k0 else None
    train_bpn(
        model=model,
        use_skips=use_skips,
        X_input=X_input,
        Y_target=Y_target,
        width=width,
        gamma_0=gamma_0,
        param_type=param_type,
        optim_id=param_optim_id,
        param_lr=param_lr,
        n_train_iters=n_train_iters,
        save_dir=save_dir,
        store_grads=False,
        loss_id=loss_id,
        h_k0_steps=h_k0_steps,
    )
    losses = np.load(f"{save_dir}/losses.npy")
    h_k0_traj = np.stack(h_k0_steps, axis=1) if collect_h_k0 else None
    return losses, h_k0_traj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "tiny-CIFAR10", "Fashion-MNIST", "CIFAR10"])
    parser.add_argument("--input_dim", type=int, default=40)
    parser.add_argument("--n_samples", type=int, default=20)

    # Model parameters
    parser.add_argument("--act_fn", type=str, default="linear", choices=["linear", "tanh", "relu"])
    parser.add_argument("--param_type", type=str, default="mupc", choices=["mupc", "sp", "my-mup"])
    parser.add_argument("--use_skips", action="store_true", default=False)
    parser.add_argument("--n_hidden", type=int, default=5)
    parser.add_argument("--width", type=int, default=2048)

    # Training parameters (shared)
    parser.add_argument("--gamma_0", type=float, default=1.0)
    parser.add_argument("--n_train_iters", type=int, default=20)
    parser.add_argument("--loss_id", type=str, default="mse", choices=["mse", "ce"])

    # BP training parameters
    parser.add_argument("--param_lr", type=float, default=0.05)
    parser.add_argument("--param_optim", type=str, default="gd", choices=["gd", "adam"])

    # PC training / inference parameters
    parser.add_argument("--param_lr_pc", type=float, default=0.5)
    parser.add_argument(
        "--pc_infer_mode",
        type=str,
        default="infer",
        choices=["infer", "closed_form"],
        help=(
            "PC inference mode. 'infer' runs --n_infer_iters steps of "
            "gradient descent on the activities each training step. "
            "'closed_form' solves the linear equilibrium activities "
            "directly (requires --act_fn linear)."
        ),
    )
    parser.add_argument("--n_infer_iters", type=int, default=5)
    parser.add_argument("--activity_lr", type=float, default=0.05)
    parser.add_argument(
        "--nonlin_beta",
        type=float,
        default=1.0,
        help="Steepness for the tanh feature nonlinearity phi.",
    )

    # Timepoints
    parser.add_argument(
        "--n_timepoints",
        type=int,
        default=6,
        help=(
            "Number of equally spaced training-step timepoints "
            "(including t=0 and t=T-1) used for the feature-kernel "
            "heatmap grids. Displacement and alignment line plots use a "
            "denser grid (every step if T<=31, else every 2 steps)."
        ),
    )

    # Loop parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=1,
        help=(
            "Number of independent weight-init seeds (dataset is shared). "
            "If greater than 1, plot kernel concentration across seeds "
            "for both PC and BP. Default 1 skips that analysis."
        ),
    )

    parser.add_argument(
        "--cleanup_npy",
        action="store_true",
        default=True,
        help=(
            "After the run, delete finite-sim result directories "
            "(*_input_dim under results_dir), keeping plot pngs."
        ),
    )
    args = parser.parse_args()

    if args.pc_infer_mode == "closed_form" and args.act_fn != "linear":
        parser.error(
            "--pc_infer_mode closed_form requires --act_fn linear."
        )
    if args.n_seeds < 1:
        parser.error("--n_seeds must be >= 1.")

    os.makedirs(args.results_dir, exist_ok=True)

    # Two independent children of --seed: dataset and weight init. BP
    # copies the PC Linear weights after construction, so both networks
    # start from the same parameters (the BP constructor key is unused
    # for the copied weights).
    data_key, model_parent = jax.random.split(jax.random.PRNGKey(args.seed))

    if args.dataset == "toy":
        X, y = create_toy_dataset(
            key=data_key, D=args.input_dim, P=args.n_samples
        )
        input_dim = args.input_dim
        output_dim = 1
    elif args.dataset == "tiny-CIFAR10":
        input_dim = CIFAR_GRAY_DIM
        X, y = create_tiny_cifar10_dataset(
            key=data_key, D=input_dim, P=args.n_samples
        )
        output_dim = 1
        print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    else:
        from torch import Generator as TorchGenerator

        loader_gen = TorchGenerator()
        loader_gen.manual_seed(int(np.asarray(data_key)[0]) & 0x7FFFFFFF)
        train_loader, _ = get_dataloaders(
            args.dataset, args.n_samples, generator=loader_gen
        )
        img_batch, label_batch = next(iter(train_loader))

        input_dim = img_batch.shape[1]
        output_dim = label_batch.shape[1]
        print(f"Input dim: {input_dim}, Output dim: {output_dim}")

        X = img_batch.numpy().T
        y = label_batch.numpy()

    X_input = jnp.asarray(X.T, dtype=jnp.float32)
    Y_target = y[:, None] if y.ndim == 1 else y
    Y_target = jnp.asarray(Y_target, dtype=jnp.float32)
    C_y = _target_kernel(Y_target)
    C_x = _input_kernel(X_input)
    n_label_cols = 1 if np.asarray(Y_target).ndim == 1 else int(
        np.asarray(Y_target).shape[1]
    )
    print(
        f"Target kernel C^y shape: {C_y.shape}, "
        f"input kernel C^x shape: {C_x.shape}"
    )
    loss_id = (
        "mse" if args.dataset in ("toy", "tiny-CIFAR10") else args.loss_id
    )

    n_hidden = args.n_hidden
    T_train = args.n_train_iters
    width = args.width
    infer_mode = "optim" if args.pc_infer_mode == "infer" else "closed_form"
    n_seeds = args.n_seeds
    phi_fn, _ = get_nonlinearity(args.act_fn, beta=args.nonlin_beta)
    feat_sym = feature_kernel_symbol(args.act_fn)
    feat_tex = r"\phi" if feat_sym == "phi" else "h"
    heatmap_timepoints = _select_timepoints(T_train, args.n_timepoints)
    curve_timepoints = _select_curve_timepoints(T_train)
    kernel_times = sorted(set(heatmap_timepoints) | set(curve_timepoints))
    print(
        f"\nHeatmap timepoints t = {heatmap_timepoints} (of T = {T_train})"
    )
    print(f"Curve timepoints t = {curve_timepoints}")
    print(f"Feature kernel: C^{feat_sym}")
    print(f"n_seeds = {n_seeds}\n")

    plots_dir = os.path.join(
        args.results_dir,
        "plots",
        f"{width}_width",
        f"{args.pc_infer_mode}_pc_infer_mode",
    )
    plot_kw = dict(
        plots_dir=plots_dir,
        gamma_0=args.gamma_0,
        n_hidden=n_hidden,
        activity_lr=args.activity_lr,
        n_infer_iters=args.n_infer_iters,
        width=width,
        feature_symbol=feat_sym,
    )

    pc_kernels_by_seed = []
    bp_kernels_by_seed = []
    plot_seed = args.seed

    for seed in range(args.seed, args.seed + n_seeds):
        print(f"\n=== seed {seed} ===")
        set_seed(seed)
        model_key = jax.random.fold_in(model_parent, int(seed))

        print(
            f"\nRunning finite-size PC ({args.pc_infer_mode}) simulation "
            f"(N={width}, H={n_hidden})...\n"
        )
        pc_losses, pc_fields = _train_finite_pc(
            model_key,
            results_dir=args.results_dir,
            input_dim=input_dim,
            output_dim=output_dim,
            n_samples=args.n_samples,
            n_hidden=n_hidden,
            use_skips=args.use_skips,
            act_fn=args.act_fn,
            param_type=args.param_type,
            param_lr=args.param_lr_pc,
            gamma_0=args.gamma_0,
            param_optim_id=args.param_optim,
            n_train_iters=T_train,
            infer_mode=infer_mode,
            n_infer_iters=args.n_infer_iters,
            activity_lr=args.activity_lr,
            width=width,
            loss_id=loss_id,
            seed=seed,
            X_input=X_input,
            Y_target=Y_target,
            collect_fields=False,
            collect_h_k0=True,
            collect_init_model=True,
        )
        pc_h_traj = pc_fields["h_k0_traj"]  # (n_hidden, T, P, N)
        print(
            f"PC final training loss: "
            f"{float(np.asarray(pc_losses).flatten()[-1]):.4e}"
        )

        print(
            f"\nRunning finite-size BP simulation "
            f"(N={width}, H={n_hidden})...\n"
        )
        bp_losses, bp_h_traj = _train_finite_bp(
            model_key,
            results_dir=args.results_dir,
            input_dim=input_dim,
            output_dim=output_dim,
            n_samples=args.n_samples,
            n_hidden=n_hidden,
            use_skips=args.use_skips,
            act_fn=args.act_fn,
            param_type=args.param_type,
            param_lr=args.param_lr,
            gamma_0=args.gamma_0,
            param_optim_id=args.param_optim,
            n_train_iters=T_train,
            width=width,
            loss_id=loss_id,
            seed=seed,
            X_input=X_input,
            Y_target=Y_target,
            collect_h_k0=True,
            init_from=pc_fields["init_model"],
        )
        print(
            f"BP final training loss: "
            f"{float(np.asarray(bp_losses).flatten()[-1]):.4e}"
        )

        pc_kernels_by_t = {
            t: _feature_kernels_from_h(pc_h_traj[:, t], phi_fn)
            for t in kernel_times
        }
        bp_kernels_by_t = {
            t: _feature_kernels_from_h(bp_h_traj[:, t], phi_fn)
            for t in kernel_times
        }
        t0 = 0 if 0 in pc_kernels_by_t else kernel_times[0]
        print("Sanity-checking PC vs BP feature kernels at t=0...")
        _assert_init_pc_bp_cka(
            pc_kernels_by_t[t0], bp_kernels_by_t[t0], seed=seed
        )
        pc_kernels_by_seed.append(pc_kernels_by_t)
        bp_kernels_by_seed.append(bp_kernels_by_t)

        if seed != plot_seed:
            continue

        plot_pc_bp_loss(
            pc_losses,
            bp_losses,
            plots_dir=plots_dir,
            n_hidden=n_hidden,
            gamma_0=args.gamma_0,
            activity_lr=args.activity_lr,
            n_infer_iters=args.n_infer_iters,
            width=width,
        )

        for t in heatmap_timepoints:
            plot_final_kernel_grid(
                [
                    ("PC", kernels_to_correlations(pc_kernels_by_t[t])),
                    ("Backprop", kernels_to_correlations(bp_kernels_by_t[t])),
                ],
                plots_dir=plots_dir,
                gamma_0=args.gamma_0,
                n_hidden=n_hidden,
                activity_lr=args.activity_lr,
                n_infer_iters=args.n_infer_iters,
                width=width,
                filename=f"feature_kernels_grid_t{t}.png",
                share_clim=True,
                vmin=-1.0,
                vmax=1.0,
                title=(
                    rf"$C^{{{feat_tex}}}$ feature kernels "
                    rf"($t={t}$, correlation)"
                ),
                dir_name="alignment",
            )

        displacement_records = []
        alignment_records = []
        change_alignment_records = []
        target_alignment_records = []
        rank_records = []
        change_target_alignment_records = []
        input_alignment_records = []
        evec_overlap_records = []
        subspace_overlap_records = []
        for t in curve_timepoints:
            for l in range(n_hidden):
                pc_C0 = pc_kernels_by_t[t0][l]
                bp_C0 = bp_kernels_by_t[t0][l]
                pc_Ct = pc_kernels_by_t[t][l]
                bp_Ct = bp_kernels_by_t[t][l]
                displacement_records.append({
                    "t": t,
                    "layer": l,
                    "method": "pc",
                    "displacement": float(
                        cosine_similarity(pc_C0, pc_Ct, eps=1e-30)
                    ),
                    "rel_displacement": float(
                        relative_frobenius_displacement(
                            pc_C0, pc_Ct, eps=1e-30
                        )
                    ),
                })
                displacement_records.append({
                    "t": t,
                    "layer": l,
                    "method": "bp",
                    "displacement": float(
                        cosine_similarity(bp_C0, bp_Ct, eps=1e-30)
                    ),
                    "rel_displacement": float(
                        relative_frobenius_displacement(
                            bp_C0, bp_Ct, eps=1e-30
                        )
                    ),
                })
                alignment_records.append({
                    "t": t,
                    "layer": l,
                    "alignment": float(
                        centered_kernel_alignment(pc_Ct, bp_Ct, eps=1e-30)
                    ),
                })
                if t != t0:
                    change_alignment_records.append({
                        "t": t,
                        "layer": l,
                        "alignment": float(
                            centered_kernel_alignment(
                                np.asarray(pc_Ct) - np.asarray(pc_C0),
                                np.asarray(bp_Ct) - np.asarray(bp_C0),
                                eps=1e-30,
                            )
                        ),
                    })
                for method, Ct in (("pc", pc_Ct), ("bp", bp_Ct)):
                    target_alignment_records.append({
                        "t": t,
                        "layer": l,
                        "method": method,
                        "alignment": float(
                            centered_kernel_alignment(Ct, C_y, eps=1e-30)
                        ),
                    })
                    rank_records.append({
                        "t": t,
                        "layer": l,
                        "method": method,
                        "effective_rank": float(participation_ratio(Ct)),
                    })
                    input_alignment_records.append({
                        "t": t,
                        "layer": l,
                        "method": method,
                        "alignment": float(
                            centered_kernel_alignment(Ct, C_x, eps=1e-30)
                        ),
                    })
                    evec_overlap_records.append({
                        "t": t,
                        "layer": l,
                        "method": method,
                        "overlap": float(
                            leading_evec_label_overlap(Ct, Y_target)
                        ),
                    })
                if t != t0:
                    for method, Ct, C0 in (
                        ("pc", pc_Ct, pc_C0),
                        ("bp", bp_Ct, bp_C0),
                    ):
                        change_target_alignment_records.append({
                            "t": t,
                            "layer": l,
                            "method": method,
                            "alignment": float(
                                centered_kernel_alignment(
                                    np.asarray(Ct) - np.asarray(C0),
                                    C_y,
                                    eps=1e-30,
                                )
                            ),
                        })
                max_k = max(1, int(C_y.shape[0]) - 1)
                for k in _SUBSPACE_KS:
                    if k > max_k:
                        continue
                    subspace_overlap_records.append({
                        "t": t,
                        "layer": l,
                        "k": k,
                        "overlap": float(
                            subspace_overlap(pc_Ct, bp_Ct, k, centered=True)
                        ),
                    })

        plot_kernel_displacement_per_timepoint(
            pd.DataFrame(displacement_records),
            **plot_kw,
        )
        plot_kernel_displacement_per_timepoint(
            pd.DataFrame(displacement_records),
            metric="rel_frob",
            **plot_kw,
        )
        plot_pc_bp_alignment_vs_time(
            pd.DataFrame(alignment_records),
            **plot_kw,
        )
        plot_pc_bp_alignment_vs_time(
            pd.DataFrame(change_alignment_records),
            ylabel=(
                rf"$\mathrm{{CKA}}(\Delta C^{{{feat_tex},\ell}}"
                rf"_{{\mathrm{{PC}}}}(t), "
                rf"\Delta C^{{{feat_tex},\ell}}_{{\mathrm{{BP}}}}(t))$"
            ),
            filename="pc_bp_kernel_change_alignment_vs_time.png",
            title=(
                "PC vs backprop feature-kernel change alignment over training"
            ),
            **plot_kw,
        )
        plot_kernel_target_alignment_vs_time(
            pd.DataFrame(target_alignment_records),
            **plot_kw,
        )
        plot_kernel_target_change_alignment_vs_time(
            pd.DataFrame(change_target_alignment_records),
            **plot_kw,
        )
        plot_kernel_input_alignment_vs_time(
            pd.DataFrame(input_alignment_records),
            **plot_kw,
        )
        evec_ylabel = (
            r"$\left|\cos(v_1^{\ell}(t), y)\right|$"
            if n_label_cols == 1
            else r"$\|U_y^{\top} v_1^{\ell}(t)\|$"
        )
        plot_leading_evec_label_overlap_vs_time(
            pd.DataFrame(evec_overlap_records),
            ylabel=evec_ylabel,
            **plot_kw,
        )
        plot_kernel_effective_rank_vs_time(
            pd.DataFrame(rank_records),
            **plot_kw,
        )
        plot_pc_bp_subspace_overlap_vs_time(
            pd.DataFrame(subspace_overlap_records),
            **plot_kw,
        )

        t_final = kernel_times[-1]
        final_spectrum_df = pd.DataFrame(
            _spectrum_records(pc_kernels_by_t[t_final], "pc")
            + _spectrum_records(bp_kernels_by_t[t_final], "bp")
        )
        plot_kernel_spectrum(
            final_spectrum_df,
            ylabel=rf"$\lambda_i(C^{{{feat_tex},\ell}})$",
            title=(
                rf"Final $C^{{{feat_tex}}}$ feature-kernel spectrum "
                rf"($t={t_final}$)"
            ),
            filename="kernel_spectrum_final.png",
            annotate_rank=True,
            **plot_kw,
        )

        pc_temporal_kernels = _sample_traced_feature_kernels_from_h_traj(
            pc_h_traj, phi_fn
        )
        bp_temporal_kernels = _sample_traced_feature_kernels_from_h_traj(
            bp_h_traj, phi_fn
        )
        plot_temporal_kernel_grid(
            [
                ("PC", kernels_to_correlations(pc_temporal_kernels)),
                ("Backprop", kernels_to_correlations(bp_temporal_kernels)),
            ],
            plots_dir=plots_dir,
            gamma_0=args.gamma_0,
            n_hidden=n_hidden,
            activity_lr=args.activity_lr,
            n_infer_iters=args.n_infer_iters,
            width=width,
            filename="temporal_kernels_grid.png",
            dir_name="alignment",
            vmin=-1.0,
            vmax=1.0,
            title=(
                rf"Sample-traced $C^{{{feat_tex}}}$ feature kernels "
                rf"(correlation)"
            ),
        )

        temporal_spectrum_df = pd.DataFrame(
            _spectrum_records(pc_temporal_kernels, "pc")
            + _spectrum_records(bp_temporal_kernels, "bp")
        )
        plot_kernel_spectrum(
            temporal_spectrum_df,
            ylabel=rf"$\lambda_i(C^{{{feat_tex},\ell}}_{{\mathrm{{temp}}}})$",
            title=rf"Sample-traced $C^{{{feat_tex}}}$ feature-kernel spectrum",
            filename="temporal_kernel_spectrum.png",
            annotate_rank=True,
            **plot_kw,
        )
        temporal_rank_records = (
            temporal_spectrum_df.groupby(["layer", "method"], as_index=False)
            .agg(effective_rank=("effective_rank", "first"))
        )
        plot_temporal_kernel_effective_rank_vs_layer(
            temporal_rank_records,
            **plot_kw,
        )
        temporal_cka_records = []
        for l, (C_pc, C_bp) in enumerate(
            zip(pc_temporal_kernels, bp_temporal_kernels)
        ):
            cka = float(centered_kernel_alignment(C_pc, C_bp, eps=1e-30))
            temporal_cka_records.append({
                "layer": l,
                "alignment": cka,
            })
            print(
                f"  temporal PC-BP CKA (layer={l + 1}): {cka:.4f}"
            )
        plot_temporal_pc_bp_alignment_vs_layer(
            pd.DataFrame(temporal_cka_records),
            **plot_kw,
        )
        for _, row in temporal_rank_records.sort_values(
            ["layer", "method"]
        ).iterrows():
            print(
                f"  temporal R_eff (layer={int(row['layer']) + 1}, "
                f"{row['method']}): {float(row['effective_rank']):.4f}"
            )

    if n_seeds > 1:
        print(
            f"\nKernel concentration across {n_seeds} seeds "
            f"(mean pairwise CKA)..."
        )
        conc_records = []
        for t in curve_timepoints:
            for l in range(n_hidden):
                for method, by_seed in (
                    ("pc", pc_kernels_by_seed),
                    ("bp", bp_kernels_by_seed),
                ):
                    kernels = [seed_kernels[t][l] for seed_kernels in by_seed]
                    cka_mean, cka_std = _mean_pairwise_cka(kernels)
                    conc_records.append({
                        "t": t,
                        "layer": l,
                        "method": method,
                        "cka_mean": cka_mean,
                        "cka_std": cka_std,
                    })
        conc_df = pd.DataFrame(conc_records)
        plot_kernel_concentration_per_timepoint(
            conc_df, n_seeds=n_seeds, **plot_kw
        )
        plot_kernel_concentration_vs_time(
            conc_df, n_seeds=n_seeds, **plot_kw
        )
    else:
        print(
            "\nSkipping kernel-concentration analysis "
            "(pass --n_seeds > 1 to enable)."
        )

    if args.cleanup_npy:
        removed_dirs = cleanup_experiment_dirs(args.results_dir)
        if removed_dirs:
            print(
                f"\nRemoved {len(removed_dirs)} experiment dir(s) "
                f"under {args.results_dir} (png plots kept):"
            )
            for d in removed_dirs:
                print(f"  - {d}")
        else:
            print(f"\nNo *_input_dim dirs to remove under {args.results_dir}.")


######### LINEAR ##########
###########################

# Infer mode (default)
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 20 --n_hidden 5 --width 10000 --gamma_0 1.0 --param_lr 0.1 --param_lr_pc 0.2 --activity_lr 0.01 --pc_infer_mode infer --n_infer_iters 5 --n_train_iters 21

# Closed-form inference for PC
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 20 --n_hidden 5 --width 10000 --gamma_0 1.0 --param_lr 0.1 --param_lr_pc 0.2 --pc_infer_mode closed_form --n_train_iters 21

# Closed-form inference for PC (tiny-CIFAR10)
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 40 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.05 --param_lr_pc 0.5 --pc_infer_mode closed_form --n_train_iters 501 --dataset tiny-CIFAR10


############ NONLINEAR ##################
#########################################

# Infer mode (default)
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 40 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.5 --param_lr_pc 1.0 --activity_lr 0.05 --pc_infer_mode infer --n_infer_iters 100 --n_train_iters 31 --act_fn tanh --dataset tiny-CIFAR10

# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 40 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.2 --param_lr_pc 0.5 --activity_lr 0.1 --pc_infer_mode infer --n_infer_iters 200 --n_train_iters 101 --act_fn tanh --dataset tiny-CIFAR10

# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 40 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.05 --param_lr_pc 0.5 --activity_lr 0.1 --pc_infer_mode infer --n_infer_iters 200 --n_train_iters 801 --act_fn tanh --dataset tiny-CIFAR10