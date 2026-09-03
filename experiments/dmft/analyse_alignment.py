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
training step), this produces:

- PC vs BP training-loss curves
  (``alignment/pc_bp_loss.png``).
- Feature-kernel grid: one figure per timepoint, a ``P x P`` heatmap per
  hidden layer, top row PC / bottom row backprop
  (``alignment/feature_kernels_grid_t{t}.png``).
- Kernel displacement: one figure with a subplot per timepoint (auto
  grid), cosine similarity of each layer's feature kernel at ``t`` vs.
  at ``t=0``, PC and BP curves, x-axis is layer
  (``alignment/kernel_displacement_vs_layer_grid.png``).
- PC-BP alignment vs. time: a single figure, cosine similarity between
  the PC and backprop feature kernels at each layer, x-axis is training
  time ``t``, one curve per layer
  (``alignment/pc_bp_kernel_alignment_vs_time.png``).
- Sample-traced temporal kernels: a single figure using the same grid
  scheme as the feature-kernel grid (top row PC / bottom row backprop,
  one column per layer), but with ``T x T`` kernels traced over samples
  and spanning the whole training trajectory at once
  (``alignment/temporal_kernels_grid.png``).

``--seed`` draws two independent RNG streams (dataset, weight init); PC
and BP get independent weight-init keys folded from the same parent so
neither reuses the other's randomness. PC and BP always train on the
same ``(X, y)``.

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
    create_tiny_cifar10_dataset,
    create_toy_dataset,
    cleanup_experiment_dirs,
    cosine_similarity,
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
    plot_pc_bp_alignment_vs_time,
    plot_pc_bp_loss,
)


def _select_timepoints(n_train_iters, n_timepoints):
    """``n_timepoints`` equally spaced training-step indices.

    Always includes both endpoints (``t=0`` and ``t=n_train_iters - 1``).
    If ``n_timepoints >= n_train_iters``, every training step is used.
    """
    n_timepoints = max(2, min(n_timepoints, n_train_iters))
    idx = np.round(
        np.linspace(0, n_train_iters - 1, n_timepoints)
    ).astype(int)
    return np.unique(idx).tolist()


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
):
    """Run one finite-width BP training job.

    Returns ``(losses, h_k0_traj)``. ``h_k0_traj`` has shape
    ``(n_hidden, T, P, N)`` — the hidden pre-activations ``h^l`` (see
    ``bp_hidden_preactivations`` in ``experiments.dmft.utils``) at every
    training step, before that step's parameter update — or ``None`` if
    ``collect_h_k0`` is False.
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
            "grid, displacement, and alignment-vs-time plots."
        ),
    )

    # Loop parameters
    parser.add_argument("--seed", type=int, default=0)

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

    os.makedirs(args.results_dir, exist_ok=True)

    # Two independent children of --seed: dataset and weight init. PC and
    # BP weight-init keys are folded from the same parent so neither
    # reuses the other's (or the data's) randomness.
    data_key, model_parent = jax.random.split(jax.random.PRNGKey(args.seed))
    model_key = jax.random.fold_in(model_parent, int(args.seed))
    pc_key, bp_key = jax.random.split(model_key)

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
    loss_id = (
        "mse" if args.dataset in ("toy", "tiny-CIFAR10") else args.loss_id
    )

    set_seed(args.seed)

    n_hidden = args.n_hidden
    T_train = args.n_train_iters
    width = args.width
    infer_mode = "optim" if args.pc_infer_mode == "infer" else "closed_form"

    # --- Finite-size PC simulation ---
    print(
        f"\nRunning finite-size PC ({args.pc_infer_mode}) simulation "
        f"(N={width}, H={n_hidden})...\n"
    )
    pc_losses, pc_fields = _train_finite_pc(
        pc_key,
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
        seed=args.seed,
        X_input=X_input,
        Y_target=Y_target,
        collect_fields=False,
        collect_h_k0=True,
    )
    pc_h_traj = pc_fields["h_k0_traj"]  # (n_hidden, T, P, N)
    print(f"PC final training loss: {float(np.asarray(pc_losses).flatten()[-1]):.4e}")

    # --- Finite-size BP simulation ---
    print(
        f"\nRunning finite-size BP simulation (N={width}, H={n_hidden})...\n"
    )
    bp_losses, bp_h_traj = _train_finite_bp(
        bp_key,
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
        seed=args.seed,
        X_input=X_input,
        Y_target=Y_target,
        collect_h_k0=True,
    )
    print(f"BP final training loss: {float(np.asarray(bp_losses).flatten()[-1]):.4e}")

    phi_fn, _ = get_nonlinearity(args.act_fn, beta=args.nonlin_beta)
    feat_sym = feature_kernel_symbol(args.act_fn)
    feat_tex = r"\phi" if feat_sym == "phi" else "h"

    timepoints = _select_timepoints(T_train, args.n_timepoints)
    print(f"\nUsing timepoints t = {timepoints} (of T = {T_train})")
    print(f"Feature kernel: C^{feat_sym}\n")

    # Feature kernels (P x P), per layer, at every selected timepoint.
    # Linear: C^h (phi = id). Nonlinear: C^phi = phi(h) phi(h)^T / N.
    pc_kernels_by_t = {
        t: _feature_kernels_from_h(pc_h_traj[:, t], phi_fn)
        for t in timepoints
    }
    bp_kernels_by_t = {
        t: _feature_kernels_from_h(bp_h_traj[:, t], phi_fn)
        for t in timepoints
    }

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

    # --- Feature-kernel grid: one figure per timepoint, top=PC, bottom=BP ---
    for t in timepoints:
        plot_final_kernel_grid(
            [
                ("PC", pc_kernels_by_t[t]),
                ("Backprop", bp_kernels_by_t[t]),
            ],
            plots_dir=plots_dir,
            gamma_0=args.gamma_0,
            n_hidden=n_hidden,
            activity_lr=args.activity_lr,
            n_infer_iters=args.n_infer_iters,
            width=width,
            filename=f"feature_kernels_grid_t{t}.png",
            share_clim=True,
            title=rf"$C^{{{feat_tex}}}$ feature kernels ($t={t}$)",
            dir_name="alignment",
        )

    # --- Kernel displacement from t0 and PC-BP alignment, per layer ---
    t0 = timepoints[0]
    displacement_records = []
    alignment_records = []
    for t in timepoints:
        for l in range(n_hidden):
            displacement_records.append({
                "t": t,
                "layer": l,
                "method": "pc",
                "displacement": float(
                    cosine_similarity(
                        pc_kernels_by_t[t0][l],
                        pc_kernels_by_t[t][l],
                        eps=1e-30,
                    )
                ),
            })
            displacement_records.append({
                "t": t,
                "layer": l,
                "method": "bp",
                "displacement": float(
                    cosine_similarity(
                        bp_kernels_by_t[t0][l],
                        bp_kernels_by_t[t][l],
                        eps=1e-30,
                    )
                ),
            })
            alignment_records.append({
                "t": t,
                "layer": l,
                "alignment": float(
                    cosine_similarity(
                        pc_kernels_by_t[t][l],
                        bp_kernels_by_t[t][l],
                        eps=1e-30,
                    )
                ),
            })

    plot_kernel_displacement_per_timepoint(
        pd.DataFrame(displacement_records),
        **plot_kw,
    )

    plot_pc_bp_alignment_vs_time(
        pd.DataFrame(alignment_records),
        **plot_kw,
    )

    # --- Sample-traced temporal kernels (T x T): same grid scheme, one plot ---
    pc_temporal_kernels = _sample_traced_feature_kernels_from_h_traj(
        pc_h_traj, phi_fn
    )
    bp_temporal_kernels = _sample_traced_feature_kernels_from_h_traj(
        bp_h_traj, phi_fn
    )
    plot_temporal_kernel_grid(
        [
            ("PC", pc_temporal_kernels),
            ("Backprop", bp_temporal_kernels),
        ],
        plots_dir=plots_dir,
        gamma_0=args.gamma_0,
        n_hidden=n_hidden,
        activity_lr=args.activity_lr,
        n_infer_iters=args.n_infer_iters,
        width=width,
        filename="temporal_kernels_grid.png",
        dir_name="alignment",
        title=rf"Sample-traced $C^{{{feat_tex}}}$ feature kernels",
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
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 20 --n_hidden 5 --width 10000 --gamma_0 1.0 --param_lr 0.1 --param_lr_pc 0.2 --activity_lr 0.01 --pc_infer_mode infer --n_infer_iters 5 --n_train_iters 20

# Closed-form PC inference
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 20 --n_hidden 5 --width 10000 --gamma_0 1.0 --param_lr 0.1 --param_lr_pc 0.2 --pc_infer_mode closed_form --n_train_iters 20


############ NONLINEAR ##################
#########################################

# Infer mode (default)
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 10 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.5 --param_lr_pc 1.0 --activity_lr 0.05 --pc_infer_mode infer --n_infer_iters 10 --n_train_iters 30 --act_fn tanh --dataset tiny-CIFAR10
