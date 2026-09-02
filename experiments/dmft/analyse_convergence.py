"""PC DMFT theory vs finite-width analysis: kernels, width alignment, and loss.

Kernel / width figures are selected by ``--plot_mode`` (default ``auto``).
Loss curves are always produced, including overlays when ``--n_hiddens``,
``--gamma_0s``, or ``--n_infer_iters`` contain multiple values.

Kernel / width figures:
- ``kernels``: final feature-kernel grid (closed-form / infer / DMFT;
  closed-form omitted for nonlinear nets) at the largest ``--widths``
  value and the first seed. ``--skip_theory`` drops the DMFT row (and
  skips solving theory) on this path only. ``--skip_closed_form`` drops
  the closed-form row and does not run closed-form inference.
- ``width``: finite-width vs theory kernel alignment against width, for
  all hidden layers (``C^h`` at ``k=0``; ``C^Δ`` at the last inference
  step ``k=K``; readout omitted).
- ``both``: solve DMFT once, then run the ``kernels`` grid (largest
  width, first seed) and the ``width`` alignment sweep.
- ``auto``: ``kernels`` if a single ``--widths`` value is given, else
  ``both``. Extra ``K`` values (when several ``--n_infer_iters`` are
  given) do not get kernel figures in ``auto``; pass ``kernels`` /
  ``width`` / ``both`` to request them.

Loss figures:
- Single ``H``, ``gamma_0``, and ``K``: theory vs finite-size loss
  (all widths) for that combination.
- Several ``--n_hiddens``: overlay loss vs depth at the largest width.
- Several ``--gamma_0s``: overlay loss vs ``gamma`` at the largest width.
- Several ``--n_infer_iters``: overlay loss vs ``K`` at the largest
  width. DMFT is solved and plotted only for the smallest ``K`` unless
  kernel figures for the other ``K`` are requested (explicit
  ``--plot_mode``). Linear nets also get one finite-size closed-form
  curve (independent of ``K``).

``--skip_theory`` skips DMFT on the kernel-grid-only path and on loss
plots; it has no effect when the width-alignment path runs (theory is
required). ``--skip_finite`` skips finite-width simulations (theory
loss only; kernel figures are dropped).

Theory is solved once per hyperparameter combination and reused across
seeds. ``--seed`` draws three independent RNG streams (dataset, weight
init, DMFT Monte Carlo) so theory and finite-size always share the same
``(X, y)`` without sharing PRNG keys. Finite-size trials only vary
Gaussian weight initialisation.

Use the ``PC_dmft_env`` conda environment:
    /data/ndcn-computational-neuroscience/mert5001/envs/PC_dmft_env/bin/python analyse_convergence.py
"""

import os
import sys
from pathlib import Path

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import jpc

from experiments.datasets import get_dataloaders
from experiments.mupc_paper.utils import set_seed
from experiments.limits_paper.utils import setup_pc_experiment
from experiments.dmft.utils import (
    CIFAR_GRAY_DIM,
    create_tiny_cifar10_dataset,
    create_toy_dataset,
    cleanup_experiment_dirs,
    train_pcn,
    collect_final_pc_kernel_fields,
    cosine_similarity,
    empirical_pc_kernel,
    final_time_pc_kernel,
)
from theory_pc_utils import solve_pc_kernels
from theory_pc_nonlin_utils import solve_pc_kernels_nonlin, get_nonlinearity
from plot_dmft_results import (
    plot_pc_theory_vs_finite_loss,
    plot_pc_param_sweep_loss,
    plot_pc_kernel_width_alignment,
    plot_final_kernel_grid,
)


def _train_finite_pc(
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
    infer_mode,
    n_infer_iters,
    activity_lr,
    width,
    loss_id,
    seed,
    X_input,
    Y_target,
    collect_fields=True,
):
    """Run one finite-width PC training job.

    Returns the loss trajectory and, when ``collect_fields`` is True,
    hidden ``h`` at ``k=0`` / ``Δ`` at the last inference step
    ``k=n_infer_iters`` of the trained network (else ``None``).
    """
    save_dir = setup_pc_experiment(
        results_dir=results_dir,
        input_dim=input_dim,
        n_samples=n_samples,
        n_hidden=n_hidden,
        use_skips=use_skips,
        act_fn=act_fn,
        param_type=param_type,
        param_lr=param_lr,
        gamma_0=gamma_0,
        param_optim_id=param_optim_id,
        n_train_iters=n_train_iters,
        infer_mode=infer_mode,
        n_infer_iters=n_infer_iters,
        activity_lr=activity_lr,
        width=width,
        loss_id=loss_id,
        seed=seed,
    )
    model = jpc.make_mlp(
        key,
        input_dim=input_dim,
        width=width,
        depth=n_hidden + 1,
        output_dim=output_dim,
        act_fn=act_fn,
        use_bias=False,
        param_type=param_type,
    )
    _, model, skip_model = train_pcn(
        model=model,
        use_skips=use_skips,
        X_input=X_input,
        Y_target=Y_target,
        width=width,
        gamma_0=gamma_0,
        param_type=param_type,
        infer_mode=infer_mode,
        n_infer_iters=n_infer_iters,
        activity_lr=activity_lr,
        param_optim_id=param_optim_id,
        param_lr=param_lr,
        n_train_iters=n_train_iters,
        save_dir=save_dir,
        store_grads=False,
        loss_id=loss_id,
    )
    losses = np.load(f"{save_dir}/train_losses.npy")
    if not collect_fields:
        return losses, None
    fields = collect_final_pc_kernel_fields(
        model=model,
        skip_model=skip_model,
        X_input=X_input,
        Y_target=Y_target,
        width=width,
        param_type=param_type,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        loss_id=loss_id,
        n_infer_iters=n_infer_iters,
    )
    return losses, fields


def _loss_records(losses, **meta):
    records = []
    for t, loss in enumerate(np.asarray(losses).flatten(), start=1):
        records.append({**meta, "t": t, "loss": float(loss)})
    return records


def _kernel_align_records(
    fields,
    all_Ch,
    all_Cdelta,
    phi_fn,
    num_inference_steps,
    num_training_steps,
    num_samples,
    **meta,
):
    """Cosine alignment of last-``t`` kernels, per hidden layer.

    ``C^h`` is compared at ``k=0`` and ``C^Δ`` at the last inference step
    ``k=num_inference_steps``. All ``H`` hidden layers are included; the
    readout layer is omitted.
    """
    records = []
    h_final = fields["h"]
    delta_final = fields["delta"]
    n_hidden = h_final.shape[0]
    if len(all_Ch) != n_hidden or len(all_Cdelta) != n_hidden:
        raise ValueError(
            "theory kernel count does not match finite hidden layers: "
            f"Ch={len(all_Ch)}, Cdelta={len(all_Cdelta)}, N_layers={n_hidden}"
        )
    if n_hidden < 1:
        return records
    slice_kw = dict(
        num_inference_steps=num_inference_steps,
        num_training_steps=num_training_steps,
        num_samples=num_samples,
    )
    for l in range(n_hidden):
        phi_l = np.asarray(phi_fn(jnp.asarray(h_final[l])))
        C_h_ex = empirical_pc_kernel(phi_l)
        C_delta_ex = empirical_pc_kernel(delta_final[l])
        records.append(
            {
                **meta,
                "layer": l,
                "kernel": "h",
                "alignment": float(
                    cosine_similarity(
                        final_time_pc_kernel(all_Ch[l], k=0, **slice_kw),
                        C_h_ex,
                        eps=1e-30,
                    )
                ),
            }
        )
        records.append(
            {
                **meta,
                "layer": l,
                "kernel": "delta",
                "alignment": float(
                    cosine_similarity(
                        final_time_pc_kernel(
                            all_Cdelta[l], k=num_inference_steps, **slice_kw
                        ),
                        C_delta_ex,
                        eps=1e-30,
                    )
                ),
            }
        )
    return records


def _final_feature_kernels(fields, phi_fn):
    """Empirical ``C^h`` at ``k=0`` for every hidden layer (readout omitted)."""
    h_final = fields["h"]
    kernels = []
    for l in range(h_final.shape[0]):
        phi_l = np.asarray(phi_fn(jnp.asarray(h_final[l])))
        kernels.append(empirical_pc_kernel(phi_l))
    return kernels


def _final_dmft_feature_kernels(
    all_Ch, num_inference_steps, num_training_steps, num_samples
):
    """DMFT ``C^h`` sample-sample block at ``k=0`` and last training step."""
    slice_kw = dict(
        num_inference_steps=num_inference_steps,
        num_training_steps=num_training_steps,
        num_samples=num_samples,
    )
    return [
        final_time_pc_kernel(Ch_l, k=0, t=-1, **slice_kw) for Ch_l in all_Ch
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "tiny-CIFAR10", "Fashion-MNIST", "CIFAR10"])
    parser.add_argument("--input_dim", type=int, default=40)
    parser.add_argument("--n_samples", type=int, default=5) # 20)

    # Model parameters
    parser.add_argument("--act_fn", type=str, default="linear", choices=["linear", "tanh", "relu"])
    parser.add_argument("--param_types", type=str, nargs='+', default=["mupc"], choices=["mupc", "sp", "my-mup"])
    parser.add_argument("--use_skips", nargs='+', default=[False])

    # Training parameters
    parser.add_argument("--param_optim", type=str, default="gd")
    parser.add_argument("--gamma_0s", type=float, nargs='+', default=[1])
    parser.add_argument("--n_train_iters", type=int, default=20) # 100)
    parser.add_argument("--loss_id", type=str, default="mse", choices=["mse", "ce"])
    parser.add_argument("--n_fixed_point_steps", type=int, default=10)

    # Inference parameters
    parser.add_argument("--param_lr_pc", type=float, default=0.5)
    parser.add_argument("--infer_mode", type=str, default="optim", choices=["optim", "closed_form"])
    parser.add_argument("--n_infer_iters", type=int, nargs='+', default=[5])
    parser.add_argument("--activity_lrs", type=float, nargs='+', default=[0.05])

    # Loop parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--n_hiddens", type=int, nargs='+', default=[5])
    parser.add_argument("--widths", type=int, nargs='+',
        # default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        default=[128, 512, 2048, 8192]
    )
    parser.add_argument(
        "--plot_mode",
        type=str,
        default="auto",
        choices=["auto", "kernels", "width", "both"],
        help=(
            "Which figure(s) to produce after DMFT. 'auto' (default) draws "
            "the final kernel grid when a single --widths value is given, "
            "and both the grid and kernel alignment vs width when several "
            "widths are given. 'kernels' always draws the grid (largest "
            "width, first seed). 'width' always draws alignment vs width. "
            "'both' solves DMFT once then draws both figures."
        ),
    )

    # PC DMFT theory parameters
    parser.add_argument(
        "--nonlin_beta",
        type=float,
        default=1.0,
        help="Steepness for tanh/softplus in nonlinear DMFT theory.",
    )
    parser.add_argument(
        "--num_mc_samples",
        type=int,
        default=1000,
        help="Monte-Carlo samples for nonlinear PC DMFT theory.",
    )
    parser.add_argument(
        "--pc_damping",
        type=float,
        default=1.0,
        help="Kernel mixing factor for PC DMFT fixed-point updates.",
    )
    parser.add_argument(
        "--pc_tolerance",
        type=float,
        default=1e-5,
        help="Early-stop tolerance for PC DMFT fixed-point residual.",
    )
    parser.add_argument(
        "--pc_backend",
        type=str,
        default="optimised",
        choices=["optimised", "reference"],
        help=(
            "PC DMFT linear solver: 'optimised' (default, reduced Delta "
            "system + jitted Jacobi sweep) or 'reference' (full 2n x 2n "
            "block system; slower, for debugging)."
        ),
    )
    parser.add_argument(
        "--num_jacobian_samples",
        type=int,
        default=None,
        help=(
            "MC samples for nonlinear PC response Jacobians "
            "(default: min(num_mc_samples, 200))."
        ),
    )
    parser.add_argument(
        "--jacobian_batch_size",
        type=int,
        default=25,
        help="Batch size for nonlinear PC Jacobian samples.",
    )
    parser.add_argument(
        "--skip_theory",
        action="store_true",
        default=False,
        help=(
            "Skip PC DMFT theory on loss plots and on the kernel-grid-only "
            "path. Has no effect when the width-alignment path also runs "
            "(theory is required)."
        ),
    )
    parser.add_argument(
        "--skip_finite",
        action="store_true",
        default=False,
        help=(
            "Skip finite-width PC simulations. Plots DMFT theory loss "
            "only; kernel-grid and width-alignment figures are dropped."
        ),
    )
    parser.add_argument(
        "--skip_closed_form",
        action="store_true",
        default=False,
        help=(
            "On the kernel-grid path, skip the closed-form finite-size "
            "row and plot infer (and DMFT, unless --skip_theory). Has no "
            "effect on kernel alignment vs width. Nonlinear nets already "
            "omit closed-form."
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
    if args.skip_theory and args.skip_finite:
        parser.error("Cannot set both --skip_theory and --skip_finite.")

    # PC DMFT inverts (K*T*P) matrices; float64 helps stability.
    jax.config.update("jax_enable_x64", True)

    os.makedirs(args.results_dir, exist_ok=True)
    use_nonlin_theory = args.act_fn != "linear"
    overlay_n_hiddens = len(args.n_hiddens) > 1
    overlay_gamma_0s = len(args.gamma_0s) > 1
    overlay_n_infer_iters = len(args.n_infer_iters) > 1
    overlay_any = (
        overlay_n_hiddens or overlay_gamma_0s or overlay_n_infer_iters
    )
    run_widths = list(args.widths)
    if args.plot_mode == "kernels":
        plot_kernels, plot_width = True, False
    elif args.plot_mode == "width":
        plot_kernels, plot_width = False, True
    elif args.plot_mode == "both":
        plot_kernels, plot_width = True, True
    else:
        # auto: single width -> kernel grid only; several widths -> both
        plot_kernels = True
        plot_width = len(run_widths) > 1
    if args.skip_finite:
        if plot_width:
            print(
                "skip_finite: skipping width-alignment "
                "(needs finite-size simulations)"
            )
            plot_width = False
        if plot_kernels:
            print(
                "skip_finite: skipping kernel grid "
                "(needs finite-size simulations)"
            )
            plot_kernels = False
    kernel_plot_width = max(run_widths)
    kernel_grid_seed = args.seed
    # Loss overlays use the largest width; width-alignment needs every
    # width. Kernel-grid-only likewise uses the largest width.
    if overlay_any and not plot_width:
        run_widths = [kernel_plot_width]
    elif plot_kernels and not plot_width:
        run_widths = [kernel_plot_width]
    # Width-alignment always needs theory; skip_theory only applies when
    # the kernel-grid path runs alone (and to loss plots).
    skip_kernel_theory = args.skip_theory and plot_kernels and not plot_width
    skip_loss_theory = args.skip_theory and not plot_width
    skip_closed_form = args.skip_closed_form and plot_kernels
    explicit_kernel_mode = args.plot_mode in ("kernels", "width", "both")
    min_K = min(args.n_infer_iters)
    print(
        f"plot_mode={args.plot_mode} "
        f"(kernels={plot_kernels}, width={plot_width}); "
        f"overlay_any={overlay_any}; "
        f"skip_theory={skip_kernel_theory}; "
        f"skip_finite={args.skip_finite}; "
        f"skip_closed_form={skip_closed_form}; "
        f"finite widths={run_widths}"
    )
    phi_fn, _ = get_nonlinearity(args.act_fn, beta=args.nonlin_beta)
    kernel_align_records = []
    theory_cache = {}

    # Three independent children of --seed: dataset, weight init, DMFT Monte
    # Carlo. Previously theory used PRNGKey(args.seed) itself, which is the
    # parent of the data (and, with n_seeds=1, init) keys — JAX forbids
    # reusing a key, so the DMFT Gaussians were correlated with the samples.
    data_key, model_parent, theory_mc_key = jax.random.split(
        jax.random.PRNGKey(args.seed), 3
    )

    # Dataset is built once so every trial and the theory solve share Kx, y.
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

    Kx = jnp.asarray(X.T @ X / input_dim, dtype=jnp.float64)
    X_input = jnp.asarray(X.T, dtype=jnp.float64)
    Y_target = y[:, None] if y.ndim == 1 else y
    Y_target = jnp.asarray(Y_target, dtype=jnp.float64)
    loss_id = (
        "mse" if args.dataset in ("toy", "tiny-CIFAR10") else args.loss_id
    )

    n_seeds_run = 1 if (
        (plot_kernels and not plot_width) or args.skip_finite
    ) else args.n_seeds
    for seed in range(args.seed, args.seed + n_seeds_run):
        print(f"\nRunning experiment for seed: {seed}")

        set_seed(seed)
        model_key = jax.random.fold_in(model_parent, int(seed))

        theory_records = []
        finite_records = []

        for n_hidden in args.n_hiddens:
            print(f"\n\tn hidden H = {n_hidden}")

            for use_skips in args.use_skips:
                print(f"\n\t\tuse_skips = {use_skips}")

                for gamma_0 in args.gamma_0s:
                    print(f"\n\t\t\tgamma_0 = {gamma_0}")

                    for param_type in args.param_types:
                        print(f"\n\t\t\t\tparam_type = {param_type}")

                        width_keys = jax.random.split(
                            model_key, len(args.widths)
                        )
                        width_key_map = dict(zip(args.widths, width_keys))

                        for activity_lr in args.activity_lrs:
                            print(f"\n\t\t\t\t\tactivity_lr = {activity_lr}")

                            for K_inf in args.n_infer_iters:
                                print(f"\n\t\t\t\t\t\tn_infer_iters = {K_inf}")

                                T_train = args.n_train_iters
                                P = args.n_samples
                                # auto: kernel figures and DMFT only for
                                # the smallest K unless --plot_mode is an
                                # explicit kernels / width / both request.
                                kernels_this_K = plot_kernels and (
                                    explicit_kernel_mode or K_inf == min_K
                                )
                                width_this_K = plot_width and (
                                    explicit_kernel_mode or K_inf == min_K
                                )
                                solve_theory = False
                                if width_this_K:
                                    solve_theory = True
                                elif kernels_this_K and not skip_kernel_theory:
                                    solve_theory = True
                                elif (not skip_loss_theory) and (
                                    (not overlay_n_infer_iters)
                                    or K_inf == min_K
                                ):
                                    solve_theory = True

                                # --- Calculate theory (PC), cached across seeds ---
                                if not solve_theory:
                                    all_Ch = None
                                    all_Cdelta = None
                                    pc_dmft_loss = None
                                else:
                                    theory_key = (
                                        n_hidden,
                                        bool(use_skips),
                                        gamma_0,
                                        param_type,
                                        activity_lr,
                                        K_inf,
                                    )
                                    if theory_key in theory_cache:
                                        all_Ch, all_Cdelta, pc_dmft_loss = (
                                            theory_cache[theory_key]
                                        )
                                    else:
                                        n_pc = K_inf * T_train * P
                                        if use_nonlin_theory:
                                            print(
                                                "\t\t\t\t\tCalculating nonlinear PC Theory "
                                                f"(act_fn={args.act_fn}, "
                                                f"matrix size n = K*T*P = {n_pc})...\n"
                                            )
                                            (
                                                all_Ch,
                                                all_Cdelta,
                                                _all_Rh,
                                                _all_Rdelta,
                                                _C_delta_top,
                                                pc_dmft_loss,
                                                _mean_delta_top,
                                                pc_diagnostics,
                                            ) = solve_pc_kernels_nonlin(
                                                Kx=Kx,
                                                y=Y_target,
                                                depth=n_hidden,
                                                eta=args.param_lr_pc,
                                                gamma=gamma_0,
                                                beta_h=activity_lr,
                                                hidden_energy_scaling=n_hidden + 1,
                                                num_training_steps=T_train,
                                                num_inference_steps=K_inf,
                                                num_fixed_point_steps=args.n_fixed_point_steps,
                                                num_mc_samples=args.num_mc_samples,
                                                num_jacobian_samples=args.num_jacobian_samples,
                                                jacobian_batch_size=args.jacobian_batch_size,
                                                damping=args.pc_damping,
                                                nonlinearity=args.act_fn,
                                                beta=args.nonlin_beta,
                                                tolerance=args.pc_tolerance,
                                                key=theory_mc_key,
                                            )
                                        else:
                                            print(
                                                "\t\t\t\t\tCalculating PC Theory "
                                                f"(matrix size n = K*T*P = {n_pc})...\n"
                                            )
                                            (
                                                all_Ch,
                                                all_Cdelta,
                                                _all_Rh,
                                                _all_Rdelta,
                                                _C_delta_top,
                                                pc_dmft_loss,
                                                _mean_delta_top,
                                                pc_diagnostics,
                                            ) = solve_pc_kernels(
                                                Kx=Kx,
                                                y=Y_target,
                                                depth=n_hidden,
                                                eta=args.param_lr_pc,
                                                gamma=gamma_0,
                                                beta_h=activity_lr,
                                                hidden_energy_scaling=n_hidden + 1,
                                                num_training_steps=T_train,
                                                num_inference_steps=K_inf,
                                                num_fixed_point_steps=args.n_fixed_point_steps,
                                                damping=args.pc_damping,
                                                tolerance=args.pc_tolerance,
                                                backend=args.pc_backend,
                                            )
                                        print(
                                            "\t\t\t\t\tPC fixed-point residual = "
                                            f"{float(pc_diagnostics['fixed_point_residual']):.3e}, "
                                            "equation residual = "
                                            f"{float(pc_diagnostics['equation_residual']):.3e} "
                                            f"after {pc_diagnostics['iterations']} iters\n"
                                        )
                                        theory_cache[theory_key] = (
                                            all_Ch,
                                            all_Cdelta,
                                            pc_dmft_loss,
                                        )

                                sweep_meta = dict(
                                    n_hidden=n_hidden,
                                    gamma_0=gamma_0,
                                    activity_lr=activity_lr,
                                    n_infer_iters=K_inf,
                                    param_type=param_type,
                                    use_skips=use_skips,
                                )
                                if pc_dmft_loss is not None:
                                    theory_records.extend(
                                        _loss_records(
                                            pc_dmft_loss, **sweep_meta
                                        )
                                    )

                                if width_this_K:
                                    finite_widths = run_widths
                                elif overlay_any:
                                    finite_widths = [kernel_plot_width]
                                else:
                                    finite_widths = run_widths
                                collect_fields = kernels_this_K or width_this_K

                                # --- Finite-size PC simulation (infer) ---
                                finite_pc_records = []
                                infer_feature_kernels = None
                                if not args.skip_finite:
                                    print(
                                        "\t\t\t\t\tRunning finite-size PC simulation "
                                        f"for widths {finite_widths}...\n"
                                    )
                                    for width in finite_widths:
                                        print(
                                            "\t\t\t\t\tNumerical PC simulation "
                                            f"for width N = {width}"
                                        )
                                        losses, fields = _train_finite_pc(
                                            width_key_map[width],
                                            results_dir=args.results_dir,
                                            input_dim=input_dim,
                                            output_dim=output_dim,
                                            n_samples=args.n_samples,
                                            n_hidden=n_hidden,
                                            use_skips=use_skips,
                                            act_fn=args.act_fn,
                                            param_type=param_type,
                                            param_lr=args.param_lr_pc,
                                            gamma_0=gamma_0,
                                            param_optim_id=args.param_optim,
                                            n_train_iters=T_train,
                                            infer_mode="optim",
                                            n_infer_iters=K_inf,
                                            activity_lr=activity_lr,
                                            width=width,
                                            loss_id=loss_id,
                                            seed=seed,
                                            X_input=X_input,
                                            Y_target=Y_target,
                                            collect_fields=collect_fields,
                                        )
                                        recs = _loss_records(
                                            losses,
                                            width=width,
                                            infer_mode="infer",
                                            **sweep_meta,
                                        )
                                        finite_pc_records.extend(recs)
                                        finite_records.extend(recs)
                                        if width_this_K:
                                            print(
                                                "\t\t\t\t\tKernel alignment "
                                                f"for width N = {width}"
                                            )
                                            kernel_align_records.extend(
                                                _kernel_align_records(
                                                    fields,
                                                    all_Ch,
                                                    all_Cdelta,
                                                    phi_fn,
                                                    num_inference_steps=K_inf,
                                                    num_training_steps=T_train,
                                                    num_samples=P,
                                                    width=width,
                                                    seed=seed,
                                                    **sweep_meta,
                                                )
                                            )
                                        if (
                                            kernels_this_K
                                            and seed == kernel_grid_seed
                                            and width == kernel_plot_width
                                        ):
                                            infer_feature_kernels = (
                                                _final_feature_kernels(
                                                    fields, phi_fn
                                                )
                                            )
                                        del fields

                                plots_dir = os.path.join(
                                    args.results_dir, "plots"
                                )
                                if not overlay_any:
                                    plot_pc_theory_vs_finite_loss(
                                        pc_dmft_loss=(
                                            pc_dmft_loss
                                            if pc_dmft_loss is not None
                                            else jnp.zeros(T_train)
                                        ),
                                        finite_df=pd.DataFrame(
                                            finite_pc_records
                                        ),
                                        plots_dir=plots_dir,
                                        gamma_0=gamma_0,
                                        n_hidden=n_hidden,
                                        activity_lr=activity_lr,
                                        n_infer_iters=K_inf,
                                        update_mode="infer",
                                        skip_theory=skip_loss_theory
                                        or pc_dmft_loss is None,
                                        skip_finite=args.skip_finite,
                                    )

                                if (
                                    kernels_this_K
                                    and seed == kernel_grid_seed
                                ):
                                    closed_form_feature_kernels = None
                                    if (
                                        not use_nonlin_theory
                                        and not skip_closed_form
                                        and not args.skip_finite
                                    ):
                                        print(
                                            "\t\t\t\t\tNumerical PC simulation "
                                            "(closed-form) for width N = "
                                            f"{kernel_plot_width}"
                                        )
                                        _, fields_cf = _train_finite_pc(
                                            width_key_map[kernel_plot_width],
                                            results_dir=args.results_dir,
                                            input_dim=input_dim,
                                            output_dim=output_dim,
                                            n_samples=args.n_samples,
                                            n_hidden=n_hidden,
                                            use_skips=use_skips,
                                            act_fn=args.act_fn,
                                            param_type=param_type,
                                            param_lr=args.param_lr_pc,
                                            gamma_0=gamma_0,
                                            param_optim_id=args.param_optim,
                                            n_train_iters=T_train,
                                            infer_mode="closed_form",
                                            n_infer_iters=K_inf,
                                            activity_lr=activity_lr,
                                            width=kernel_plot_width,
                                            loss_id=loss_id,
                                            seed=seed,
                                            X_input=X_input,
                                            Y_target=Y_target,
                                        )
                                        closed_form_feature_kernels = (
                                            _final_feature_kernels(
                                                fields_cf, phi_fn
                                            )
                                        )
                                        del fields_cf

                                    kernel_rows = []
                                    if closed_form_feature_kernels is not None:
                                        kernel_rows.append(
                                            (
                                                "Closed-form",
                                                closed_form_feature_kernels,
                                            )
                                        )
                                    kernel_rows.append(
                                        ("Infer", infer_feature_kernels)
                                    )
                                    if all_Ch is not None:
                                        kernel_rows.append(
                                            (
                                                "DMFT",
                                                _final_dmft_feature_kernels(
                                                    all_Ch,
                                                    num_inference_steps=K_inf,
                                                    num_training_steps=T_train,
                                                    num_samples=P,
                                                ),
                                            )
                                        )
                                    plot_final_kernel_grid(
                                        kernel_rows,
                                        plots_dir=plots_dir,
                                        gamma_0=gamma_0,
                                        n_hidden=n_hidden,
                                        activity_lr=activity_lr,
                                        n_infer_iters=K_inf,
                                        width=kernel_plot_width,
                                        filename="final_pc_kernels_grid.png",
                                        share_clim=True,
                                    )

                            if (
                                overlay_n_infer_iters
                                and (not args.skip_finite)
                                and (not use_nonlin_theory)
                            ):
                                print(
                                    "\t\t\t\t\tRunning finite-size PC "
                                    "simulation (closed-form) for "
                                    f"width N = {kernel_plot_width}...\n"
                                )
                                closed_losses, _ = _train_finite_pc(
                                    width_key_map[kernel_plot_width],
                                    results_dir=args.results_dir,
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    n_samples=args.n_samples,
                                    n_hidden=n_hidden,
                                    use_skips=use_skips,
                                    act_fn=args.act_fn,
                                    param_type=param_type,
                                    param_lr=args.param_lr_pc,
                                    gamma_0=gamma_0,
                                    param_optim_id=args.param_optim,
                                    n_train_iters=args.n_train_iters,
                                    infer_mode="closed_form",
                                    n_infer_iters=min_K,
                                    activity_lr=activity_lr,
                                    width=kernel_plot_width,
                                    loss_id=loss_id,
                                    seed=seed,
                                    X_input=X_input,
                                    Y_target=Y_target,
                                    collect_fields=False,
                                )
                                finite_records.extend(
                                    _loss_records(
                                        closed_losses,
                                        n_hidden=n_hidden,
                                        gamma_0=gamma_0,
                                        activity_lr=activity_lr,
                                        n_infer_iters=min_K,
                                        param_type=param_type,
                                        use_skips=use_skips,
                                        width=kernel_plot_width,
                                        infer_mode="closed_form",
                                    )
                                )

        if overlay_any:
            theory_df = pd.DataFrame(theory_records)
            finite_df = pd.DataFrame(finite_records)
            plots_root = os.path.join(args.results_dir, "plots")
            sweep_kwargs = dict(
                theory_df=theory_df,
                finite_df=finite_df,
                plots_dir=plots_root,
                skip_theory=skip_loss_theory,
                skip_finite=args.skip_finite,
            )
            if overlay_n_hiddens:
                plot_pc_param_sweep_loss(
                    swept_col="n_hidden",
                    plot_closed_form=False,
                    **sweep_kwargs,
                )
            if overlay_gamma_0s:
                plot_pc_param_sweep_loss(
                    swept_col="gamma_0",
                    plot_closed_form=False,
                    **sweep_kwargs,
                )
            if overlay_n_infer_iters:
                plot_pc_param_sweep_loss(
                    swept_col="n_infer_iters",
                    plot_closed_form=(not use_nonlin_theory),
                    **sweep_kwargs,
                )

    if plot_width:
        if kernel_align_records:
            align_df = pd.DataFrame(kernel_align_records)
            group_cols = [
                "n_hidden",
                "gamma_0",
                "activity_lr",
                "n_infer_iters",
            ]
            plots_root = os.path.join(args.results_dir, "plots")
            for keys, sub in align_df.groupby(group_cols, dropna=False):
                n_hidden_k, gamma_0_k, activity_lr_k, K_inf_k = keys
                plot_pc_kernel_width_alignment(
                    align_df=sub,
                    plots_dir=plots_root,
                    n_hidden=n_hidden_k,
                    gamma_0=gamma_0_k,
                    activity_lr=activity_lr_k,
                    n_infer_iters=K_inf_k,
                )
        else:
            print("\nNo kernel-alignment records were collected.")

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

# Plot final feature kernels, finite-size only (no DMFT row)
# CUDA_VISIBLE_DEVICES=1 python analyse_convergence.py --n_samples 20 --n_hiddens 5 --widths 10000 --plot_mode kernels --skip_theory --gamma_0s 1.0 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 --n_train_iters 20 --n_fixed_point_steps 100 --pc_damping 0.05

# Across depth
# CUDA_VISIBLE_DEVICES=1 python analyse_convergence.py --n_samples 20 --n_hiddens 2 3 4 5  --widths 10000 --gamma_0s 1.0 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 --n_train_iters 20 --n_fixed_point_steps 100 --pc_damping 0.05

# Across gamma
# CUDA_VISIBLE_DEVICES=1 python analyse_convergence.py --n_samples 20 --n_hiddens 5 --widths 10000 --gamma_0s 0.1 0.5 1.0 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 --n_train_iters 20 --n_fixed_point_steps 100 --pc_damping 0.05

# Across K (DMFT only for smallest K; closed-form finite-size overlay)
# CUDA_VISIBLE_DEVICES=1 python analyse_convergence.py --n_samples 20 --n_hiddens 5 --widths 10000 --gamma_0s 1.0 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 20 50 200 500 --n_train_iters 20 --n_fixed_point_steps 100 --pc_damping 0.05

# Across widths (convergence of kernels + plot final kernels)
# CUDA_VISIBLE_DEVICES=1 python analyse_convergence.py --n_samples 20 --n_hiddens 5 --widths 10 25 100 250 1000 2500 10000 --plot_mode both --gamma_0s 1.0 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 --n_train_iters 20 --n_fixed_point_steps 500 --pc_damping 0.05 --pc_tolerance 1e-10 --n_seeds 3


############ NONLINEAR ##################
#########################################

# Plot final feature kernels, finite-size only (no DMFT row)
# CUDA_VISIBLE_DEVICES=1 python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10000 --plot_mode kernels --skip_theory --gamma_0s 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 --n_train_iters 30 --n_fixed_point_steps 100 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10

# Across depth
# CUDA_VISIBLE_DEVICES=1 python analyse_convergence.py --n_samples 8 --n_hiddens 1 2 3  --widths 10000 --gamma_0s 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 --n_train_iters 30 --n_fixed_point_steps 100 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10

# Across gamma
# CUDA_VISIBLE_DEVICES=1 python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10000 --gamma_0s 0.1 0.5 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 --n_train_iters 30 --n_fixed_point_steps 100 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10

# Across K (DMFT only for smallest K)
# CUDA_VISIBLE_DEVICES=1 python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10000 --gamma_0s 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 50 100 500 1000 --n_train_iters 30 --n_fixed_point_steps 100 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10

# Across widths (convergence of kernels + plot final kernels)
# CUDA_VISIBLE_DEVICES=1 python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10 25 100 250 1000 2500 10000 --plot_mode both --gamma_0s 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 --n_train_iters 30 --n_fixed_point_steps 100 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10 --n_seeds 3
