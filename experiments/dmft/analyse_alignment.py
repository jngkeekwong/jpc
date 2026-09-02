"""Kernel-alignment analysis for PC and BP DMFT theory.

Runs PC DMFT theory, then BP DMFT theory, and plots kernel displacement
and PC–BP feature-kernel alignment. Finite-width simulations are omitted.

``--seed`` draws three independent RNG streams (dataset, weight init,
PC DMFT Monte Carlo). BP Monte Carlo is folded in from the PC stream so
the two solvers do not share keys. Theory always uses the same ``(X, y)``
as would a finite-size run with this seed.

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
from experiments.dmft.utils import (
    CIFAR_GRAY_DIM,
    create_tiny_cifar10_dataset,
    create_toy_dataset,
    cosine_similarity,
    final_time_pc_kernel,
    bp_sample_kernel_at,
)
from theory_utils import solve_kernels, solve_kernels_nonlin, get_Delta, solve_Delta
from theory_pc_utils import solve_pc_kernels
from theory_pc_nonlin_utils import solve_pc_kernels_nonlin
from plot_dmft_results import (
    plot_dmft_kernels_and_loss,
    plot_pc_dmft_kernels_and_loss,
    plot_kernel_displacement,
    plot_pc_bp_kernel_alignment,
    plot_final_kernel_grid,
)


def _alignment_and_displacement_records(
    all_Ch,
    all_H,
    num_inference_steps,
    num_training_steps,
    num_samples,
    **meta,
):
    """Per-layer PC/BP feature-kernel displacement and PC-BP alignment.

    ``all_Ch`` are PC theory kernels (sample-sample block taken at
    ``k=0``); ``all_H`` are BP theory kernels, or ``None`` to skip BP.
    "Initial" is training step ``t=0`` (post-init, pre-update); "final"
    is the last training step.

    Returns ``(displacement_records, alignment_records, pc_final_kernels,
    bp_final_kernels)``, where the last two are lists of ``P x P`` arrays
    (one per layer) suitable for ``plot_final_kernel_grid``.
    """
    displacement_records = []
    alignment_records = []
    pc_final_kernels = []
    bp_final_kernels = [] if all_H is not None else None

    slice_kw = dict(
        num_inference_steps=num_inference_steps,
        num_training_steps=num_training_steps,
        num_samples=num_samples,
    )
    if all_H is not None and len(all_H) != len(all_Ch):
        raise ValueError(
            "PC and BP theory have a different number of layers: "
            f"PC={len(all_Ch)}, BP={len(all_H)}."
        )

    for l, Ch_l in enumerate(all_Ch):
        pc_init = final_time_pc_kernel(Ch_l, k=0, t=0, **slice_kw)
        pc_final = final_time_pc_kernel(Ch_l, k=0, t=-1, **slice_kw)
        pc_final_kernels.append(pc_final)
        displacement_records.append({
            **meta,
            "layer": l,
            "method": "pc",
            "displacement": float(
                cosine_similarity(pc_init, pc_final, eps=1e-30)
            ),
        })

        if all_H is not None:
            H_l = all_H[l]
            bp_init = bp_sample_kernel_at(H_l, t=0)
            bp_final = bp_sample_kernel_at(H_l, t=-1)
            bp_final_kernels.append(bp_final)
            displacement_records.append({
                **meta,
                "layer": l,
                "method": "bp",
                "displacement": float(
                    cosine_similarity(bp_init, bp_final, eps=1e-30)
                ),
            })
            alignment_records.append({
                **meta,
                "layer": l,
                "alignment": float(
                    cosine_similarity(pc_final, bp_final, eps=1e-30)
                ),
            })

    return (
        displacement_records,
        alignment_records,
        pc_final_kernels,
        bp_final_kernels,
    )


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
    parser.add_argument("--param_lr", type=float, default=0.05)
    parser.add_argument("--gamma_0s", type=float, nargs='+', default=[1])
    parser.add_argument("--n_train_iters", type=int, default=20) # 100)
    parser.add_argument("--n_fixed_point_steps", type=int, default=10)

    # Inference parameters
    parser.add_argument("--param_lr_pc", type=float, default=0.5)
    parser.add_argument("--n_infer_iters", type=int, nargs='+', default=[5])
    parser.add_argument("--activity_lrs", type=float, nargs='+', default=[0.05])

    # Loop parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--n_hiddens", type=int, nargs='+', default=[5])

    # DMFT theory parameters (shared by BP and PC)
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
        help="Monte-Carlo samples for nonlinear BP/PC DMFT theory.",
    )

    # BP DMFT parameters
    parser.add_argument(
        "--bp_damping",
        type=float,
        default=1.0,
        help="Kernel mixing factor for nonlinear BP DMFT fixed-point updates.",
    )
    parser.add_argument(
        "--skip_theory_bp",
        action="store_true",
        default=False,
        help="Skip BP DMFT theory (kernels and loss).",
    )

    # PC DMFT parameters
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
        help="Batch size for nonlinear PC Jacobian samples. Batch size fixed at 50 for BP",
    )
    args = parser.parse_args()

    # PC DMFT inverts (K*T*P) matrices; float64 helps stability.
    jax.config.update("jax_enable_x64", True)

    os.makedirs(args.results_dir, exist_ok=True)
    use_nonlin_theory = args.act_fn != "linear"
    if use_nonlin_theory and args.act_fn == "relu" and not args.skip_theory_bp:
        raise ValueError(
            "Nonlinear BP DMFT (solve_kernels_nonlin) supports only "
            "'tanh' (and softplus in the solver API). Use --act_fn tanh "
            "or --skip_theory_bp."
        )

    data_key, _model_parent, pc_theory_key = jax.random.split(
        jax.random.PRNGKey(args.seed), 3
    )
    bp_theory_key = jax.random.fold_in(pc_theory_key, 1)

    if args.dataset == "toy":
        X, y = create_toy_dataset(
            key=data_key, D=args.input_dim, P=args.n_samples
        )
        input_dim = args.input_dim
    elif args.dataset == "tiny-CIFAR10":
        input_dim = CIFAR_GRAY_DIM
        X, y = create_tiny_cifar10_dataset(
            key=data_key, D=input_dim, P=args.n_samples
        )
        print(f"Input dim: {input_dim}")
    else:
        from torch import Generator as TorchGenerator

        loader_gen = TorchGenerator()
        loader_gen.manual_seed(int(np.asarray(data_key)[0]) & 0x7FFFFFFF)
        train_loader, _ = get_dataloaders(
            args.dataset, args.n_samples, generator=loader_gen
        )
        img_batch, label_batch = next(iter(train_loader))

        input_dim = img_batch.shape[1]
        print(f"Input dim: {input_dim}, Output dim: {label_batch.shape[1]}")

        X = img_batch.numpy().T
        y = label_batch.numpy()

    Kx = jnp.asarray(X.T @ X / input_dim, dtype=jnp.float64)
    Y_target = y[:, None] if y.ndim == 1 else y
    Y_target = jnp.asarray(Y_target, dtype=jnp.float64)
    y_bp = (
        jnp.squeeze(Y_target, axis=-1)
        if Y_target.ndim == 2 and Y_target.shape[-1] == 1
        else jnp.asarray(y)
    )
    # BP theory does not depend on activity_lr or n_infer_iters; PC theory
    # is cached across --n_seeds since the dataset is fixed.
    bp_theory_cache = {}
    pc_theory_cache = {}

    for seed in range(args.seed, args.seed + args.n_seeds):
        print(f"\nRunning experiment for seed: {seed}")

        set_seed(seed)

        displacement_records = []
        pcbp_alignment_records = []

        for n_hidden in args.n_hiddens:
            print(f"\n\tn hidden H = {n_hidden}")

            for use_skips in args.use_skips:
                print(f"\n\t\tuse_skips = {use_skips}")

                for gamma_0 in args.gamma_0s:
                    print(f"\n\t\t\tgamma_0 = {gamma_0}")

                    for param_type in args.param_types:
                        print(f"\n\t\t\t\tparam_type = {param_type}")

                        for activity_lr in args.activity_lrs:
                            print(f"\n\t\t\t\t\tactivity_lr = {activity_lr}")

                            for K_inf in args.n_infer_iters:
                                print(
                                    "\n\t\t\t\t\t\tn_infer_iters = "
                                    f"{K_inf}"
                                )

                                T_train = args.n_train_iters
                                P = args.n_samples
                                plots_dir = os.path.join(
                                    args.results_dir, "plots"
                                )

                                # --- Calculate theory (PC), cached across seeds ---
                                n_pc = K_inf * T_train * P
                                pc_key = (
                                    n_hidden,
                                    bool(use_skips),
                                    gamma_0,
                                    param_type,
                                    activity_lr,
                                    K_inf,
                                )
                                if pc_key in pc_theory_cache:
                                    all_Ch, all_Cdelta, pc_dmft_loss = (
                                        pc_theory_cache[pc_key]
                                    )
                                else:
                                    if use_nonlin_theory:
                                        print(
                                            "\t\t\t\t\t\tCalculating nonlinear PC "
                                            f"Theory (act_fn={args.act_fn}, "
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
                                            key=pc_theory_key,
                                        )
                                    else:
                                        print(
                                            "\t\t\t\t\t\tCalculating PC Theory "
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
                                        "\t\t\t\t\t\tPC fixed-point residual = "
                                        f"{float(pc_diagnostics['fixed_point_residual']):.3e}, "
                                        "equation residual = "
                                        f"{float(pc_diagnostics['equation_residual']):.3e} "
                                        f"after {pc_diagnostics['iterations']} iters\n"
                                    )
                                    plot_pc_dmft_kernels_and_loss(
                                        all_Ch=all_Ch,
                                        all_Cdelta=all_Cdelta,
                                        pc_dmft_loss=pc_dmft_loss,
                                        plots_dir=plots_dir,
                                        num_inference_steps=K_inf,
                                        num_training_steps=T_train,
                                        num_samples=P,
                                        gamma_0=gamma_0,
                                        n_hidden=n_hidden,
                                        activity_lr=activity_lr,
                                    )
                                    pc_theory_cache[pc_key] = (
                                        all_Ch,
                                        all_Cdelta,
                                        pc_dmft_loss,
                                    )

                                # --- Calculate theory (BP), cached since it
                                # does not depend on activity_lr / K_inf ---
                                all_H = None
                                if not args.skip_theory_bp:
                                    bp_key = (
                                        n_hidden,
                                        bool(use_skips),
                                        gamma_0,
                                        param_type,
                                    )
                                    if bp_key in bp_theory_cache:
                                        all_H, all_G, dmft_loss = (
                                            bp_theory_cache[bp_key]
                                        )
                                    else:
                                        if use_nonlin_theory:
                                            print(
                                                "\t\t\t\t\t\tCalculating "
                                                "nonlinear BP Theory "
                                                f"(act_fn={args.act_fn})...\n"
                                            )
                                            all_H, all_G, _, _ = (
                                                solve_kernels_nonlin(
                                                    Kx=Kx,
                                                    y=y_bp,
                                                    depth=n_hidden,
                                                    eta=args.param_lr,
                                                    gamma=gamma_0,
                                                    T=args.n_train_iters,
                                                    num_iter=args.n_fixed_point_steps,
                                                    samples=args.num_mc_samples,
                                                    damping=args.bp_damping,
                                                    nonlin=args.act_fn,
                                                    beta=args.nonlin_beta,
                                                    key=bp_theory_key,
                                                )
                                            )
                                            Delta_theory = solve_Delta(
                                                Kx=Kx,
                                                y=y_bp,
                                                all_Phi=all_H,
                                                all_G=all_G,
                                                eta=args.param_lr,
                                            )
                                            dmft_loss = 0.5 * jnp.mean(
                                                Delta_theory**2, axis=1
                                            )
                                        else:
                                            print(
                                                "\t\t\t\t\t\tCalculating BP "
                                                "Theory...\n"
                                            )
                                            all_H, all_G, _, _ = solve_kernels(
                                                Kx=Kx,
                                                y=y_bp,
                                                depth=n_hidden,
                                                eta=args.param_lr,
                                                gamma=gamma_0,
                                                T=args.n_train_iters,
                                                num_steps=args.n_fixed_point_steps
                                            )
                                            Delta_theory = get_Delta(
                                                all_H=all_H,
                                                all_G=all_G,
                                                Kx=Kx,
                                                y=y_bp,
                                                eta=args.param_lr
                                            )
                                            dmft_loss = 0.5 * jnp.mean(
                                                jnp.sum(
                                                    Delta_theory**2, axis=2
                                                ),
                                                axis=1,
                                            )

                                        plot_dmft_kernels_and_loss(
                                            all_H=all_H,
                                            all_G=all_G,
                                            dmft_loss=dmft_loss,
                                            plots_dir=plots_dir,
                                            gamma_0=gamma_0,
                                            n_hidden=n_hidden,
                                        )
                                        bp_theory_cache[bp_key] = (
                                            all_H, all_G, dmft_loss
                                        )

                                # --- Kernel displacement / PC-BP alignment ---
                                sweep_meta = dict(
                                    n_hidden=n_hidden,
                                    use_skips=use_skips,
                                    gamma_0=gamma_0,
                                    param_type=param_type,
                                    activity_lr=activity_lr,
                                    n_infer_iters=K_inf,
                                )
                                (
                                    disp_recs,
                                    align_recs,
                                    pc_final_kernels,
                                    bp_final_kernels,
                                ) = _alignment_and_displacement_records(
                                    all_Ch=all_Ch,
                                    all_H=all_H,
                                    num_inference_steps=K_inf,
                                    num_training_steps=T_train,
                                    num_samples=P,
                                    **sweep_meta,
                                )
                                displacement_records.extend(disp_recs)
                                pcbp_alignment_records.extend(align_recs)

                                kernel_rows = [("PC", pc_final_kernels)]
                                if bp_final_kernels is not None:
                                    kernel_rows.append(
                                        ("Backprop", bp_final_kernels)
                                    )
                                plot_final_kernel_grid(
                                    kernel_rows,
                                    plots_dir=plots_dir,
                                    gamma_0=gamma_0,
                                    n_hidden=n_hidden,
                                    activity_lr=activity_lr,
                                    n_infer_iters=K_inf,
                                )

        # --- Aggregate displacement / PC-BP alignment plots ---
        # Grouped so several gamma_0s and/or n_infer_iters values sweep
        # together on the same axes.
        plots_dir = os.path.join(args.results_dir, "plots")
        group_cols = ["n_hidden", "use_skips", "param_type", "activity_lr"]
        if displacement_records:
            disp_df = pd.DataFrame(displacement_records)
            for keys, group in disp_df.groupby(group_cols, dropna=False):
                n_hidden_k, _use_skips_k, _param_type_k, activity_lr_k = keys
                plot_kernel_displacement(
                    group,
                    plots_dir=plots_dir,
                    n_hidden=n_hidden_k,
                    activity_lr=activity_lr_k,
                )
        if pcbp_alignment_records:
            align_df = pd.DataFrame(pcbp_alignment_records)
            for keys, group in align_df.groupby(group_cols, dropna=False):
                n_hidden_k, _use_skips_k, _param_type_k, activity_lr_k = keys
                plot_pc_bp_kernel_alignment(
                    group,
                    plots_dir=plots_dir,
                    n_hidden=n_hidden_k,
                    activity_lr=activity_lr_k,
                )


######### LINEAR ##########
###########################

# Displacement (PC vs BP overlay) + PC–BP alignment + final kernel grid
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 20 --n_hiddens 5 --gamma_0s 1.0 --param_lr 0.1 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 --n_train_iters 20 --n_fixed_point_steps 100 --pc_damping 0.05

# Displacement / PC–BP alignment across gamma (PC curves only on displacement plot)
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 20 --n_hiddens 5 --gamma_0s 0.1 0.5 1.0 --param_lr 0.1 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 --n_train_iters 20 --n_fixed_point_steps 100 --pc_damping 0.05

# Displacement / PC–BP alignment across K (PC curves only on displacement plot)
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 20 --n_hiddens 5 --gamma_0s 1.0 --param_lr 0.1 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 10 15 20 --n_train_iters 20 --n_fixed_point_steps 100 --pc_damping 0.05


############ NONLINEAR ##################
#########################################

# Displacement (PC vs BP overlay) + PC–BP alignment + final kernel grid
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 10 --n_hiddens 3 --gamma_0s 1.0 --param_lr 0.5 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 --n_train_iters 20 --n_fixed_point_steps 100 --bp_damping 0.5 --pc_damping 0.05 --act_fn tanh --num_mc_samples 1000

# Displacement / PC–BP alignment across gamma (PC curves only on displacement plot)
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 10 --n_hiddens 3 --gamma_0s 0.1 0.5 1.0 --param_lr 0.5 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 --n_train_iters 20 --n_fixed_point_steps 100 --bp_damping 0.5 --pc_damping 0.05 --act_fn tanh --num_mc_samples 1000

# Displacement / PC–BP alignment across K (PC curves only on displacement plot)
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 10 --n_hiddens 3 --gamma_0s 1.0 --param_lr 0.5 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 10 15 20 --n_train_iters 20 --n_fixed_point_steps 100 --bp_damping 0.5 --pc_damping 0.05 --act_fn tanh --num_mc_samples 1000
