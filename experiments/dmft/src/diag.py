"""CIFAR CNN diagnostics: delta-energy check and a step-by-step training trace.

Shared setup matches ``train_benchmark.py`` (CIFAR-10, width 128, three residual
blocks, batch 64, ReLU, spatial-mean hidden energy). Flags after the subcommand
override those defaults.

From ``experiments/dmft``:

    JAX_ENABLE_X64=1 python src/diag.py check --width 32

    python src/diag.py blowup --param_optim adam --param_lr 0.3 --param_lr_pc 0.3 \
        --collapse_pools --activity_update downward --n_infer_iters 4 --activity_lr 0.02

``check`` compares ``pc_energy_fn`` with and without delta activities. With
``JAX_ENABLE_X64=1`` the two must agree in value, activity gradient, and weight
gradient. Without it, the float32 delta gradients are compared with the standard
form; the mismatch should shrink in proportion to the displacement.

``blowup`` trains until the weights become non-finite. It logs per-layer weight
RMS, the relative update, activity displacement, and the cosine of the PC
gradient with the backprop gradient. ``--bp`` trains with backprop on the same
learning-rate tree. ``--probe_ks`` reprints that cosine at other inference
lengths. ``--curv_every N`` prints ``activity_lr × s_l λmax`` on each map.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

_DMFT = Path(__file__).resolve().parents[1]
if str(_DMFT) not in sys.path:
    sys.path.insert(0, str(_DMFT))

import train_benchmark as tb  # noqa: E402
from src import cnn_pc  # noqa: E402


def parse():
    defaults = [
        "--dataset", "CIFAR10", "--arch", "cnn", "--width", "128",
        "--n_res_blocks", "3", "--batch_size", "64",
        "--param_optim", "sgd_momentum", "--act_fn", "relu",
        "--hidden_energy_reduce", "spatial_mean",
        "--activity_lr", "0.05", "--param_lr", "0.01",
    ]
    # Defaults first so a flag on the command line overrides them.
    sys.argv = [sys.argv[0]] + defaults + sys.argv[1:]
    args = tb.parse_args()
    args.dataset = tb.normalize_dataset_id(args.dataset)
    tb._scalarize_sweep_fields(args)
    return args


def setup(args):
    spec = tb.DATASET_SPECS[args.dataset]
    key = jax.random.PRNGKey(args.seed)
    _, energy_l = cnn_pc.resolve_cnn_ls(args)
    lam = tb.get_output_energy_scaling(args.param_type, args.gamma, args.width, energy_l)
    kappa = tb.get_hidden_energy_scaling(args.param_type, energy_l)
    tb.prepare_data(args)
    pc_model, bp_model, _ = tb.make_models(key, args, spec)
    collapse = bool(args.collapse_pools)
    hidden_fn = (cnn_pc.cnn_hidden_energy_scales_collapsed if collapse
                 else cnn_pc.cnn_hidden_energy_scales)
    kap = hidden_fn(pc_model, kappa, args.hidden_energy_layers)
    hw_r = None
    if args.hidden_energy_reduce == "spatial_mean":
        hw = cnn_pc.hidden_spatial_hw(pc_model, spec["in_channels"],
                                      spec["input_size"], collapse)
        hw_r = cnn_pc.readout_spatial_hw(pc_model, args.width)
        kap = kappa * (hw_r / hw)
        lam = lam * hw_r
    n = int(kap.shape[0])
    gain = args.hidden_kappa_ratio ** jnp.arange(n - 1, -1, -1, dtype=jnp.float32)
    top_kappa = float(getattr(args, "top_kappa_mult", 1.0))
    if top_kappa != 1.0:
        gain = gain.at[-1].multiply(top_kappa)
    args._kappa_depth_gain = gain
    kap = kap * gain
    return pc_model, bp_model, float(lam), kap, hw_r


def flat(t):
    return jnp.concatenate([jnp.ravel(a) for a in jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_array))])


def rel(a, b):
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-300))


def rms_tree(t):
    leaves = jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_array))
    if not leaves:
        return None
    return float(jnp.sqrt(sum(jnp.sum(a ** 2) for a in leaves) / sum(a.size for a in leaves)))


def check():
    args = parse()
    model, _, lam, kap, _ = setup(args)
    x, y = next(tb.iter_train_batches(args, 1))
    x, y = jnp.asarray(x, dtype=float), jnp.asarray(y, dtype=float)
    base = list(tb.jpc.init_activities_with_ffwd(model=model, input=x, skip_model=None, param_type="sp"))
    kw = dict(x=x, loss=args.loss_id, output_energy_scaling=lam, hidden_energy_scaling=kap)
    params = (model, None)
    std = lambda p, a: cnn_pc.pc_energy_fn(p, a, y, **kw)
    dlt = lambda p, a: cnn_pc.pc_energy_fn(p, a, y, base=base, **kw)
    dlt_w = lambda p, a: cnn_pc.pc_energy_fn(p, a, y, base=base, with_param_grads=True, **kw)
    print("dtype", base[0].dtype)

    # The standard energy recomputes f(base) and carries its rounding into e; the
    # mismatch should shrink in proportion to the displacement size.
    keys = jax.random.split(jax.random.PRNGKey(0), len(base))
    for scale in (1e-1, 1e-3):
        deltas = [scale * jax.random.normal(k, b.shape, dtype=b.dtype) for k, b in zip(keys, base)]
        z = [b + d for b, d in zip(base, deltas)]
        print(f"-- delta scale {scale:g}")
        print("energy std / delta / delta_w:", float(std(params, z)), float(dlt(params, deltas)),
              float(dlt_w(params, deltas)))
        ga_s = jax.grad(std, argnums=1)(params, z)
        ga_d = jax.grad(dlt, argnums=1)(params, deltas)
        print("activity grad rel err per map:",
              " ".join(f"{rel(a, b):.1e}" for a, b in zip(ga_d[:-1], ga_s[:-1])))
        gw_s = eqx.filter_grad(std)(params, z)[0]
        gw_d = eqx.filter_grad(dlt_w)(params, deltas)[0]
        print("weight grad rel err per layer:", " ".join(
            f"{rel(flat(a), flat(b)):.1e}" for a, b in zip(gw_d.layers, gw_s.layers)
            if jax.tree_util.tree_leaves(eqx.filter(a, eqx.is_array))))


def blowup():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--steps", type=int, default=1500)
    pre.add_argument("--every", type=int, default=25)
    pre.add_argument("--bp", action="store_true", help="train with BP instead (same lr tree)")
    pre.add_argument("--curv_every", type=int, default=0,
                     help="every N steps print activity_lr × s_l λmax(H_ll) per map")
    pre.add_argument("--probe_ks", type=int, nargs="*", default=[],
                     help="at logged steps, also print cos(g_pc, g_bp) for these K")
    own, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = parse()
    pc_model, _, lam, kap, hw_r = setup(args)
    args._readout_spatial_hw = hw_r
    jpc_kw = tb.pc_jpc_kwargs(args)
    _, energy_l = cnn_pc.resolve_cnn_ls(args)
    fwd_l, _ = cnn_pc.resolve_cnn_ls(args)
    optim, opt_state = tb.make_pc_param_optim(pc_model, None, args, fwd_l)
    act_optim = optax.sgd(args.activity_lr * args.batch_size)
    collapse = bool(args.collapse_pools)
    print(f"λ={lam:g} optim={args.param_optim} lr_pc={args.param_lr_pc} "
          f"activity_lr={args.activity_lr} K={args.n_infer_iters} collapse={collapse}")

    step_scales = None

    def map_curvatures(model, x, y, iters=20):
        nonlocal step_scales
        acts = (cnn_pc.init_activities_collapse_pools(model, x) if collapse else
                list(tb.jpc.init_activities_with_ffwd(model=model, input=x, skip_model=None,
                                                      param_type="sp")))
        if step_scales is None:
            s = jnp.asarray([cnn_pc.activity_hw(z) / hw_r for z in acts[:-1]] + [1.0])
            gain = getattr(args, "_kappa_depth_gain", None)
            if gain is not None:
                s = s / jnp.append(gain, 1.0)
            m = float(getattr(args, "top_activity_lr_mult", 1.0))
            s = s.at[len(acts) - 2].multiply(m)
            pm = float(getattr(args, "pool_activity_lr_mult", 1.0))
            if not collapse and pm != 1.0:
                for i, layer in enumerate(model.layers[: len(acts) - 1]):
                    if not jax.tree_util.tree_leaves(eqx.filter(layer, eqx.is_array)):
                        s = s.at[i].multiply(pm)
            step_scales = s
        B = x.shape[0]
        grad_fn = jax.grad(lambda a: B * cnn_pc.pc_energy_fn(
            (model, None), a, y, x=x, loss=args.loss_id, output_energy_scaling=lam,
            hidden_energy_scaling=kap, collapse_pools=collapse))
        out = []
        for l in range(len(acts) - 1):
            v = jax.random.normal(jax.random.PRNGKey(l), acts[l].shape)
            v = v / jnp.linalg.norm(v)
            ev = 0.0
            for _ in range(iters):
                tang = [jnp.zeros_like(a) for a in acts]
                tang[l] = v
                hv = jax.jvp(grad_fn, (acts,), (tang,))[1][l]
                ev = float(jnp.vdot(v, hv))
                v = hv / (jnp.linalg.norm(hv) + 1e-30)
            out.append(args.activity_lr * float(step_scales[l]) * ev)
        return out

    def capture():
        # Zero update; the state keeps the raw gradient. Reading it as
        # (W + g) - W loses low-layer gradients (~1e-9 of W) to float32 rounding.
        # update_pc_params calls optim.update(updates=..., state=..., params=...).
        def init_fn(params):
            return jax.tree_util.tree_map(jnp.zeros_like, params)

        def update_fn(updates, state, params=None):
            del state, params
            return jax.tree_util.tree_map(jnp.zeros_like, updates), updates

        return optax.GradientTransformation(init_fn, update_fn)

    def pc_grad(m, x, y):
        cap = capture()
        state = cap.init((eqx.filter(m, eqx.is_array), None))
        _, _, state, _, _, _ = tb.pc_infer_and_update(
            m, None, x, y, act_optim, cap, state, args, jpc_kw, lam, kap)
        return state[0]

    model = pc_model
    if own.bp:
        bp_optim, opt_state = tb.make_bp_param_optim(model, args, fwd_l)
        bp_step = tb.make_bp_step(args.loss_id)
    epoch = 1
    batches = tb.iter_train_batches(args, epoch)
    names = [type(l).__name__[:6] for l in model.layers]
    for t in range(own.steps):
        try:
            x, y = next(batches)
        except StopIteration:
            epoch += 1
            batches = tb.iter_train_batches(args, epoch)
            x, y = next(batches)
        # float64 when run with JAX_ENABLE_X64=1 (precision check), else float32.
        x, y = jnp.asarray(x, dtype=float), jnp.asarray(y, dtype=float)
        old = model
        if own.bp:
            model, opt_state = bp_step(model, opt_state, bp_optim, x, y)
            energy, ok, deltas = float("nan"), True, None
        else:
            model, _, opt_state, energy, ok, deltas = tb.pc_infer_and_update(
                model, None, x, y, act_optim, optim, opt_state, args, jpc_kw, lam, kap)
        finite = all(bool(jnp.all(jnp.isfinite(a)))
                     for a in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
        if t % own.every == 0 or not finite or not ok:
            logits = jax.vmap(old)(x)
            ce = float(tb.supervised_loss(logits, y, args.loss_id))
            w = [rms_tree(l) for l in old.layers]
            dw = [None if a is None else rms_tree(jax.tree_util.tree_map(
                lambda p, q: p - q, eqx.filter(l1, eqx.is_array), eqx.filter(l0, eqx.is_array)))
                for a, l1, l0 in zip(w, model.layers, old.layers)]
            print(f"step {t:5d} E={energy:.4g} λCE_ff={lam*ce:.4g} CE_ff={ce:.3f} "
                  f"|logit|={float(jnp.sqrt(jnp.mean(logits**2))):.3g} ok={ok}")
            print("   W rms : " + " ".join(f"{n}:{v:.3g}" for n, v in zip(names, w) if v is not None))
            print("   ΔW/W  : " + " ".join(f"{n}:{d/v:.2e}" for n, v, d in zip(names, w, dw)
                                         if v is not None and d is not None))
            if deltas is not None:
                print("   Δz rms: " + " ".join(f"{v:.3g}" for v in deltas))
            g_bp = eqx.filter_grad(
                lambda m: tb.supervised_loss(jax.vmap(m)(x), y, args.loss_id))(old)
            if own.bp:
                g_pc = g_bp
            else:
                g_pc = pc_grad(old, x, y)
            cos_bp, grow = [], []
            for gl_pc, gl_bp, wl, n in zip(g_pc.layers, g_bp.layers, old.layers, names):
                a = jax.tree_util.tree_leaves(eqx.filter(gl_pc, eqx.is_array))
                if not a:
                    continue
                a = jnp.concatenate([jnp.ravel(v) for v in a])
                b = jnp.concatenate([jnp.ravel(v) for v in
                                     jax.tree_util.tree_leaves(eqx.filter(gl_bp, eqx.is_array))])
                wv = jnp.concatenate([jnp.ravel(v) for v in
                                      jax.tree_util.tree_leaves(eqx.filter(wl, eqx.is_array))])
                c = lambda u, v: float(jnp.dot(u, v) / (jnp.linalg.norm(u) * jnp.linalg.norm(v) + 1e-30))
                cos_bp.append(f"{n}:{c(a, b):.2f}")
                grow.append(f"{n}:{c(-a, wv):+.3f}")
            print("   cos(g, g_bp): " + " ".join(cos_bp))
            print("   cos(-g, W)  : " + " ".join(grow))
            k_run = args.n_infer_iters
            for K in own.probe_ks:
                args.n_infer_iters = K
                gk = pc_grad(old, x, y)
                row = []
                for gl_pc, gl_bp, n in zip(gk.layers, g_bp.layers, names):
                    a = jax.tree_util.tree_leaves(eqx.filter(gl_pc, eqx.is_array))
                    if not a:
                        continue
                    a = jnp.concatenate([jnp.ravel(v) for v in a])
                    b = jnp.concatenate([jnp.ravel(v) for v in
                                         jax.tree_util.tree_leaves(eqx.filter(gl_bp, eqx.is_array))])
                    row.append(f"{n}:{float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-30)):.2f}")
                print(f"   K={K:3d} cos(g, g_bp): " + " ".join(row))
            args.n_infer_iters = k_run
        if own.curv_every and t % own.curv_every == 0:
            print(f"   step {t} activity_lr·sλmax per map: "
                  + " ".join(f"{v:.2f}" for v in map_curvatures(model, x, y)))
        if not finite:
            print(f"non-finite weights after step {t}")
            break


def main():
    usage = "usage: python src/diag.py {check,blowup} [args...]"
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print(usage)
        return
    command = sys.argv[1]
    if command not in ("check", "blowup"):
        print(usage, file=sys.stderr)
        raise SystemExit(2)
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if command == "check":
        check()
    else:
        blowup()


if __name__ == "__main__":
    main()
