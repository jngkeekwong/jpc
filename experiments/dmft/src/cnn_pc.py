"""CNN-only PC helpers: two depth counts and per-layer hidden precision.

MLP training still uses ``jpc`` with a scalar ``κ = L``. This module is used
only for the ResNet path in ``train_benchmark.py``, so ``jpc`` and
``limits_paper`` stay unchanged.

``L_fwd`` is the residual µP depth (``1/√L`` on ``ResNetBlock``s, Adam
residual LRs). It is realised by passing
``additive_depth_factor = L_fwd - n_res_blocks`` into the existing
``ResNet`` constructor.

``L_energy`` is the PC–BP matching depth: ``λ = γ² N L_energy`` and
``κ = L_energy`` on selected hidden layers.

``--collapse_pools`` folds each AvgPool into the next weight layer. Without
it, pool maps stay free activities. ``--hidden_energy_reduce spatial_mean``
matches the MLP on a convolutional readout: ``λ = γ² N L (H_r W_r)`` and
``κ_ℓ = L (H_r W_r) / (H_ℓ W_ℓ)`` on every hidden map, including free pools.
The readout multiplies by ``1/(γ N H_r W_r)``, so ``λ`` times that scale is
``γ L``, the same cancellation as the MLP's ``λ · 1/(γ N)``. The activity
step multiplies map ``ℓ`` by ``(H_ℓ W_ℓ) / (H_r W_r)``, so the effective
spring stays ``L``. ``update_pc_activities`` steps every free map from the
same activities. ``update_pc_activities_downward`` takes that same step one
map at a time, from the readout end down to the first convolution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, ArrayLike, PyTree, Scalar
from typing import Callable, Optional, Sequence, Tuple, Union

_CNN_DIR = Path(__file__).resolve().parents[2] / "limits_paper" / "cnn"
if str(_CNN_DIR) not in sys.path:
    sys.path.insert(0, str(_CNN_DIR))

from model import ResNetBlock, ScaledConv2d  # noqa: E402

RESNET_L_PRESETS = ("n_res_blocks", "n_weight_layers", "n_modules")
HIDDEN_ENERGY_LAYER_CHOICES = ("weight", "all", "residual")

ResnetLSpec = Union[str, int]


def parse_resnet_l_arg(value: str) -> ResnetLSpec:
    """Argparse type: preset name or a positive integer."""
    key = str(value).strip()
    if key in RESNET_L_PRESETS:
        return key
    try:
        parsed = int(key)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid ResNet L '{value}'. Use {', '.join(RESNET_L_PRESETS)} "
            "or a positive integer."
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"ResNet L must be a positive integer, got {parsed}."
        )
    return parsed


def _as_int(value) -> int:
    if isinstance(value, (list, tuple)):
        if len(value) < 1:
            raise ValueError("Expected a non-empty list of ints.")
        value = value[0]
    return int(value)


def resolve_resnet_l(spec: ResnetLSpec, n_res_blocks: int) -> int:
    """Map a preset or int to a concrete depth."""
    n = _as_int(n_res_blocks)
    if isinstance(spec, bool):
        raise ValueError("ResNet L spec cannot be a bool.")
    if isinstance(spec, int):
        val = int(spec)
    elif spec == "n_res_blocks":
        val = n
    elif spec == "n_weight_layers":
        val = n + 4
    elif spec == "n_modules":
        val = n + 7
    else:
        raise ValueError(
            f"Unknown ResNet L spec '{spec}'. "
            f"Use {', '.join(RESNET_L_PRESETS)} or a positive int."
        )
    if val <= 0:
        raise ValueError(f"ResNet L must be positive, got {val}.")
    return val


def resolve_cnn_ls(args) -> Tuple[int, int]:
    """Return ``(L_fwd, L_energy)`` from CNN CLI flags.

    ``--additive_depth_factor``, when set, overrides ``--resnet_energy_l``
    with ``n_res_blocks + factor``.
    """
    n = _as_int(args.n_res_blocks)
    fwd_l = resolve_resnet_l(
        getattr(args, "resnet_fwd_l", "n_res_blocks"), n
    )
    if getattr(args, "additive_depth_factor", None) is not None:
        energy_l = n + _as_int(args.additive_depth_factor)
        if energy_l <= 0:
            raise ValueError(
                f"n_res_blocks + additive_depth_factor must be positive, "
                f"got {energy_l}."
            )
    else:
        energy_l = resolve_resnet_l(
            getattr(args, "resnet_energy_l", "n_weight_layers"), n
        )
    return fwd_l, energy_l


def resnet_fwd_additive_depth_factor(fwd_l: int, n_res_blocks: int) -> int:
    """Value to pass to ``ResNet(..., additive_depth_factor=...)``.

    ``ResNet`` uses ``depth = n_res_blocks + additive_depth_factor`` for
    residual ``1/√L``. Choosing ``additive = L_fwd - n_res_blocks`` makes
    that depth equal ``L_fwd`` without changing ``limits_paper``.
    """
    return _as_int(fwd_l) - _as_int(n_res_blocks)


def _hidden_layer_kind(layer) -> str:
    if isinstance(layer, nn.AvgPool2d):
        return "pool"
    if isinstance(layer, ResNetBlock):
        return "residual"
    if isinstance(layer, ScaledConv2d):
        return "stem"
    return "other"


def cnn_hidden_energy_scales(
    model,
    kappa: float,
    hidden_energy_layers: str,
) -> jnp.ndarray:
    """Per-hidden-layer κ for a ResNet (length ``len(model) - 1``).

    The readout is the output term (scaled by ``λ``, not this vector).

    * ``all``: every hidden module, including pools, gets ``κ``.
    * ``weight`` (default): stems and residual blocks get ``κ``; pools get 1.
    * ``residual``: only ``ResNetBlock``s get ``κ``; stems and pools get 1.
    """
    if hidden_energy_layers not in HIDDEN_ENERGY_LAYER_CHOICES:
        raise ValueError(
            f"Unknown --hidden_energy_layers '{hidden_energy_layers}'. "
            f"Use {', '.join(HIDDEN_ENERGY_LAYER_CHOICES)}."
        )
    kappa = float(kappa)
    scales = []
    for layer in model.layers[:-1]:
        kind = _hidden_layer_kind(layer)
        if hidden_energy_layers == "all":
            scales.append(kappa)
        elif kind == "pool":
            scales.append(1.0)
        elif hidden_energy_layers == "residual":
            scales.append(kappa if kind == "residual" else 1.0)
        else:
            scales.append(kappa)
    return jnp.asarray(scales, dtype=jnp.float32)


def is_pool_layer(layer) -> bool:
    return isinstance(layer, nn.AvgPool2d)


def collapse_plan(model) -> Tuple[Tuple[Tuple[int, ...], int], ...]:
    """Map ResNet modules to PC nodes with pools folded into the next weight layer.

    Each item is ``(pool_indices, layer_index)``. The forward from the previous
    free activity (or the input) is those pools, then ``model[layer_index]``.
    Pool layers are not free activities.
    """
    plan = []
    pending = []
    for idx, layer in enumerate(model.layers):
        if is_pool_layer(layer):
            pending.append(idx)
        else:
            plan.append((tuple(pending), idx))
            pending = []
    if pending:
        raise ValueError(
            f"ResNet ends with pool layers {pending}; cannot fold into a "
            "following weight layer."
        )
    if not plan:
        raise ValueError("ResNet has no non-pool layers.")
    return tuple(plan)


def apply_composed_module(model, pool_idxs, layer_idx, z):
    for pool_idx in pool_idxs:
        z = model[pool_idx](z)
    return model[layer_idx](z)


def vmap_composed_module(model, pool_idxs, layer_idx, batch):
    return vmap(
        lambda z: apply_composed_module(model, pool_idxs, layer_idx, z)
    )(batch)


def init_activities_collapse_pools(model, x):
    """Feedforward activities for the collapsed-pool PC graph (includes logits)."""
    activities = []
    z = x
    for pool_idxs, layer_idx in collapse_plan(model):
        z = vmap_composed_module(model, pool_idxs, layer_idx, z)
        activities.append(z)
    return activities


def activity_hw(activity) -> float:
    """Spatial ``H*W`` of an activity. Vector readouts contribute 1."""
    activity = jnp.asarray(activity)
    if activity.ndim >= 3:
        return float(activity.shape[-2] * activity.shape[-1])
    return 1.0


def readout_spatial_hw(model, width: int) -> float:
    """``H_r W_r`` inside the readout fan-in, not the last free map.

    The linear layer sees ``width * H_r * W_r`` features.
    For the CIFAR ResNet with three pools that is the ``4×4`` map, 16, even
    when a pool is folded and the last free activity is ``8×8``.
    """
    in_features = int(model.layers[-1].linear.weight.shape[-1])
    channels = int(width)
    if channels < 1 or in_features % channels != 0:
        raise ValueError(
            f"Readout fan-in {in_features} is not divisible by channel count "
            f"{channels}."
        )
    hw = in_features // channels
    if hw < 1:
        raise ValueError(f"Readout spatial size must be positive, got {hw}.")
    return float(hw)


def hidden_spatial_hw(model, in_channels, input_size, collapse_pools) -> jnp.ndarray:
    """``H*W`` for each hidden PC node (readout excluded).

    Under ``spatial_mean`` the stored precision is
    ``κ_ℓ = κ (H_r W_r) / (H_ℓ W_ℓ)`` in front of ``(1/2) Σ e²``.
    """
    x = jnp.zeros((1, int(in_channels), int(input_size), int(input_size)))
    if collapse_pools:
        hidden = init_activities_collapse_pools(model, x)[:-1]
    else:
        hidden = []
        z = x
        for layer in model.layers[:-1]:
            z = vmap(layer)(z)
            hidden.append(z)
    if not hidden:
        raise ValueError("CNN has no hidden activities.")
    return jnp.asarray([activity_hw(z) for z in hidden], dtype=jnp.float32)


def cnn_hidden_energy_scales_collapsed(
    model,
    kappa: float,
    hidden_energy_layers: str,
) -> jnp.ndarray:
    """Per-hidden-node ``κ`` for ``collapse_plan`` (no pool nodes)."""
    if hidden_energy_layers not in HIDDEN_ENERGY_LAYER_CHOICES:
        raise ValueError(
            f"Unknown --hidden_energy_layers '{hidden_energy_layers}'. "
            f"Use {', '.join(HIDDEN_ENERGY_LAYER_CHOICES)}."
        )
    kappa = float(kappa)
    scales = []
    for _, layer_idx in collapse_plan(model)[:-1]:
        kind = _hidden_layer_kind(model.layers[layer_idx])
        if hidden_energy_layers == "all":
            scales.append(kappa)
        elif hidden_energy_layers == "residual":
            scales.append(kappa if kind == "residual" else 1.0)
        else:
            scales.append(kappa)
    return jnp.asarray(scales, dtype=jnp.float32)


def _as_hidden_scales(hidden_energy_scaling, n_hidden: int) -> jnp.ndarray:
    if hidden_energy_scaling is None:
        return jnp.ones((n_hidden,), dtype=jnp.float32)
    scales = jnp.asarray(hidden_energy_scaling, dtype=jnp.float32)
    return jnp.broadcast_to(scales, (n_hidden,))


def pc_energy_fn_layered(
    params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
    activities: PyTree[ArrayLike],
    y: ArrayLike,
    *,
    x: Optional[ArrayLike] = None,
    loss: str = "mse",
    output_energy_scaling: Optional[Scalar] = None,
    hidden_energy_scaling: Optional[Union[Scalar, Array, Sequence[float]]] = None,
) -> Scalar:
    """PC energy with optional per-hidden-layer ``κ``.

    CNN µP already lives inside ``ResNet``, so jpc MLP scalings are not
    applied (same as ``pc_jpc_kwargs`` using ``param_type='sp'``).
    """
    model, skip_model = params
    batch_size = y.shape[0]
    start_activity_l = 1 if x is not None else 2
    n_activity_layers = len(activities) - 1
    n_hidden = len(model) - 1

    if skip_model is None:
        skip_model = [None] * len(model)

    output_scale = (
        1.0 if output_energy_scaling is None else output_energy_scaling
    )
    hidden_scales = _as_hidden_scales(hidden_energy_scaling, n_hidden)

    if loss == "mse":
        eL = y - vmap(model[-1])(activities[-2])
        energies = [0.5 * output_scale * jnp.sum(eL ** 2)]
    elif loss == "ce":
        logits = vmap(model[-1])(activities[-2])
        energies = [-output_scale * jnp.sum(y * jax.nn.log_softmax(logits))]
    else:
        raise ValueError(f"Unknown loss '{loss}'.")

    for act_l, net_l in zip(
        range(start_activity_l, n_activity_layers),
        range(1, n_hidden),
    ):
        err = activities[act_l] - vmap(model[net_l])(activities[act_l - 1])
        if skip_model[net_l] is not None:
            err = err - vmap(skip_model[net_l])(activities[act_l - 1])
        energies.append(0.5 * hidden_scales[net_l] * jnp.sum(err ** 2))

    if x is not None:
        e1 = activities[0] - vmap(model[0])(x)
    else:
        e1 = activities[1] - vmap(model[0])(activities[0])
    energies.append(0.5 * hidden_scales[0] * jnp.sum(e1 ** 2))

    return jnp.sum(jnp.stack(energies)) / batch_size


def _output_energy(loss: str, output_scale, logits, y) -> Scalar:
    if loss == "mse":
        err = y - logits
        return 0.5 * output_scale * jnp.sum(err ** 2)
    if loss == "ce":
        return -output_scale * jnp.sum(y * jax.nn.log_softmax(logits))
    raise ValueError(f"Unknown loss '{loss}'.")


def pc_energy_fn_collapse_pools(
    params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
    activities: PyTree[ArrayLike],
    y: ArrayLike,
    *,
    x: Optional[ArrayLike] = None,
    loss: str = "mse",
    output_energy_scaling: Optional[Scalar] = None,
    hidden_energy_scaling: Optional[Union[Scalar, Array, Sequence[float]]] = None,
) -> Scalar:
    """PC energy with AvgPool folded into the next weight layer.

    CE/MSE depends on ``Readout(Pool(z_last_res))``, so the output error
    reaches the last residual activity at the first inference step.
    Each hidden error uses the current activities on both sides.
    """
    if x is None:
        raise ValueError("collapse-pools energy requires the input batch x.")
    model, _skip = params
    plan = collapse_plan(model)
    n_hidden = len(plan) - 1
    if n_hidden < 1:
        raise ValueError("Collapsed PC graph has no hidden nodes.")
    output_scale = (
        1.0 if output_energy_scaling is None else output_energy_scaling
    )
    hidden_scales = _as_hidden_scales(hidden_energy_scaling, n_hidden)

    logits = vmap_composed_module(
        model, plan[-1][0], plan[-1][1], activities[-2]
    )
    energies = [_output_energy(loss, output_scale, logits, y)]

    pred0 = vmap_composed_module(model, plan[0][0], plan[0][1], x)
    energies.append(
        0.5 * hidden_scales[0] * jnp.sum((activities[0] - pred0) ** 2)
    )
    for node in range(1, n_hidden):
        pools, layer_idx = plan[node]
        pred = vmap_composed_module(
            model, pools, layer_idx, activities[node - 1]
        )
        energies.append(
            0.5 * hidden_scales[node] * jnp.sum((activities[node] - pred) ** 2)
        )
    return jnp.sum(jnp.stack(energies)) / y.shape[0]


def _relu_diff(a, d):
    """``relu(a + d) - relu(a)`` without cancellation when ``|d| << |a|``."""
    return jnp.where(a > 0, jnp.maximum(d, -a), jnp.maximum(a + d, 0.0))


def _layer_diff(layer, z0, d):
    """``layer(z0 + d) - layer(z0)`` for one sample, computed on ``d`` directly.

    Convs, pools and the readout are linear, so their JVP is the exact
    difference. A ReLU residual block is ``d + conv(relu(z0 + d) - relu(z0))``.
    """
    if isinstance(layer, ResNetBlock):
        if layer.act_fn is not jax.nn.relu:
            raise ValueError("--pc_delta_activities supports ReLU residual blocks only.")
        u = _relu_diff(z0, d)
        return d + jax.jvp(layer.scaled_conv, (jax.nn.relu(z0),), (u,))[1]
    return jax.jvp(layer, (z0,), (d,))[1]


def pc_energy_fn_delta(
    params,
    deltas,
    y,
    *,
    base,
    x,
    loss="mse",
    output_energy_scaling=None,
    hidden_energy_scaling=None,
    with_param_grads=False,
) -> Scalar:
    """``pc_energy_fn_layered`` at ``z = base + deltas``, with ``base`` the
    feedforward pass of the current weights.

    Since ``base_ℓ = f_ℓ(base_{ℓ-1})``, the error is
    ``e_ℓ = δ_ℓ - [f_ℓ(base_{ℓ-1} + δ_{ℓ-1}) - f_ℓ(base_{ℓ-1})]``. Low maps move
    by ~1e-7 of their size, below float32 resolution of ``z``, so both the
    stored state and the error are kept in this displacement form.

    With ``with_param_grads`` the weight gradient is that of the standard
    energy at ``z`` (``-κ e ∂f(z_{ℓ-1})/∂W``) and the error value is unchanged.
    """
    model, skip_model = params
    if skip_model is not None and any(s is not None for s in skip_model):
        raise ValueError("--pc_delta_activities does not support skip models.")
    n_hidden = len(model) - 1
    output_scale = 1.0 if output_energy_scaling is None else output_energy_scaling
    hidden_scales = _as_hidden_scales(hidden_energy_scaling, n_hidden)
    model_sg = jax.tree_util.tree_map(
        lambda a: jax.lax.stop_gradient(a) if eqx.is_array(a) else a, model
    )

    errs = [deltas[0]]
    for l in range(1, n_hidden):
        diff = vmap(lambda z0, d: _layer_diff(model_sg[l], z0, d))(
            base[l - 1], deltas[l - 1]
        )
        errs.append(deltas[l] - diff)

    if with_param_grads:
        prev = [x] + [jax.lax.stop_gradient(b + d) for b, d in zip(base, deltas)]
        for l in range(n_hidden):
            pred = vmap(model[l])(prev[l])
            errs[l] = jax.lax.stop_gradient(errs[l]) - (
                pred - jax.lax.stop_gradient(pred)
            )

    logits = vmap(model[-1])(base[-2] + deltas[-2])
    energies = [_output_energy(loss, output_scale, logits, y)]
    for l, err in enumerate(errs):
        energies.append(0.5 * hidden_scales[l] * jnp.sum(err ** 2))
    return jnp.sum(jnp.stack(energies)) / y.shape[0]


def pc_energy_fn(
    params,
    activities,
    y,
    *,
    x=None,
    loss="mse",
    output_energy_scaling=None,
    hidden_energy_scaling=None,
    collapse_pools=False,
    base=None,
    with_param_grads=False,
):
    """``base`` given: ``activities`` are displacements from it (see
    ``pc_energy_fn_delta``)."""
    if base is not None:
        if collapse_pools:
            raise ValueError("--pc_delta_activities does not support collapsed pools.")
        return pc_energy_fn_delta(
            params,
            activities,
            y,
            base=base,
            x=x,
            loss=loss,
            output_energy_scaling=output_energy_scaling,
            hidden_energy_scaling=hidden_energy_scaling,
            with_param_grads=with_param_grads,
        )
    if collapse_pools:
        return pc_energy_fn_collapse_pools(
            params,
            activities,
            y,
            x=x,
            loss=loss,
            output_energy_scaling=output_energy_scaling,
            hidden_energy_scaling=hidden_energy_scaling,
        )
    return pc_energy_fn_layered(
        params,
        activities,
        y,
        x=x,
        loss=loss,
        output_energy_scaling=output_energy_scaling,
        hidden_energy_scaling=hidden_energy_scaling,
    )


def activity_rms_deltas(activities, activities0):
    """Per-layer RMS ``‖z − z₀‖`` after inference (includes logits)."""
    deltas = []
    for z, z0 in zip(activities, activities0):
        deltas.append(jnp.sqrt(jnp.mean(jnp.square(z - z0))))
    return jnp.stack(deltas)


@eqx.filter_jit
def update_pc_activities(
    params,
    activities,
    optim,
    opt_state,
    output,
    *,
    input=None,
    loss_id="mse",
    output_energy_scaling=None,
    hidden_energy_scaling=None,
    collapse_pools=False,
    activity_grad_scales=None,
    base=None,
):
    """One local activity step on every free activity."""
    energy, grads = jax.value_and_grad(pc_energy_fn, argnums=1)(
        params,
        activities,
        output,
        x=input,
        loss=loss_id,
        output_energy_scaling=output_energy_scaling,
        hidden_energy_scaling=hidden_energy_scaling,
        collapse_pools=collapse_pools,
        base=base,
    )
    if activity_grad_scales is not None:
        scales = jnp.asarray(activity_grad_scales, dtype=jnp.float32)
        leaves, treedef = jax.tree_util.tree_flatten(grads)
        leaves = [g * scales[i] for i, g in enumerate(leaves)]
        grads = jax.tree_util.tree_unflatten(treedef, leaves)
    updates, opt_state = optim.update(
        updates=grads,
        state=opt_state,
        params=activities,
    )
    activities = eqx.apply_updates(model=activities, updates=updates)
    return {
        "energy": energy,
        "activities": activities,
        "grads": grads,
        "opt_state": opt_state,
    }


def _mask_activity_grads(grads, index, activity_grad_scales):
    """Keep one activity gradient and zero the rest.

    ``index`` is a Python int. The output activity is never selected: the
    loss depends on the readout of the last hidden map, not on that entry.
    """
    leaves, treedef = jax.tree_util.tree_flatten(grads)
    scales = (
        None
        if activity_grad_scales is None
        else jnp.asarray(activity_grad_scales, dtype=jnp.float32)
    )
    masked = []
    for j, grad in enumerate(leaves):
        if j != index:
            masked.append(jnp.zeros_like(grad))
        elif scales is None:
            masked.append(grad)
        else:
            masked.append(grad * scales[j])
    return jax.tree_util.tree_unflatten(treedef, masked)


@eqx.filter_jit
def update_pc_activities_downward(
    params,
    activities,
    optim,
    opt_state,
    output,
    *,
    input=None,
    loss_id="mse",
    output_energy_scaling=None,
    hidden_energy_scaling=None,
    collapse_pools=False,
    activity_grad_scales=None,
    base=None,
):
    """One downward sweep of local activity steps, from the top hidden map.

    Each step is the same gradient as ``update_pc_activities``, restricted
    to one map. The map below is differentiated after the map above has
    moved. The final entry (logits) is left unchanged.
    """

    def energy_of(acts):
        return pc_energy_fn(
            params,
            acts,
            output,
            x=input,
            loss=loss_id,
            output_energy_scaling=output_energy_scaling,
            hidden_energy_scaling=hidden_energy_scaling,
            collapse_pools=collapse_pools,
            base=base,
        )

    energy = None
    grads = None
    # Last entry is the output activity. CE and MSE both read activities[-2].
    for index in range(len(activities) - 2, -1, -1):
        energy, grads = jax.value_and_grad(energy_of)(activities)
        grads = _mask_activity_grads(grads, index, activity_grad_scales)
        updates, opt_state = optim.update(
            updates=grads,
            state=opt_state,
            params=activities,
        )
        activities = eqx.apply_updates(model=activities, updates=updates)
    return {
        "energy": energy,
        "activities": activities,
        "grads": grads,
        "opt_state": opt_state,
    }


@eqx.filter_jit
def update_pc_params(
    params,
    activities,
    optim,
    opt_state,
    output,
    *,
    input=None,
    loss_id="mse",
    output_energy_scaling=None,
    hidden_energy_scaling=None,
    collapse_pools=False,
    base=None,
):
    grads = eqx.filter_grad(pc_energy_fn)(
        params,
        activities,
        output,
        x=input,
        loss=loss_id,
        output_energy_scaling=output_energy_scaling,
        hidden_energy_scaling=hidden_energy_scaling,
        collapse_pools=collapse_pools,
        base=base,
        with_param_grads=True,
    )
    updates, opt_state = optim.update(
        updates=grads,
        state=opt_state,
        params=params,
    )
    model, skip_model = eqx.apply_updates(model=params, updates=updates)
    return {
        "model": model,
        "skip_model": skip_model,
        "grads": grads,
        "opt_state": opt_state,
    }
