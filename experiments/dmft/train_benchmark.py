"""Finite-size PC vs backprop benchmark on classification datasets.

Trains a predictive-coding network and a backprop network on the **same**
minibatches (augment once, then both update) for ``--n_epochs`` passes over
the training set, then overlays train/test loss and accuracy.

Energy scalings match ``train.py`` / ``train_pcn``:

    λ = γ² N L    (µPC output precision)
    κ = L         (µPC hidden precision)

For Adam, the PC parameter LR is divided the same way as BP
(``1/√N``, or ``1/√(N L)`` with MLP skips). GD keeps the ``train.py``
convention: PC uses the raw ``--param_lr_pc`` (scale lives in the energy);
BP bakes ``γ² N`` into the optimiser.

CNN residual blocks still use the per-parameter Adam tree from
``configure_cnn_param_optim``. ImageNet is streamed from Hugging Face.

Default architecture: MLP for MNIST / Fashion-MNIST, CNN otherwise.
Override with ``--arch``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import torch
from torch.utils.data import DataLoader

import jpc

from experiments.datasets import (
    get_dataset,
    get_tinyimagenet_loaders,
    TinyImageNet,
)
from experiments.dmft.utils import (
    MLP,
    copy_mlp_linear_params,
    get_hidden_energy_scaling,
    get_output_energy_scaling,
)
from experiments.limits_paper.utils import configure_param_optim
from experiments.mupc_paper.utils import set_seed

_CNN_DIR = Path(__file__).resolve().parents[1] / "limits_paper" / "cnn"
if str(_CNN_DIR) not in sys.path:
    sys.path.insert(0, str(_CNN_DIR))

from model import ResNet  # noqa: E402
from optim import configure_cnn_param_optim  # noqa: E402
from experiments.limits_paper.cnn.utils import _import_hf_load_dataset  # noqa: E402


DATASET_ALIASES = {
    "mnist": "MNIST",
    "fashion-mnist": "Fashion-MNIST",
    "fashionmnist": "Fashion-MNIST",
    "cifar10": "CIFAR10",
    "cifar-10": "CIFAR10",
    "cifar": "CIFAR10",
    "tinyimagenet": "TinyImageNet",
    "tiny-imagenet": "TinyImageNet",
    "tiny_imagenet": "TinyImageNet",
    "imagenet": "ImageNet",
}

DATASET_SPECS = {
    "MNIST": dict(in_channels=1, input_size=28, n_classes=10, flatten_dim=784),
    "Fashion-MNIST": dict(
        in_channels=1, input_size=28, n_classes=10, flatten_dim=784
    ),
    "CIFAR10": dict(
        in_channels=3, input_size=32, n_classes=10, flatten_dim=3072
    ),
    "TinyImageNet": dict(
        in_channels=3, input_size=64, n_classes=200, flatten_dim=12288
    ),
    "ImageNet": dict(
        in_channels=3, input_size=224, n_classes=1000, flatten_dim=150528
    ),
}

MLP_DATASETS = {"MNIST", "Fashion-MNIST"}
IMAGENET_TRAIN_SIZE = 1_281_167
IMAGENET_VAL_SIZE = 50_000
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_dataset_id(name):
    key = name.strip().lower().replace(" ", "-")
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    if name in DATASET_SPECS:
        return name
    raise ValueError(
        f"Unknown dataset '{name}'. Options: MNIST, Fashion-MNIST, "
        "CIFAR10, TinyImageNet, ImageNet."
    )


def default_arch_for_dataset(dataset_id):
    return "mlp" if dataset_id in MLP_DATASETS else "cnn"


def cnn_energy_depth(n_res_blocks, additive_depth_factor):
    """µP / energy depth ``L`` for the CNN, matching ``configure_cnn_param_optim``."""
    return int(n_res_blocks) + int(additive_depth_factor)


def copy_eqx_arrays(src, dst):
    """Copy array leaves from ``src`` onto ``dst`` (same pytree structure)."""
    src_params, _ = eqx.partition(src, eqx.is_array)
    _, dst_static = eqx.partition(dst, eqx.is_array)
    return eqx.combine(src_params, dst_static)


def mlp_adam_lr(param_lr, param_type, use_skips, width, depth):
    """Adam LR matching ``configure_param_optim`` (BP and, here, PC)."""
    if param_type == "sp":
        return param_lr
    if use_skips:
        return param_lr / (np.sqrt(width) * np.sqrt(depth))
    return param_lr / np.sqrt(width)


def supervised_loss(preds, y, loss_id):
    if loss_id == "mse":
        return jpc.mse_loss(preds, y)
    return jpc.cross_entropy_loss(preds, y)


def accuracy_pct(preds, y):
    return jpc.compute_accuracy(y, preds)


def _to_numpy_batch(x, y):
    if hasattr(x, "numpy"):
        x = x.numpy()
    if hasattr(y, "numpy"):
        y = y.numpy()
    return np.asarray(x), np.asarray(y)


def maybe_flatten(x, arch):
    if arch == "mlp" and np.ndim(x) > 2:
        return np.reshape(x, (x.shape[0], -1))
    return x


def to_jax_batch(x, y, arch):
    x, y = _to_numpy_batch(x, y)
    x = maybe_flatten(x, arch)
    return jnp.asarray(x), jnp.asarray(y)


def _parse_imagenet_example(ex, transform):
    img = None
    for key in ("image", "img", "pixels", "jpg"):
        if key in ex and ex[key] is not None:
            img = ex[key]
            break
    if img is None:
        raise KeyError(
            f"Could not find an image in example keys: {list(ex.keys())}"
        )
    if isinstance(img, (bytes, bytearray)):
        import io
        from PIL import Image

        img = Image.open(io.BytesIO(img)).convert("RGB")
    elif hasattr(img, "convert"):
        img = img.convert("RGB")
    else:
        raise TypeError(f"Unrecognized ImageNet image type: {type(img)}")

    x = transform(img).cpu().numpy().astype(np.float32)

    label = None
    for key in ("label", "labels", "cls"):
        if key in ex and ex[key] is not None:
            label = ex[key]
            break
    if label is None:
        raise KeyError(
            f"Could not find a label in example keys: {list(ex.keys())}"
        )
    label = int(label)
    y = np.zeros((1000,), dtype=np.float32)
    y[label] = 1.0
    return x, y


def _hf_imagenet_split(dataset, names):
    for name in names:
        if name in dataset:
            return dataset[name]
    raise KeyError(
        f"None of {names} found in ImageNet splits {list(dataset.keys())}"
    )


def iter_imagenet_hf(
    split,
    batch_size,
    seed,
    n_examples,
    *,
    train,
    drop_last=True,
):
    """Yield JAX ``(x, y)`` batches from streamed ImageNet-1K."""
    try:
        load_dataset = _import_hf_load_dataset()
    except Exception as exc:
        raise ImportError(
            "Failed to import Hugging Face `datasets.load_dataset`. "
            "Install `datasets` and set HF_TOKEN for gated ImageNet-1K."
        ) from exc

    from torchvision import transforms

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGINGFACE_HUB_TOKEN"
    )
    dataset = load_dataset(
        "timm/imagenet-1k-wds",
        streaming=True,
        token=hf_token,
    )
    if train:
        split_ds = _hf_imagenet_split(dataset, ("train",))
        split_ds = split_ds.shuffle(seed=int(seed), buffer_size=10_000)
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        split_ds = _hf_imagenet_split(
            dataset, ("validation", "val", "test")
        )
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    xs, ys = [], []
    n_seen = 0
    for ex in split_ds:
        if n_seen >= n_examples:
            break
        x, y = _parse_imagenet_example(ex, transform)
        xs.append(x)
        ys.append(y)
        n_seen += 1
        if len(xs) == batch_size:
            yield (
                jnp.asarray(np.stack(xs, axis=0)),
                jnp.asarray(np.stack(ys, axis=0)),
            )
            xs, ys = [], []

    if xs and not drop_last:
        yield (
            jnp.asarray(np.stack(xs, axis=0)),
            jnp.asarray(np.stack(ys, axis=0)),
        )


def make_torch_loaders(dataset_id, batch_size, flatten, seed):
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    if dataset_id in ("MNIST", "Fashion-MNIST", "CIFAR10"):
        train_data = get_dataset(
            id=dataset_id, train=True, normalise=True, flatten=flatten
        )
        test_data = get_dataset(
            id=dataset_id, train=False, normalise=True, flatten=flatten
        )
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=gen,
        )
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        return train_loader, test_loader

    if dataset_id == "TinyImageNet":
        train_loader, _ = get_tinyimagenet_loaders(
            batch_size=batch_size, generator=gen
        )
        val_data = TinyImageNet(split="val")
        test_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        return train_loader, test_loader

    raise ValueError(f"No torch loaders for dataset '{dataset_id}'")


def _iter_prepared(raw_batches, arch):
    for x, y in raw_batches:
        yield to_jax_batch(x, y, arch)


def iter_train_batches(args, epoch):
    if args.dataset == "ImageNet":
        raw = iter_imagenet_hf(
            split="train",
            batch_size=args.batch_size,
            seed=args.seed + epoch,
            n_examples=IMAGENET_TRAIN_SIZE,
            train=True,
            drop_last=True,
        )
    else:
        raw = args._train_loader
    return _iter_prepared(raw, args.arch)


def iter_test_batches(args):
    if args.dataset == "ImageNet":
        raw = iter_imagenet_hf(
            split="val",
            batch_size=args.batch_size,
            seed=args.seed,
            n_examples=IMAGENET_VAL_SIZE,
            train=False,
            drop_last=False,
        )
    else:
        raw = args._test_loader
    return _iter_prepared(raw, args.arch)


def make_models(key, args, spec):
    if args.arch == "mlp":
        pc_model = jpc.make_mlp(
            key,
            input_dim=spec["flatten_dim"],
            width=args.width,
            depth=args.n_hidden + 1,
            output_dim=spec["n_classes"],
            act_fn=args.act_fn,
            use_bias=False,
            param_type=args.param_type,
        )
        bp_model = MLP(
            key=key,
            d_in=spec["flatten_dim"],
            N=args.width,
            L=args.n_hidden + 1,
            d_out=spec["n_classes"],
            act_fn=args.act_fn,
            param_type=args.param_type,
            gamma=args.gamma,
            use_bias=False,
            use_skips=args.use_skips,
        )
        bp_model = copy_mlp_linear_params(pc_model, bp_model)
        skip_model = (
            jpc.make_skip_model(len(pc_model)) if args.use_skips else None
        )
        return pc_model, bp_model, skip_model

    pc_model = ResNet(
        key=key,
        width=args.width,
        n_res_blocks=args.n_res_blocks,
        in_channels=spec["in_channels"],
        input_size=spec["input_size"],
        out_features=spec["n_classes"],
        param_type=args.param_type,
        act_fn=args.act_fn,
        scale_non_res_layers=args.scale_non_res_layers,
        additive_depth_factor=args.additive_depth_factor,
    )
    bp_model = ResNet(
        key=key,
        width=args.width,
        n_res_blocks=args.n_res_blocks,
        in_channels=spec["in_channels"],
        input_size=spec["input_size"],
        out_features=spec["n_classes"],
        param_type=args.param_type,
        act_fn=args.act_fn,
        scale_non_res_layers=args.scale_non_res_layers,
        additive_depth_factor=args.additive_depth_factor,
    )
    bp_model = copy_eqx_arrays(pc_model, bp_model)
    return pc_model, bp_model, None


def pc_jpc_kwargs(args):
    """Kwargs for jpc init / updates.

    CNN µP lives inside ``ResNet``; passing ``param_type='mupc'`` would apply
    extra MLP scalings (and index ``Linear`` weights that do not exist).
    """
    if args.arch == "cnn":
        return dict(param_type="sp", gamma=None)
    return dict(param_type=args.param_type, gamma=args.gamma)


def make_pc_param_optim(pc_model, skip_model, args, depth):
    if args.arch == "cnn":
        if args.param_optim == "gd":
            # train.py PC GD: raw LR; width/gamma/depth live in the energy.
            param_optim = optax.sgd(args.param_lr_pc)
        else:
            param_optim = configure_cnn_param_optim(
                pc_model,
                optim_id=args.param_optim,
                param_type=args.param_type,
                param_lr=args.param_lr_pc,
                width=args.width,
                depth=depth,
                gamma_0=args.gamma,
                params_for_pc=True,
            )
        opt_state = param_optim.init(
            (eqx.filter(pc_model, eqx.is_array), skip_model)
        )
        return param_optim, opt_state

    if args.param_optim == "gd":
        param_optim = optax.sgd(args.param_lr_pc)
    elif args.param_optim == "adam":
        param_optim = optax.adam(
            mlp_adam_lr(
                args.param_lr_pc,
                args.param_type,
                args.use_skips,
                args.width,
                depth,
            )
        )
    else:
        raise ValueError(f"Invalid optimiser: {args.param_optim}")
    opt_state = param_optim.init(
        (eqx.filter(pc_model, eqx.is_array), skip_model)
    )
    return param_optim, opt_state


def make_bp_param_optim(bp_model, args, depth):
    if args.arch == "cnn":
        optim = configure_cnn_param_optim(
            bp_model,
            optim_id=args.param_optim,
            param_type=args.param_type,
            param_lr=args.param_lr,
            width=args.width,
            depth=depth,
            gamma_0=args.gamma,
            params_for_pc=False,
        )
        return optim, optim.init(eqx.filter(bp_model, eqx.is_array))

    optim = configure_param_optim(
        args.param_optim,
        args.param_type,
        args.use_skips,
        args.param_lr,
        args.width,
        depth,
        args.gamma,
    )
    return optim, optim.init(eqx.filter(bp_model, eqx.is_array))


def pc_ffwd_preds(model, x, skip_model, jpc_kw):
    activities = jpc.init_activities_with_ffwd(
        model=model,
        input=x,
        skip_model=skip_model,
        **jpc_kw,
    )
    return activities[-1]


def pc_batch_metrics(model, x, y, skip_model, jpc_kw, loss_id):
    preds = pc_ffwd_preds(model, x, skip_model, jpc_kw)
    return float(supervised_loss(preds, y, loss_id)), float(accuracy_pct(preds, y))


def bp_batch_metrics(model, x, y, loss_id):
    preds = jax.vmap(model)(x)
    return float(supervised_loss(preds, y, loss_id)), float(accuracy_pct(preds, y))


def evaluate_pc(model, args, skip_model, jpc_kw):
    total_loss, total_acc, n_seen = 0.0, 0.0, 0
    for x, y in iter_test_batches(args):
        b = int(x.shape[0])
        loss, acc = pc_batch_metrics(
            model, x, y, skip_model, jpc_kw, args.loss_id
        )
        total_loss += loss * b
        total_acc += acc * b
        n_seen += b
    if n_seen == 0:
        return float("nan"), float("nan")
    return total_loss / n_seen, total_acc / n_seen


def evaluate_bp(model, args):
    total_loss, total_acc, n_seen = 0.0, 0.0, 0
    for x, y in iter_test_batches(args):
        b = int(x.shape[0])
        loss, acc = bp_batch_metrics(model, x, y, args.loss_id)
        total_loss += loss * b
        total_acc += acc * b
        n_seen += b
    if n_seen == 0:
        return float("nan"), float("nan")
    return total_loss / n_seen, total_acc / n_seen


def pc_infer_and_update(
    model,
    skip_model,
    x,
    y,
    activity_optim,
    param_optim,
    param_opt_state,
    args,
    jpc_kw,
    output_energy_scaling,
    hidden_energy_scaling,
):
    params = (model, skip_model)
    activities = jpc.init_activities_with_ffwd(
        model=model,
        input=x,
        skip_model=skip_model,
        **jpc_kw,
    )
    activity_opt_state = activity_optim.init(activities)
    energy = None
    for _ in range(args.n_infer_iters):
        result = jpc.update_pc_activities(
            params=params,
            activities=activities,
            optim=activity_optim,
            opt_state=activity_opt_state,
            output=y,
            input=x,
            loss_id=args.loss_id,
            output_energy_scaling=output_energy_scaling,
            hidden_energy_scaling=hidden_energy_scaling,
            **jpc_kw,
        )
        activities = result["activities"]
        activity_opt_state = result["opt_state"]
        energy = result["energy"]

    energy = float(energy)
    if not np.isfinite(energy):
        return model, skip_model, param_opt_state, energy, False

    param_result = jpc.update_pc_params(
        params=params,
        activities=activities,
        optim=param_optim,
        opt_state=param_opt_state,
        output=y,
        input=x,
        loss_id=args.loss_id,
        output_energy_scaling=output_energy_scaling,
        hidden_energy_scaling=hidden_energy_scaling,
        **jpc_kw,
    )
    return (
        param_result["model"],
        param_result["skip_model"],
        param_result["opt_state"],
        energy,
        True,
    )


def make_bp_step(loss_id):
    @eqx.filter_jit
    def loss_fn(model, x, y):
        preds = jax.vmap(model)(x)
        return supervised_loss(preds, y, loss_id)

    @eqx.filter_jit
    def step(model, opt_state, optim, x, y):
        _, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optim.update(
            updates=grads,
            state=opt_state,
            params=eqx.filter(model, eqx.is_array),
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state

    return step


def setup_save_dir(args):
    depth_tag = (
        f"{args.n_hidden}_n_hidden"
        if args.arch == "mlp"
        else f"{args.n_res_blocks}_n_res_blocks"
    )
    return os.path.join(
        args.results_dir,
        args.dataset,
        args.arch,
        args.loss_id,
        f"{args.width}_width",
        depth_tag,
        f"{args.act_fn}_act_fn",
        f"{args.param_type}_param_type",
        f"{args.gamma}_gamma",
        f"{args.param_optim}_param_optim",
        f"{args.param_lr}_param_lr",
        f"{args.param_lr_pc}_param_lr_pc",
        f"{args.batch_size}_batch_size",
        f"{args.n_epochs}_n_epochs",
        f"{args.n_infer_iters}_n_infer_iters",
        f"{args.activity_lr}_activity_lr",
        f"{args.use_skips}_use_skips",
        str(args.seed),
    )


def _plot_overlay(ax, xs_pc, ys_pc, xs_bp, ys_bp, xlabel, ylabel):
    ax.plot(xs_pc, ys_pc, marker="o", color="tab:blue", label="PC", alpha=0.9)
    ax.plot(
        xs_bp, ys_bp, marker="s", color="tab:orange", label="Backprop", alpha=0.9
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)


def plot_metrics(history, save_dir, title_suffix=""):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _plot_overlay(
        axes[0, 0],
        history["epoch_train"],
        history["pc_train_loss_epoch"],
        history["epoch_train"],
        history["bp_train_loss_epoch"],
        "Epoch",
        "Train loss",
    )
    _plot_overlay(
        axes[0, 1],
        history["epoch_eval"],
        history["pc_test_loss"],
        history["epoch_eval"],
        history["bp_test_loss"],
        "Epoch",
        "Test loss",
    )
    _plot_overlay(
        axes[1, 0],
        history["epoch_train"],
        history["pc_train_acc_epoch"],
        history["epoch_train"],
        history["bp_train_acc_epoch"],
        "Epoch",
        "Train accuracy (%)",
    )
    _plot_overlay(
        axes[1, 1],
        history["epoch_eval"],
        history["pc_test_acc"],
        history["epoch_eval"],
        history["bp_test_acc"],
        "Epoch",
        "Test accuracy (%)",
    )
    fig.suptitle(f"PC vs backprop{title_suffix}")
    fig.tight_layout()
    epoch_path = os.path.join(save_dir, "pc_bp_epoch_metrics.png")
    fig.savefig(epoch_path, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    steps = np.arange(len(history["pc_train_loss_step"]))
    _plot_overlay(
        axes[0],
        steps,
        history["pc_train_loss_step"],
        steps,
        history["bp_train_loss_step"],
        "Step",
        "Train loss",
    )
    _plot_overlay(
        axes[1],
        steps,
        history["pc_train_acc_step"],
        steps,
        history["bp_train_acc_step"],
        "Step",
        "Train accuracy (%)",
    )
    fig.suptitle(f"PC vs backprop (per step){title_suffix}")
    fig.tight_layout()
    step_path = os.path.join(save_dir, "pc_bp_step_metrics.png")
    fig.savefig(step_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plots to {epoch_path} and {step_path}")
    return epoch_path, step_path


def save_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for key, value in history.items():
        np.save(os.path.join(save_dir, f"{key}.npy"), np.asarray(value))


def run_benchmark(args):
    spec = DATASET_SPECS[args.dataset]
    set_seed(args.seed)
    key = jr.PRNGKey(args.seed)

    if args.arch == "mlp":
        depth = args.n_hidden + 1
        if args.use_skips:
            print("MLP skip connections enabled (Adam LR uses 1/√(N L)).")
    else:
        if args.n_res_blocks % 3 != 0:
            raise ValueError(
                f"--n_res_blocks must be a multiple of 3, got {args.n_res_blocks}"
            )
        depth = cnn_energy_depth(
            args.n_res_blocks, args.additive_depth_factor
        )
        if args.use_skips:
            print(
                "Note: --use_skips is an MLP flag; CNN residual blocks are "
                "already in the architecture. Adam CNN LRs still use the "
                "res-block vs stage split from configure_cnn_param_optim."
            )

    output_energy_scaling = get_output_energy_scaling(
        args.param_type, args.gamma, args.width, depth
    )
    hidden_energy_scaling = get_hidden_energy_scaling(args.param_type, depth)
    jpc_kw = pc_jpc_kwargs(args)

    if args.dataset != "ImageNet":
        flatten = args.arch == "mlp"
        args._train_loader, args._test_loader = make_torch_loaders(
            args.dataset, args.batch_size, flatten, args.seed
        )
    else:
        args._train_loader = args._test_loader = None

    pc_model, bp_model, skip_model = make_models(key, args, spec)
    pc_param_optim, pc_opt_state = make_pc_param_optim(
        pc_model, skip_model, args, depth
    )
    bp_param_optim, bp_opt_state = make_bp_param_optim(bp_model, args, depth)
    bp_step = make_bp_step(args.loss_id)
    activity_optim = optax.sgd(args.activity_lr * args.batch_size)

    save_dir = setup_save_dir(args)
    os.makedirs(save_dir, exist_ok=True)
    args_to_save = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_")
    }
    with open(os.path.join(save_dir, "args.json"), "w", encoding="utf-8") as handle:
        json.dump(args_to_save, handle, indent=2, default=str)

    print(
        f"Benchmark {args.dataset} ({args.arch}), width={args.width}, "
        f"L={depth}, γ={args.gamma}, λ={output_energy_scaling}, "
        f"κ={hidden_energy_scaling}, optim={args.param_optim}, "
        f"lr_bp={args.param_lr}, lr_pc={args.param_lr_pc}"
    )

    history = {
        "epoch_eval": [],
        "epoch_train": [],
        "pc_train_loss_epoch": [],
        "bp_train_loss_epoch": [],
        "pc_train_acc_epoch": [],
        "bp_train_acc_epoch": [],
        "pc_test_loss": [],
        "bp_test_loss": [],
        "pc_test_acc": [],
        "bp_test_acc": [],
        "pc_train_loss_step": [],
        "bp_train_loss_step": [],
        "pc_train_acc_step": [],
        "bp_train_acc_step": [],
        "pc_energy_step": [],
    }

    print("Evaluating at initialization...")
    pc_test_loss, pc_test_acc = evaluate_pc(
        pc_model, args, skip_model, jpc_kw
    )
    bp_test_loss, bp_test_acc = evaluate_bp(bp_model, args)
    history["epoch_eval"].append(0)
    history["pc_test_loss"].append(pc_test_loss)
    history["bp_test_loss"].append(bp_test_loss)
    history["pc_test_acc"].append(pc_test_acc)
    history["bp_test_acc"].append(bp_test_acc)
    print(
        f"  init  PC test loss={pc_test_loss:.4f} acc={pc_test_acc:.2f}%  |  "
        f"BP test loss={bp_test_loss:.4f} acc={bp_test_acc:.2f}%"
    )
    if abs(pc_test_loss - bp_test_loss) > 1e-3:
        print(
            "  Warning: PC and BP test losses differ at init; check weight copy."
        )

    global_step = 0
    for epoch in range(1, args.n_epochs + 1):
        pc_loss_sum = bp_loss_sum = 0.0
        pc_acc_sum = bp_acc_sum = 0.0
        n_batches = 0

        for x, y in iter_train_batches(args, epoch):
            pc_loss, pc_acc = pc_batch_metrics(
                pc_model, x, y, skip_model, jpc_kw, args.loss_id
            )
            bp_loss, bp_acc = bp_batch_metrics(bp_model, x, y, args.loss_id)

            pc_model, skip_model, pc_opt_state, energy, pc_ok = (
                pc_infer_and_update(
                    pc_model,
                    skip_model,
                    x,
                    y,
                    activity_optim,
                    pc_param_optim,
                    pc_opt_state,
                    args,
                    jpc_kw,
                    output_energy_scaling,
                    hidden_energy_scaling,
                )
            )
            if not pc_ok:
                print(
                    f"  Warning: non-finite PC energy at epoch {epoch} "
                    f"step {global_step}; skipped PC parameter update."
                )

            bp_model, bp_opt_state = bp_step(
                bp_model, bp_opt_state, bp_param_optim, x, y
            )

            history["pc_train_loss_step"].append(pc_loss)
            history["bp_train_loss_step"].append(bp_loss)
            history["pc_train_acc_step"].append(pc_acc)
            history["bp_train_acc_step"].append(bp_acc)
            history["pc_energy_step"].append(energy)

            pc_loss_sum += pc_loss
            bp_loss_sum += bp_loss
            pc_acc_sum += pc_acc
            bp_acc_sum += bp_acc
            n_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                print(
                    f"  epoch {epoch} step {global_step}: "
                    f"PC loss={pc_loss:.4f} acc={pc_acc:.2f}%  |  "
                    f"BP loss={bp_loss:.4f} acc={bp_acc:.2f}%  |  "
                    f"energy={energy:.4f}"
                )

        if n_batches == 0:
            raise RuntimeError(
                f"No training batches in epoch {epoch}. Check the dataset path "
                "or Hugging Face token for ImageNet."
            )

        history["epoch_train"].append(epoch)
        history["pc_train_loss_epoch"].append(pc_loss_sum / n_batches)
        history["bp_train_loss_epoch"].append(bp_loss_sum / n_batches)
        history["pc_train_acc_epoch"].append(pc_acc_sum / n_batches)
        history["bp_train_acc_epoch"].append(bp_acc_sum / n_batches)

        print(f"Evaluating after epoch {epoch}...")
        pc_test_loss, pc_test_acc = evaluate_pc(
            pc_model, args, skip_model, jpc_kw
        )
        bp_test_loss, bp_test_acc = evaluate_bp(bp_model, args)
        history["epoch_eval"].append(epoch)
        history["pc_test_loss"].append(pc_test_loss)
        history["bp_test_loss"].append(bp_test_loss)
        history["pc_test_acc"].append(pc_test_acc)
        history["bp_test_acc"].append(bp_test_acc)
        print(
            f"  epoch {epoch}: "
            f"PC train {history['pc_train_loss_epoch'][-1]:.4f} "
            f"({history['pc_train_acc_epoch'][-1]:.2f}%)  "
            f"test {pc_test_loss:.4f} ({pc_test_acc:.2f}%)  |  "
            f"BP train {history['bp_train_loss_epoch'][-1]:.4f} "
            f"({history['bp_train_acc_epoch'][-1]:.2f}%)  "
            f"test {bp_test_loss:.4f} ({bp_test_acc:.2f}%)"
        )

        save_history(history, save_dir)
        plot_metrics(
            history,
            os.path.join(save_dir, "plots"),
            title_suffix=(
                f" ({args.dataset}, {args.arch}, N={args.width}, L={depth})"
            ),
        )

    print(f"Done. Results in {save_dir}")
    return save_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="PC vs backprop finite-size dataset benchmark."
    )
    parser.add_argument("--results_dir", type=str, default="results_benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        help="MNIST, Fashion-MNIST, CIFAR10, TinyImageNet, or ImageNet.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        choices=["mlp", "cnn"],
        help=(
            "Architecture. Default: mlp for MNIST / Fashion-MNIST, "
            "cnn otherwise."
        ),
    )

    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--n_hidden", type=int, default=3, help="MLP hidden layers.")
    parser.add_argument(
        "--n_res_blocks",
        type=int,
        default=3,
        help="CNN residual blocks (multiple of 3).",
    )
    parser.add_argument(
        "--param_type", type=str, default="mupc", choices=["sp", "mupc"]
    )
    parser.add_argument("--act_fn", type=str, default="relu")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--use_skips", action="store_true", default=False)
    parser.add_argument(
        "--scale_non_res_layers", action="store_true", default=False
    )
    parser.add_argument("--additive_depth_factor", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument(
        "--param_optim", type=str, default="adam", choices=["gd", "adam"]
    )
    parser.add_argument(
        "--param_lr",
        type=float,
        default=1e-3,
        help="Backprop parameter learning rate.",
    )
    parser.add_argument(
        "--param_lr_pc",
        type=float,
        default=1e-3,
        help="PC parameter learning rate (Adam: divided like BP).",
    )
    parser.add_argument("--loss_id", type=str, default="ce", choices=["mse", "ce"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=50)

    parser.add_argument("--activity_lr", type=float, default=0.3)
    parser.add_argument("--n_infer_iters", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.dataset = normalize_dataset_id(args.dataset)
    if args.arch is None:
        args.arch = default_arch_for_dataset(args.dataset)
        print(f"Using default --arch {args.arch} for {args.dataset}")

    base_seed = args.seed
    for seed in range(base_seed, base_seed + args.n_seeds):
        run_args = argparse.Namespace(**vars(args))
        run_args.seed = seed
        # Drop non-serialisable loader handles before the next seed.
        run_args._train_loader = None
        run_args._test_loader = None
        run_benchmark(run_args)



# # MLP, MNIST
# python train_benchmark.py --dataset MNIST --n_epochs 5 --batch_size 64 --width 128 --n_hidden 3 --param_lr 0.01 --param_lr_pc 0.1 --activity_lr 0.1 --n_infer_iters 10 --param_optim adam --act_fn relu

# # CNN, CIFAR-10
# python train_benchmark.py --dataset CIFAR10 --n_epochs 10 --batch_size 64 --width 64 --n_res_blocks 3 --n_hidden 3 --param_lr 0.01 --param_lr_pc 0.1 --activity_lr 0.1 --n_infer_iters 10 --param_optim adam --act_fn relu

# # ImageNet (HF streaming)
# python train_benchmark.py --dataset ImageNet --n_epochs 1 --batch_size 64 --width 64 --n_res_blocks 3 --n_hidden 3 --param_lr 0.01 --param_lr_pc 0.1 --activity_lr 0.1 --n_infer_iters 10 --param_optim adam --act_fn relu