from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import cycle
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from separable_qp_layer import SeparableQPProjection, SeparableQPSolveFromBeta, FixedZProjection


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class DemoCfg:
    out_dir: str = "demo_outputs"
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    b: int = 256
    eps_logx: float = 1e-12


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def print_stats_table(title: str, rows: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print nested dictionaries as a small table."""
    df = pd.DataFrame(rows).T
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 1000,
        "display.float_format", lambda v: f"{v:8.4f}",
    ):
        print(f"\n{title}\n{df}\n")


def simplex_projection_reference(v: torch.Tensor, s: float = 1.0) -> torch.Tensor:
    """
    Euclidean projection onto simplex {x>=0, sum x = s} (Duchi et al. style).
    Works on last dim, batched.
    """
    if v.dim() == 1:
        v = v.unsqueeze(0)
    B, N = v.shape
    u, _ = torch.sort(v, dim=-1, descending=True)
    cssv = torch.cumsum(u, dim=-1) - s
    ind = torch.arange(1, N + 1, device=v.device, dtype=v.dtype).view(1, -1)
    cond = u - cssv / ind > 0
    rho = cond.to(torch.int64).sum(dim=-1) - 1
    rho = torch.clamp(rho, min=0)
    theta = cssv[torch.arange(B, device=v.device), rho] / (rho.to(v.dtype) + 1.0)
    w = torch.clamp(v - theta.unsqueeze(-1), min=0.0)
    return w


def row_stats(x: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """
    Stats that are stable and interpretable for simplex-ish outputs.
    """
    x = x.detach()
    B, N = x.shape
    row_sum = x.sum(dim=-1)
    l0 = (x > 0).sum(dim=-1).float()
    top1 = x.max(dim=-1).values
    # Entropy (define 0*log0 := 0)
    p = torch.clamp(x, min=eps)
    ent = -(p * torch.log(p)).sum(dim=-1)
    return {
        "mean_value": float(x.mean().item()),
        "std_value": float(x.std().item()),
        "min_value": float(x.min().item()),
        "max_value": float(x.max().item()),
        "mean_row_sum": float(row_sum.mean().item()),
        "std_row_sum": float(row_sum.std().item()),
        "mean_L0": float(l0.mean().item()),
        "std_L0": float(l0.std().item()),
        "mean_top1": float(top1.mean().item()),
        "mean_entropy": float(ent.mean().item()),
    }


def feasibility_stats(x: torch.Tensor, m: float | torch.Tensor, M: float | torch.Tensor, xi: float | torch.Tensor) -> Dict[str, float]:
    x = x.detach()
    # broadcast scalars
    if not torch.is_tensor(m):
        m = torch.tensor(float(m), device=x.device, dtype=x.dtype)
    if not torch.is_tensor(M):
        M = torch.tensor(float(M), device=x.device, dtype=x.dtype)
    if not torch.is_tensor(xi):
        xi = torch.tensor(float(xi), device=x.device, dtype=x.dtype)

    row_sum = x.sum(dim=-1)
    sum_err = row_sum - xi.reshape(-1) if xi.numel() > 1 else row_sum - xi
    below = (m - x).clamp(min=0).max().item()
    above = (x - M).clamp(min=0).max().item()
    return {
        "mean_abs_sum_err": float(sum_err.abs().mean().item()),
        "max_abs_sum_err": float(sum_err.abs().max().item()),
        "max_below_m": float(below),
        "max_above_M": float(above),
    }


def plot_hist(
    path: str,
    title: str,
    series: Dict[str, torch.Tensor],
    bins: int = 80,
    density: bool = True,
    logy: bool = False,
    xlim: Tuple[float, float] | None = None,
    rug: bool = False,
    xlabel: str = "value",
) -> None:
    plt.figure(figsize=(8, 4.5))
    ax = plt.gca()
    for name, x in series.items():
        data = to_np(x.reshape(-1))
        ax.hist(
            data,
            bins=bins,
            density=density,
            alpha=0.55,
            label=name,
            histtype="stepfilled",
        )
        if rug:
            ax.plot(data, np.zeros_like(data), '|', color='k', alpha=0.3, transform=ax.get_xaxis_transform())

    if logy:
        plt.yscale("log")
    if xlim is not None:
        plt.xlim(*xlim)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.legend()
    savefig(path)


def plot_value_panels(
    path: str,
    title: str,
    series: Dict[str, torch.Tensor],
    xlim: Tuple[float, float] | None = None,
    bins: int = 120,
    eps: float = 1e-12,
    vline: float | None = 0.0,
) -> None:
    """
    Compact view: linear hist, log-y hist, and log10(x+eps) side-by-side.

    A thin reference line is drawn on the value-based panels so that sparse
    distributions still reveal where zero lies; the log10 view uses the clamped
    epsilon as the comparable baseline. The log panel keeps independent axes so
    that log-y scaling does not distort the linear view.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharey=False)
    for ax, (logy, xlabel) in zip(
        axes,
        [(False, "value"), (True, "value"), (False, "log10(value)")],
    ):
        for name, x in series.items():
            if xlabel == "value":
                data = to_np(x.reshape(-1))
            else:
                data = to_np(torch.log10(torch.clamp(x, min=eps)).reshape(-1))
            ax.hist(
                data,
                bins=bins,
                density=True,
                alpha=0.6,
                label=name if xlabel == "value" else f"log10({name})",
                histtype="stepfilled",
            )
        if vline is not None:
            if xlabel == "log10(value)":
                ax.axvline(np.log10(max(vline, eps)), color="k", linestyle="--", linewidth=1.1, alpha=0.7, label="baseline")
            else:
                ax.axvline(vline, color="k", linestyle="--", linewidth=1.1, alpha=0.7, label="baseline")
        if logy:
            ax.set_yscale("log")
        if xlabel == "value" and xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.set_title("log-y" if logy else ("log10" if "log10" in xlabel else "linear"))
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_hist_logx(
    path: str,
    title: str,
    series: Dict[str, torch.Tensor],
    eps: float = 1e-12,
    bins: int = 80,
    density: bool = True,
) -> None:
    plt.figure(figsize=(8, 4.5))
    for name, x in series.items():
        y = torch.log10(torch.clamp(x, min=eps))
        plt.hist(
            to_np(y.reshape(-1)),
            bins=bins,
            density=density,
            alpha=0.55,
            label=f"log10({name})",
            histtype="stepfilled",
        )
    plt.title(title + f" (eps={eps:g})")
    plt.xlabel("log10(value)")
    plt.legend()
    savefig(path)


def plot_sorted_profiles_quantiles(
    path: str,
    title: str,
    x: torch.Tensor,
    qs: Tuple[float, float, float] = (0.1, 0.5, 0.9),
    logy: bool = False,
) -> None:
    """
    Instead of plotting 20 random rows, plot quantile bands of sorted profiles.
    """
    xs = torch.sort(x.detach(), dim=-1, descending=True).values  # (B,N)
    q_lo = torch.quantile(xs, qs[0], dim=0)
    q_md = torch.quantile(xs, qs[1], dim=0)
    q_hi = torch.quantile(xs, qs[2], dim=0)

    plt.figure(figsize=(8, 4.5))
    idx = torch.arange(xs.shape[-1]).cpu().numpy()
    plt.fill_between(idx, to_np(q_lo), to_np(q_hi), alpha=0.25, label=f"q{qs[0]:.1f}-q{qs[2]:.1f}")
    plt.plot(idx, to_np(q_md), label=f"median (q{qs[1]:.1f})")
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("sorted index")
    plt.ylabel("value")
    plt.legend()
    savefig(path)


def plot_sorted_profiles_panel(
    path: str,
    title: str,
    series: Dict[str, torch.Tensor],
    qs: Tuple[float, float, float] = (0.1, 0.5, 0.9),
    logy: bool = False,
    tol: float = 1e-12,
    include_support_hist: bool = False,
) -> None:
    """
    Overlay quantile bands for multiple series and (optionally) attach a support
    size histogram when comparing methods directly (e.g., QP vs softmax).
    """
    colors = cycle(plt.rcParams["axes.prop_cycle"].by_key().get("color", []))
    fig, axes = plt.subplots(1, 2 if include_support_hist else 1, figsize=(12.5 if include_support_hist else 7.5, 4.6))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    ax_profiles = axes[0]
    idx = None

    for name, x in series.items():
        xs = torch.sort(x.detach(), dim=-1, descending=True).values  # (B,N)
        if idx is None:
            idx = torch.arange(xs.shape[-1]).cpu().numpy()
        q_lo = torch.quantile(xs, qs[0], dim=0)
        q_md = torch.quantile(xs, qs[1], dim=0)
        q_hi = torch.quantile(xs, qs[2], dim=0)

        color = next(colors)
        ax_profiles.fill_between(idx, to_np(q_lo), to_np(q_hi), alpha=0.18, color=color, label=f"{name} q{qs[0]:.1f}-{qs[2]:.1f}")
        ax_profiles.plot(idx, to_np(q_md), color=color, linewidth=2.0, label=f"{name} median")

        if include_support_hist and len(axes) > 1:
            l0 = (x.detach() > tol).sum(dim=-1).cpu().numpy()
            bins = np.arange(-0.5, xs.shape[-1] + 1.5, 1.0)
            axes[1].hist(l0, bins=bins, alpha=0.55, label=name, rwidth=0.9)

    if logy:
        ax_profiles.set_yscale("log")
    ax_profiles.set_title("Sorted profile quantiles")
    ax_profiles.set_xlabel("sorted index")
    ax_profiles.set_ylabel("value")
    ax_profiles.legend()

    if include_support_hist and len(axes) > 1:
        axes[1].set_title("Support size distribution")
        axes[1].set_xlabel("support size (# nonzeros)")
        axes[1].set_ylabel("count")
        axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_support_size_hist(
    path: str,
    title: str,
    x: torch.Tensor,
    tol: float = 1e-12,
) -> None:
    l0 = (x.detach() > tol).sum(dim=-1).cpu().numpy()
    plt.figure(figsize=(8, 4.5))
    max_count = int(l0.max())
    bins = np.arange(-0.5, max_count + 1.5, 1.0)
    plt.hist(l0, bins=bins, alpha=0.7, rwidth=0.9)
    plt.title(title)
    plt.xlabel("support size (# nonzeros)")
    plt.ylabel("count")
    savefig(path)


def plot_topk_mass_curve(
    path: str,
    title: str,
    x: torch.Tensor,
    max_k: int = 50,
    total_mass: float | None = None,
) -> None:
    xs = torch.sort(x.detach(), dim=-1, descending=True).values  # (B,N)
    max_k = min(max_k, xs.shape[-1])
    cumsum = torch.cumsum(xs[:, :max_k], dim=-1)  # (B,k)
    mean_curve = cumsum.mean(dim=0).cpu().numpy()

    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(1, max_k + 1), mean_curve)
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("mean mass in top-k")
    if total_mass is not None:
        plt.axhline(total_mass, color="k", linestyle="--", linewidth=1.2, alpha=0.7, label="total mass")
        plt.legend()
    savefig(path)


def plot_support_and_topk_panel(
    path: str,
    title: str,
    series: Dict[str, torch.Tensor],
    xi: float,
    max_k: int = 50,
    tol: float = 1e-12,
) -> None:
    """Compact view: support-size histogram + mean top-k mass curves."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.4))
    any_x = next(iter(series.values()))
    B, N = any_x.shape
    bins = np.arange(-0.5, N + 1.5, 1.0)

    for name, x in series.items():
        l0 = (x.detach() > tol).sum(dim=-1).cpu().numpy()
        ax_l.hist(l0, bins=bins, alpha=0.6, label=name, rwidth=0.9)

    ax_l.set_xlabel(f"support size (# nonzeros out of N={N})")
    ax_l.set_ylabel("count")
    ax_l.set_title("Support size distribution")
    ax_l.legend()

    k_vals = np.arange(1, min(max_k, N) + 1)
    for name, x in series.items():
        xs = torch.sort(x.detach(), dim=-1, descending=True).values
        cumsum = torch.cumsum(xs[:, : len(k_vals)], dim=-1)
        mean_curve = cumsum.mean(dim=0).cpu().numpy()
        ax_r.plot(k_vals, mean_curve, label=name)

    ax_r.axhline(xi, color="k", linestyle="--", linewidth=1.0, alpha=0.7, label=r"target $\xi$")
    ax_r.set_xlabel("k (top-k)")
    ax_r.set_ylabel("mean mass in top-k")
    ax_r.set_title("Mean cumulative mass by top-k")
    ax_r.legend()

    fig.suptitle(rf"{title}\nB={B}, N={N}, sum target $\xi={xi}$")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def compare_to_high_iter_reference(
    proj_fast: SeparableQPProjection,
    proj_ref: SeparableQPProjection,
    z: torch.Tensor,
    name: str,
) -> Dict[str, float]:
    x_fast = proj_fast(z)
    x_ref = proj_ref(z)
    err = (x_fast - x_ref).abs()
    return {
        f"{name}_mean_abs_err": float(err.mean().item()),
        f"{name}_max_abs_err": float(err.max().item()),
    }


def plot_error_hist(
    path: str,
    title: str,
    err: torch.Tensor,
    logy: bool = True,
) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.hist(to_np(err.reshape(-1)), bins=80, density=True, alpha=0.7)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("|error|")
    plt.ylabel("density")
    savefig(path)


def plot_piecewise_with_free_count(
    path: str,
    title: str,
    z: torch.Tensor,
    n_xi: int = 240,
    m: float = 0.0,
    M: float = 1.0,
    gamma: float = 1.0,
    active_tol: float = 1e-7,
) -> None:
    """
    Sweep xi and plot:
      - a handful of x_i(xi)
      - number of free variables vs xi (shows active-set changes)
    """
    z = z.reshape(1, -1)
    N = z.shape[-1]
    xis = torch.linspace(m * N, M * N, n_xi)

    layer = SeparableQPProjection(
        n=N,
        xi=1.0,
        bounds_mode="fixed",
        m_init=m,
        M_init=M,
        gamma_init=gamma,
        learn_gamma=False,
        bisection_iters=80,
    )

    X = []
    free_counts = []
    for xi in xis:
        layer.xi_buf[:] = xi
        x = layer(z).detach().cpu().reshape(-1)
        X.append(x)
        free = (x > (m + active_tol)) & (x < (M - active_tol))
        free_counts.append(float(free.sum().item()))
    X = torch.stack(X, dim=0)  # (n_xi, N)
    free_counts = np.array(free_counts)

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    for i in range(min(N, 10)):
        ax1.plot(to_np(xis), to_np(X[:, i]), alpha=0.85, linewidth=1.3)
    ax1.set_xlabel(r"$\xi$")
    ax1.set_ylabel(r"$x_i$")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(to_np(xis), free_counts, linewidth=2.0, alpha=0.8)
    ax2.set_ylabel("# free variables")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def kkt_free_residual_stats(
    z: torch.Tensor,
    x: torch.Tensor,
    gamma: torch.Tensor,
    m: torch.Tensor,
    M: torch.Tensor,
    tol: float = 1e-7,
) -> Dict[str, float]:
    """
    For the solver convention x = clip(z + lambda/(2gamma)):
      r_i = 2gamma_i (x_i - z_i) should be constant (=lambda) on free set.
    We report variance of r on the free set as a sanity check.
    """
    two_gamma = 2.0 * gamma
    r = two_gamma * (x - z)
    free = (x > (m + tol)) & (x < (M - tol))
    # per-batch variance over free coordinates
    vars_ = []
    for b in range(x.shape[0]):
        if free[b].any():
            vars_.append(float(r[b][free[b]].var(unbiased=False).item()))
    if len(vars_) == 0:
        return {"mean_free_var_r": 0.0, "max_free_var_r": 0.0, "num_batches_with_free": 0.0}
    return {
        "mean_free_var_r": float(np.mean(vars_)),
        "max_free_var_r": float(np.max(vars_)),
        "num_batches_with_free": float(len(vars_)),
    }


# -----------------------------
# Demos
# -----------------------------

def demo_simplex_vs_softmax(cfg: DemoCfg) -> None:
    B, N = cfg.b, 100
    z = torch.randn(B, N, device=cfg.device, dtype=cfg.dtype)

    proj_fast = SeparableQPProjection(n=N, bisection_iters=40)
    proj_ref = SeparableQPProjection(n=N, bisection_iters=200)

    x_qp = proj_fast(z)
    x_qp_ref = proj_ref(z)
    x_sm = F.softmax(z, dim=-1)
    x_simplex_ref = simplex_projection_reference(z, s=1.0)

    # sanity + reference errors
    print("\n[Use case 1] Simplex-like: QP projection vs softmax")
    print_stats_table(
        f"Row-wise summary (B={B}, N={N}, target $\\xi=1$)",
        {"QP": row_stats(x_qp, eps=cfg.eps_logx), "Softmax": row_stats(x_sm, eps=cfg.eps_logx)},
    )
    print_stats_table(
        "Feasibility checks",
        {"QP": feasibility_stats(x_qp, 0.0, 1.0, 1.0)},
    )
    print_stats_table(
        "Projection error vs references",
        {
            "QP vs high-iter": compare_to_high_iter_reference(proj_fast, proj_ref, z, name="qp"),
            "QP vs simplex-sort": {
                "qp_mean_abs_err": float((x_qp - x_simplex_ref).abs().mean().item()),
                "qp_max_abs_err": float((x_qp - x_simplex_ref).abs().max().item()),
            },
        },
    )

    plot_value_panels(
        os.path.join(cfg.out_dir, "simplex_value_panels.png"),
        rf"QP simplex projection vs Softmax ($\sum_i x_i = 1$, B={B}, N={N})",
        {"QP": x_qp, "Softmax": x_sm},
        xlim=(0.0, 0.25),
        bins=120,
        eps=cfg.eps_logx,
        vline=0.0,
    )

    # sorted profile quantiles + support histogram (combined view)
    plot_sorted_profiles_panel(
        os.path.join(cfg.out_dir, "simplex_sorted_profiles_panel.png"),
        "Sorted profiles (QP vs Softmax)",
        {"QP": x_qp, "Softmax": x_sm},
        logy=True,
        include_support_hist=True,
    )

    # Support size + top-k mass combined panel
    plot_support_and_topk_panel(
        os.path.join(cfg.out_dir, "simplex_support_topk_panel.png"),
        "Simplex-like projection vs softmax",
        {"QP": x_qp, "Softmax": x_sm},
        xi=1.0,
        max_k=50,
    )

    # error to references
    err_hi = (x_qp - x_qp_ref).abs()
    err_sort = (x_qp - x_simplex_ref).abs()
    plot_error_hist(
        os.path.join(cfg.out_dir, "simplex_error_vs_high_iter.png"),
        "Error |x_fast - x_ref| (QP fast vs QP high-iter)",
        err_hi,
        logy=True,
    )
    plot_error_hist(
        os.path.join(cfg.out_dir, "simplex_error_vs_sort_ref.png"),
        "Error |x_qp - x_simplex_sort| (QP vs sorting reference)",
        err_sort,
        logy=True,
    )

    # piecewise behavior + free count
    plot_piecewise_with_free_count(
        os.path.join(cfg.out_dir, "simplex_piecewise_with_freecount.png"),
        "Piecewise x_i(xi) + #free (m=0, M=1, gamma=1)",
        z[0].detach().cpu(),
        n_xi=240,
        m=0.0,
        M=1.0,
        gamma=1.0,
    )


def demo_k_hot_budget(cfg: DemoCfg) -> None:
    B, N = cfg.b, 200
    k = 10.0
    z = torch.randn(B, N, device=cfg.device, dtype=cfg.dtype)

    proj_fast = SeparableQPProjection(n=N, xi=k, bounds_mode="fixed", m_init=0.0, M_init=1.0, bisection_iters=40)
    proj_ref = SeparableQPProjection(n=N, xi=k, bounds_mode="fixed", m_init=0.0, M_init=1.0, bisection_iters=220)

    x = proj_fast(z)
    x_ref = proj_ref(z)

    print("\n[Use case 2] k-hot relaxed gating: x in [0,1], sum x = k")
    print_stats_table(
        f"Row-wise summary (B={B}, N={N}, target $\\xi={k}$)",
        {"QP": row_stats(x, eps=cfg.eps_logx)},
    )
    print_stats_table(
        "Feasibility checks",
        {"QP": feasibility_stats(x, 0.0, 1.0, k)},
    )
    print_stats_table(
        "Fast vs high-iter reference",
        {"QP": compare_to_high_iter_reference(proj_fast, proj_ref, z, name="qp")},
    )

    plot_value_panels(
        os.path.join(cfg.out_dir, "khot_value_panels.png"),
        rf"k-hot relaxed gating ($\sum_i x_i = {k}$, B={B}, N={N})",
        {"QP": x},
        xlim=(0.0, 1.0),
        bins=140,
        eps=cfg.eps_logx,
    )

    # Support size + top-k mass
    plot_support_and_topk_panel(
        os.path.join(cfg.out_dir, "khot_support_topk_panel.png"),
        "k-hot relaxed gating",
        {"QP": x},
        xi=k,
        max_k=60,
    )

    # sorted quantiles
    plot_sorted_profiles_quantiles(
        os.path.join(cfg.out_dir, "khot_sorted_profiles_quantiles.png"),
        "Sorted profiles (k-hot relaxed): quantile band",
        x,
        logy=True,
    )

    # error vs ref
    err = (x - x_ref).abs()
    plot_error_hist(
        os.path.join(cfg.out_dir, "khot_error_vs_high_iter.png"),
        "Error |x_fast - x_ref| (k-hot fast vs high-iter)",
        err,
        logy=True,
    )


def demo_adaptive_xi(cfg: DemoCfg) -> None:
    B, N = cfg.b, 80
    z = torch.randn(B, N, device=cfg.device, dtype=cfg.dtype)

    proj = SeparableQPProjection(
        n=N,
        bounds_mode="fixed",
        m_init=0.0,
        M_init=1.0,
        learn_xi=True,
        xi_mode="feasible_sigmoid",
        bisection_iters=60,
    )

    # Sweep t and show resulting mean sum (this is the actual reference you want).
    ts = torch.linspace(-6, 6, 121)
    mean_sums = []
    mean_L0 = []
    for t in ts:
        with torch.no_grad():
            proj.t_raw[:] = torch.tensor([float(t.item())], dtype=torch.float32)
        x = proj(z)
        mean_sums.append(float(x.sum(dim=-1).mean().item()))
        mean_L0.append(float((x > 0).sum(dim=-1).float().mean().item()))

    print("\n[Use case 3] Adaptive xi (learnable; xi constrained to [sum m, sum M])")
    print("Interpretation: as t sweeps, xi moves smoothly from ~0 to ~N (since m=0, M=1).")

    plt.figure(figsize=(8, 4.5))
    plt.plot(to_np(ts), np.array(mean_sums), label="mean(sum x)")
    plt.plot(to_np(ts), np.array(mean_L0), label="mean(support size)")
    plt.title(rf"Adaptive $\xi$: effect of t on $\sum_i x_i$ and sparsity (B={B}, N={N})")
    plt.xlabel("t")
    plt.legend()
    savefig(os.path.join(cfg.out_dir, "adaptive_xi_curve.png"))

    # Also show distributions for a few t values with log-y
    t_values = [-4.0, 0.0, 4.0]
    series = {}
    for tv in t_values:
        with torch.no_grad():
            proj.t_raw[:] = torch.tensor([tv], dtype=torch.float32)
        series[f"t={tv:+.1f}"] = proj(z)

    plot_hist(
        os.path.join(cfg.out_dir, "adaptive_xi_value_hist_logy.png"),
        rf"Adaptive $\xi$: value distributions (log-y, N={N})",
        {k: v for k, v in series.items()},
        bins=120,
        density=True,
        logy=True,
        xlim=(0.0, 1.0),
        xlabel=r"projected value $x_i$",
    )


def demo_learnable_bounds_unit(cfg: DemoCfg) -> None:
    """
    Toy: learn bounds (unit mode) so projected outputs match a target.
    """
    set_seed(1)
    B, N = 128, 30
    z = torch.randn(B, N, device=cfg.device, dtype=cfg.dtype)

    target_layer = SeparableQPProjection(n=N, xi=1.0, bounds_mode="fixed", m_init=0.0, M_init=1.0, bisection_iters=120)
    x_target = target_layer(z + 0.7).detach()

    proj = SeparableQPProjection(
        n=N,
        xi=1.0,
        bounds_mode="unit",
        learn_bounds=True,
        clamp_xi_to_feasible=True,
        bisection_iters=60,
    )

    with torch.no_grad():
        m0, M0 = proj.bounds(dtype=cfg.dtype, device=torch.device(cfg.device))

    opt = torch.optim.Adam(proj.parameters(), lr=3e-2)
    losses = []
    for _ in range(250):
        opt.zero_grad(set_to_none=True)
        x = proj(z)
        loss = F.mse_loss(x, x_target)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        m1, M1 = proj.bounds(dtype=cfg.dtype, device=torch.device(cfg.device))
        x_final = proj(z)
        err = (x_final - x_target).abs()

    print("\n[Use case 4] Learnable bounds (unit) + fit-to-target toy training")
    print_stats_table(
        "Loss trajectory",
        {"MSE": {"final_loss": losses[-1], "min_loss": min(losses)}},
    )
    print_stats_table(
        "Bounds summary (mean values)",
        {
            "initial": {"m_mean": float(m0.mean().item()), "M_mean": float(M0.mean().item()), "width_mean": float((M0 - m0).mean().item())},
            "final": {"m_mean": float(m1.mean().item()), "M_mean": float(M1.mean().item()), "width_mean": float((M1 - m1).mean().item())},
        },
    )
    print_stats_table(
        "Final feasibility and error",
        {
            "x_final": feasibility_stats(x_final, 0.0, 1.0, 1.0)
            | {"abs_err_mean": float(err.mean().item()), "abs_err_max": float(err.max().item())}
        },
    )

    # loss curve (log-y helps)
    plt.figure(figsize=(8, 4.5))
    plt.plot(losses, label="MSE")
    plt.yscale("log")
    plt.title("Learning bounds: MSE loss curve (log-y)")
    plt.xlabel("step")
    plt.legend()
    savefig(os.path.join(cfg.out_dir, "learn_bounds_loss_curve_logy.png"))

    # bounds hist: initial vs final + widths
    plot_hist(
        os.path.join(cfg.out_dir, "learned_bounds_m_compare.png"),
        "Bounds m: initial vs final (density)",
        {"m_init": m0, "m_final": m1},
        bins=50,
        density=True,
        logy=False,
        xlim=None,
        rug=True,
    )
    plot_hist(
        os.path.join(cfg.out_dir, "learned_bounds_M_compare.png"),
        "Bounds M: initial vs final (density)",
        {"M_init": M0, "M_final": M1},
        bins=50,
        density=True,
        logy=False,
        xlim=None,
        rug=True,
    )
    plot_hist(
        os.path.join(cfg.out_dir, "learned_bounds_width_compare_logy.png"),
        "Width (M-m): initial vs final (log-y density)",
        {"width_init": (M0 - m0), "width_final": (M1 - m1)},
        bins=60,
        density=True,
        logy=True,
        xlim=(0.0, 1.0),
    )

    # output vs target error
    plot_hist(
        os.path.join(cfg.out_dir, "learn_bounds_abs_error_logy.png"),
        "|x - target| distribution (log-y density)",
        {"abs_err": err},
        bins=120,
        density=True,
        logy=True,
    )

    # sorted profiles quantiles: target vs final
    plot_sorted_profiles_quantiles(
        os.path.join(cfg.out_dir, "learn_bounds_sorted_profiles_target.png"),
        "Target sorted profile: quantile band",
        x_target,
        logy=True,
    )
    plot_sorted_profiles_quantiles(
        os.path.join(cfg.out_dir, "learn_bounds_sorted_profiles_final.png"),
        "Final sorted profile: quantile band",
        x_final,
        logy=True,
    )


def demo_gamma_effect(cfg: DemoCfg) -> None:
    set_seed(2)
    B, N = 256, 60
    z = torch.randn(B, N, device=cfg.device, dtype=cfg.dtype)

    # Uniform gamma
    proj_uniform = SeparableQPProjection(n=N, xi=1.0, bounds_mode="fixed", m_init=0.0, M_init=1.0, gamma_init=1.0)

    # Custom gamma: small -> more willing to move; large -> stiffer
    gamma_vec = torch.ones(N)
    gamma_vec[: N // 3] = 0.2
    gamma_vec[N // 3 : 2 * N // 3] = 1.0
    gamma_vec[2 * N // 3 :] = 5.0

    proj_custom = SeparableQPProjection(
        n=N,
        xi=1.0,
        bounds_mode="fixed",
        m_init=0.0,
        M_init=1.0,
        gamma_init=gamma_vec,
        learn_gamma=False,
        gamma_per_dim=True,
    )

    x_u = proj_uniform(z)
    x_c = proj_custom(z)

    print("\n[Use case 5] Gamma (curvature) effect")
    print_stats_table(
        f"Row-wise summary (B={B}, N={N}, target $\\xi=1$)",
        {"uniform": row_stats(x_u, eps=cfg.eps_logx), "custom": row_stats(x_c, eps=cfg.eps_logx)},
    )

    # Group mass distributions (three subplots, clearer than overlaying)
    def group_sums(x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                x[:, : N // 3].sum(dim=-1),
                x[:, N // 3 : 2 * N // 3].sum(dim=-1),
                x[:, 2 * N // 3 :].sum(dim=-1),
            ],
            dim=-1,
        )

    gs_u = group_sums(x_u).detach().cpu()
    gs_c = group_sums(x_c).detach().cpu()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for i, ax in enumerate(axes):
        ax.hist(to_np(gs_u[:, i]), bins=50, density=True, alpha=0.55, label="uniform")
        ax.hist(to_np(gs_c[:, i]), bins=50, density=True, alpha=0.55, label="custom")
        ax.set_title(f"group {i+1} mass")
        ax.set_xlim(0, 1.0)
        ax.legend()
    fig.suptitle("Gamma effect: distribution of mass by coordinate group")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, "gamma_group_mass_compare.png"), dpi=160)
    plt.close(fig)

    # Support size comparison
    plot_support_size_hist(
        os.path.join(cfg.out_dir, "gamma_support_size_uniform.png"),
        "Support size (uniform gamma)",
        x_u,
    )
    plot_support_size_hist(
        os.path.join(cfg.out_dir, "gamma_support_size_custom.png"),
        "Support size (custom gamma)",
        x_c,
    )


def demo_solve_from_beta(cfg: DemoCfg) -> None:
    """
    Solve directly from beta coefficients and compare to a high-iteration reference.
    """
    set_seed(3)
    B, N = 128, 40
    beta = torch.randn(B, N, device=cfg.device, dtype=cfg.dtype)

    proj_fast = SeparableQPProjection(n=N, xi=1.0, bounds_mode="fixed", m_init=0.0, M_init=1.0, bisection_iters=40)
    proj_ref = SeparableQPProjection(n=N, xi=1.0, bounds_mode="fixed", m_init=0.0, M_init=1.0, bisection_iters=220)

    solver_fast = SeparableQPSolveFromBeta(projection=proj_fast)
    solver_ref = SeparableQPSolveFromBeta(projection=proj_ref)

    x = solver_fast(beta)
    x_ref = solver_ref(beta)

    # Recover z and gamma used internally (gamma=1 by default here)
    z = -beta / 2.0
    gamma = torch.ones_like(beta)
    m = torch.zeros_like(beta)
    M = torch.ones_like(beta)

    print("\n[Use case 6] Solve from beta (z=-beta/(2gamma))")
    print_stats_table(
        f"Row-wise summary (B={B}, N={N}, target $\\xi=1$)",
        {"x": row_stats(x, eps=cfg.eps_logx)},
    )
    print_stats_table(
        "Feasibility checks",
        {"x": feasibility_stats(x, 0.0, 1.0, 1.0)},
    )
    print_stats_table(
        "Fast vs high-iter reference",
        {"x": compare_to_high_iter_reference(proj_fast, proj_ref, z, name="x")},
    )
    print_stats_table(
        "KKT free-residual variance",
        {"free_set": kkt_free_residual_stats(z=z, x=x, gamma=gamma, m=m, M=M)},
    )

    plot_value_panels(
        os.path.join(cfg.out_dir, "solve_from_beta_value_panels.png"),
        rf"Solve-from-beta ($\sum_i x_i = 1$, B={B}, N={N})",
        {"x": x},
        xlim=(0.0, 1.0),
        bins=120,
        eps=cfg.eps_logx,
    )
    # error vs reference
    err = (x - x_ref).abs()
    plot_error_hist(
        os.path.join(cfg.out_dir, "solve_from_beta_error_vs_high_iter.png"),
        "Solve-from-beta error |x_fast - x_ref| (log-y density)",
        err,
        logy=True,
    )


def demo_fixed_z(cfg: DemoCfg) -> None:
    set_seed(4)
    N = 50
    z_const = torch.linspace(-1.5, 1.5, N)

    proj = SeparableQPProjection(n=N, xi=1.0, bounds_mode="fixed", m_init=0.0, M_init=1.0, bisection_iters=80)
    fixed_layer = FixedZProjection(z_const=z_const, projection=proj)

    x = fixed_layer(batch_size=cfg.b, device=torch.device(cfg.device), dtype=cfg.dtype)

    print("\n[Use case 7] Fixed z -> deterministic projected output")
    print("Stats:", row_stats(x, eps=cfg.eps_logx))
    print("Feasibility:", feasibility_stats(x, 0.0, 1.0, 1.0))

    plot_sorted_profiles_quantiles(
        os.path.join(cfg.out_dir, "fixed_z_sorted_profiles_quantiles.png"),
        "Fixed z: sorted profile quantile band",
        x,
        logy=True,
    )
    plot_hist(
        os.path.join(cfg.out_dir, "fixed_z_hist_logy.png"),
        "Fixed z: value distribution (log-y density)",
        {"x": x},
        bins=120,
        density=True,
        logy=True,
        xlim=(0.0, 1.0),
    )


def demo_geometry_n2(cfg: DemoCfg) -> None:
    """
    Keep your existing contour idea, but add the unconstrained minimizer z
    and the projected solution, plus the feasible segment.
    """
    set_seed(5)
    z = torch.tensor([0.6, -0.2], dtype=torch.float32)
    gamma = torch.tensor([0.7, 2.0], dtype=torch.float32)
    m = torch.tensor([-0.3, -0.1], dtype=torch.float32)
    M = torch.tensor([0.8, 0.6], dtype=torch.float32)
    xi = 0.5

    # grid
    x1 = torch.linspace(float(m[0]), float(M[0]), 260)
    x2 = torch.linspace(float(m[1]), float(M[1]), 260)
    X1, X2 = torch.meshgrid(x1, x2, indexing="xy")
    Fval = gamma[0] * (X1 - z[0]) ** 2 + gamma[1] * (X2 - z[1]) ** 2

    layer = SeparableQPProjection(
        n=2,
        xi=xi,
        bounds_mode="fixed",
        m_init=m.numpy(),
        M_init=M.numpy(),
        gamma_init=gamma.numpy(),
        learn_gamma=False,
        bisection_iters=120,
    )
    x_star = layer(z.reshape(1, 2)).detach().cpu().reshape(-1)

    # feasible segment
    x1_seg = torch.linspace(float(m[0]), float(M[0]), 600)
    x2_seg = xi - x1_seg
    mask = (x2_seg >= m[1]) & (x2_seg <= M[1])

    plt.figure(figsize=(7.2, 6.0))
    plt.contour(to_np(X1), to_np(X2), to_np(Fval), levels=25)
    plt.plot(
        to_np(x1_seg[mask]),
        to_np(x2_seg[mask]),
        linewidth=2.3,
        label=r"feasible ($\sum_i x_i = \xi$)",
    )
    plt.scatter([z[0].item()], [z[1].item()], marker="x", s=80, label="unconstrained minimizer z")
    plt.scatter([x_star[0].item()], [x_star[1].item()], s=80, label="projected solution x*")
    plt.title(r"N=2 geometry: contours + feasible $\sum_i x_i = \xi$ + solution")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.legend()
    savefig(os.path.join(cfg.out_dir, "contour_n2_geometry.png"))

    print("\n[Geometry] N=2 contour demo saved.")


def main() -> None:
    cfg = DemoCfg()
    set_seed(0)
    ensure_out_dir(cfg.out_dir)

    # optional: nicer defaults
    plt.rcParams.update({
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
    })

    demo_simplex_vs_softmax(cfg)
    demo_k_hot_budget(cfg)
    demo_adaptive_xi(cfg)
    demo_learnable_bounds_unit(cfg)
    demo_gamma_effect(cfg)
    demo_solve_from_beta(cfg)
    demo_fixed_z(cfg)
    demo_geometry_n2(cfg)

    print(f"\nSaved plots to: {os.path.abspath(cfg.out_dir)}")


if __name__ == "__main__":
    main()
