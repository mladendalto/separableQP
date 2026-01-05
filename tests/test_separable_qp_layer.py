from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch

from separable_qp_layer import (
    BetaToZ,
    BoundsParam,
    Epsilons,
    FixedZProjection,
    GammaParam,
    SeparableQPProjection,
    SeparableQPSolveFromBeta,
    _SeparableQProjectionFn,
)


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def proj_simplex_reference(v: torch.Tensor, s: float = 1.0) -> torch.Tensor:
    """
    Reference Euclidean projection onto simplex {x>=0, sum x = s}.
    Duchi et al. style sorting algorithm. Works on last dim.
    """
    if v.dim() == 1:
        v = v.unsqueeze(0)
    B, N = v.shape
    u, _ = torch.sort(v, dim=-1, descending=True)
    cssv = torch.cumsum(u, dim=-1) - s
    ind = torch.arange(1, N + 1, device=v.device, dtype=v.dtype).view(1, -1)
    cond = u - cssv / ind > 0
    # rho = max {j: cond true}
    rho = cond.to(torch.int64).sum(dim=-1) - 1  # (B,)
    rho = torch.clamp(rho, min=0)
    theta = cssv[torch.arange(B, device=v.device), rho] / (rho.to(v.dtype) + 1.0)
    w = torch.clamp(v - theta.unsqueeze(-1), min=0.0)
    return w


def kkt_check(
    z: torch.Tensor,
    gamma: torch.Tensor,
    m: torch.Tensor,
    M: torch.Tensor,
    xi: torch.Tensor,
    x: torch.Tensor,
    tol: float = 1e-6,
) -> None:
    """
    Check KKT conditions for the weighted projection form:
        min sum_i gamma_i (x_i - z_i)^2  s.t. sum x = xi, m<=x<=M

    With the solver convention x = clip(z + lambda/(2*gamma), [m,M]):
      - free:   2*gamma*(x - z) == lambda
      - lower:  2*gamma*(x - z) >= lambda
      - upper:  2*gamma*(x - z) <= lambda
    """
    B, N = x.shape
    two_gamma = 2.0 * gamma

    r = two_gamma * (x - z)  # should equal lambda on free set
    free = (x > m + 1e-7) & (x < M - 1e-7)

    lam_hat = torch.zeros(B, device=x.device, dtype=x.dtype)
    for b in range(B):
        if free[b].any():
            lam_hat[b] = r[b][free[b]].mean()
        else:
            # all clamped; lambda not identifiable, but inequalities should hold for any subgradient
            lam_hat[b] = r[b].median()

    # Primal feasibility
    assert torch.all(x >= m - tol)
    assert torch.all(x <= M + tol)
    assert torch.allclose(x.sum(dim=-1), xi.reshape(B), rtol=0.0, atol=1e-4)

    # Stationarity on free set
    if free.any():
        diff = (r[free] - lam_hat.unsqueeze(-1).expand_as(r)[free]).abs().max().item()
        assert diff < 5e-4

    # Correct inequality directions for this lambda convention:
    lower = x <= (m + 1e-7)
    if lower.any():
        assert torch.all(
            r[lower] >= lam_hat.unsqueeze(-1).expand_as(r)[lower] - 5e-4
        )

    upper = x >= (M - 1e-7)
    if upper.any():
        assert torch.all(
            r[upper] <= lam_hat.unsqueeze(-1).expand_as(r)[upper] + 5e-4
        )



def analytic_free_solution(
    z: torch.Tensor, gamma: torch.Tensor, xi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    If bounds are wide so all variables are free:
      x = z + lambda/(2gamma)
      lambda = 2*(xi - sum z) / sum(1/gamma)
    """
    B, N = z.shape
    inv_gamma = 1.0 / gamma
    denom = inv_gamma.sum(dim=-1)  # (B,)
    lam = 2.0 * (xi.reshape(B) - z.sum(dim=-1)) / denom
    x = z + lam.unsqueeze(-1) / (2.0 * gamma)
    return x, lam


# -----------------------
# Parameterization tests
# -----------------------

def test_gamma_param_fixed_positive() -> None:
    set_seed(0)
    n = 7
    gp = GammaParam(n=n, init=0.5, learnable=False, per_dim=True, eps=1e-6)
    g = gp(dtype=torch.float32, device=torch.device("cpu"))
    assert g.shape == (n,)
    assert torch.all(g > 0)


def test_gamma_param_learnable_positive() -> None:
    set_seed(0)
    n = 5
    gp = GammaParam(n=n, init=1.2, learnable=True, per_dim=True, eps=1e-6)
    g = gp(dtype=torch.float64, device=torch.device("cpu"))
    assert g.shape == (n,)
    assert torch.all(g > 0)


def test_bounds_param_fixed() -> None:
    set_seed(0)
    n = 4
    bp = BoundsParam(n=n, mode="fixed", m_init=-1.0, M_init=2.0, learnable=False, eps_width=1e-6)
    m, M = bp(dtype=torch.float32, device=torch.device("cpu"))
    assert m.shape == (n,)
    assert M.shape == (n,)
    assert torch.all(M > m)


def test_bounds_param_free_enforces_width() -> None:
    set_seed(0)
    n = 6
    bp = BoundsParam(n=n, mode="free", m_init=-0.2, M_init=0.1, learnable=True, eps_width=1e-4)
    m, M = bp(dtype=torch.float32, device=torch.device("cpu"))
    assert torch.all(M > m + 1e-4 * 0.9)


def test_bounds_param_unit_in_0_1() -> None:
    set_seed(0)
    n = 8
    bp = BoundsParam(n=n, mode="unit", m_init=0.2, M_init=0.7, learnable=True, eps_width=1e-5)
    m, M = bp(dtype=torch.float32, device=torch.device("cpu"))
    assert torch.all(m >= 0)
    assert torch.all(M <= 1.0 + 1e-6)
    assert torch.all(M > m)


# -----------------------
# Core solver correctness
# -----------------------

def test_projection_feasibility_default_simplex() -> None:
    set_seed(0)
    B, N = 16, 20
    z = torch.randn(B, N)
    layer = SeparableQPProjection(n=N)  # m=0, M=1, xi=1, gamma=1
    x = layer(z)
    assert x.shape == (B, N)
    assert torch.all(x >= -1e-6)
    assert torch.all(x <= 1.0 + 1e-6)
    assert torch.allclose(x.sum(dim=-1), torch.ones(B), atol=2e-4, rtol=0.0)


def test_projection_matches_simplex_reference_when_caps_irrelevant() -> None:
    set_seed(0)
    B, N = 8, 50
    z = torch.randn(B, N)
    layer = SeparableQPProjection(n=N)  # simplex-like
    x = layer(z)
    x_ref = proj_simplex_reference(z, s=1.0)
    assert torch.allclose(x, x_ref, atol=2e-4, rtol=0.0)


def test_projection_kkt_conditions_random_box() -> None:
    set_seed(0)
    B, N = 8, 30
    z = torch.randn(B, N)
    gamma = torch.rand(B, N) + 0.5
    m = -0.2 * torch.ones(B, N)
    M = 0.3 * torch.ones(B, N)
    xi = torch.full((B,), 0.5 * N * (0.3 - 0.2))  # within [sum m, sum M]
    x = _SeparableQProjectionFn.apply(z, gamma, m, M, xi, 60, 1e-8)
    kkt_check(z, gamma, m, M, xi, x)


def test_projection_extreme_xi_all_lower_or_upper() -> None:
    set_seed(0)
    B, N = 4, 10
    z = torch.randn(B, N)
    gamma = torch.ones(B, N)
    m = -0.3 * torch.ones(B, N)
    M = 0.7 * torch.ones(B, N)

    xi_low = m.sum(dim=-1)  # should give x=m
    x_low = _SeparableQProjectionFn.apply(z, gamma, m, M, xi_low, 40, 1e-8)
    assert torch.allclose(x_low, m, atol=3e-4, rtol=0.0)

    xi_high = M.sum(dim=-1)  # should give x=M
    x_high = _SeparableQProjectionFn.apply(z, gamma, m, M, xi_high, 40, 1e-8)
    assert torch.allclose(x_high, M, atol=3e-4, rtol=0.0)


def test_free_solution_matches_analytic_when_bounds_wide() -> None:
    set_seed(0)
    B, N = 6, 17
    z = torch.randn(B, N, dtype=torch.float64)
    gamma = (torch.rand(B, N, dtype=torch.float64) + 0.5)
    m = -100.0 * torch.ones(B, N, dtype=torch.float64)
    M = 100.0 * torch.ones(B, N, dtype=torch.float64)
    xi = torch.randn(B, dtype=torch.float64)

    x = _SeparableQProjectionFn.apply(z, gamma, m, M, xi, 60, 1e-10)
    x_ref, _ = analytic_free_solution(z, gamma, xi)

    assert torch.allclose(x, x_ref, atol=1e-9, rtol=0.0)


# -----------------------
# Autograd / gradcheck
# -----------------------

def test_gradcheck_core_function_all_free() -> None:
    set_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64

    B, N = 2, 7
    z = torch.randn(B, N, device=device, dtype=dtype, requires_grad=True)
    gamma = (torch.rand(B, N, device=device, dtype=dtype) + 0.7).requires_grad_(True)
    m = (-100.0 * torch.ones(B, N, device=device, dtype=dtype)).requires_grad_(True)
    M = (100.0 * torch.ones(B, N, device=device, dtype=dtype)).requires_grad_(True)
    xi = (torch.randn(B, device=device, dtype=dtype)).requires_grad_(True)

    def fn(z_, gamma_, m_, M_, xi_):
        return _SeparableQProjectionFn.apply(z_, gamma_, m_, M_, xi_, 80, 1e-12)

    assert torch.autograd.gradcheck(fn, (z, gamma, m, M, xi), eps=1e-6, atol=1e-4, rtol=1e-4)


def test_backward_shapes_and_finiteness_with_clamps() -> None:
    set_seed(0)
    B, N = 4, 25
    z = torch.randn(B, N, requires_grad=True)
    layer = SeparableQPProjection(n=N, xi=5.0, bounds_mode="fixed", m_init=0.0, M_init=1.0)
    x = layer(z)
    loss = (x**2).sum()
    loss.backward()
    assert z.grad is not None
    assert z.grad.shape == z.shape
    assert torch.isfinite(z.grad).all()


# -----------------------
# Wrapper layers
# -----------------------

def test_beta_to_z_matches_formula() -> None:
    set_seed(0)
    beta = torch.randn(3, 5)
    gamma = torch.rand(3, 5) + 0.5
    b2z = BetaToZ(eps_gamma=1e-6)
    z = b2z(beta, gamma)
    assert torch.allclose(z, -beta / (2.0 * gamma), atol=0.0, rtol=0.0)


def test_solve_from_beta_matches_projection_on_z() -> None:
    set_seed(0)
    B, N = 5, 13
    beta = torch.randn(B, N)
    proj = SeparableQPProjection(n=N, xi=1.0, bounds_mode="fixed", m_init=0.0, M_init=1.0)
    solver = SeparableQPSolveFromBeta(projection=proj)

    # Use proj's internal gamma=1 => z = -beta/2
    x1 = solver(beta)
    x2 = proj(-beta / 2.0)
    assert torch.allclose(x1, x2, atol=2e-4, rtol=0.0)


def test_fixed_z_projection_is_constant() -> None:
    set_seed(0)
    N = 10
    z_const = torch.linspace(-1.0, 1.0, N)
    proj = SeparableQPProjection(n=N, xi=1.0, bounds_mode="fixed", m_init=0.0, M_init=1.0)
    fixed = FixedZProjection(z_const=z_const, projection=proj)

    x1 = fixed(batch_size=3)
    x2 = fixed(batch_size=3)
    assert torch.allclose(x1, x2, atol=0.0, rtol=0.0)
    assert torch.allclose(x1.sum(dim=-1), torch.ones(3), atol=2e-4, rtol=0.0)
