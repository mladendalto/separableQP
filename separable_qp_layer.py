from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorLike = Union[float, int, torch.Tensor]


# -----------------------------
# Parameterizations
# -----------------------------

@dataclass(frozen=True)
class Epsilons:
    """Separate epsilons for each constrained transform."""
    gamma: float = 1e-6        # ensures gamma > 0
    width: float = 1e-6        # ensures M - m > 0
    active_set: float = 1e-7   # tolerance for determining free variables


class GammaParam(nn.Module):
    """Gamma parameterization ensuring gamma > 0."""

    def __init__(
        self,
        n: int,
        init: TensorLike = 1.0,
        learnable: bool = False,
        per_dim: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.eps = float(eps)
        self.learnable = bool(learnable)
        self.per_dim = bool(per_dim)

        init_t = torch.as_tensor(init, dtype=torch.float32)
        if self.per_dim:
            init_t = init_t.expand(self.n).clone()
        else:
            init_t = init_t.reshape(1).clone()

        if learnable:
            # Inverse-softplus init: softplus(raw) ~= init - eps
            target = torch.clamp(init_t - self.eps, min=1e-12)
            raw = torch.log(torch.expm1(target))
            self.raw = nn.Parameter(raw)
        else:
            self.register_buffer("raw", init_t)

    def forward(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        raw = self.raw.to(dtype=dtype, device=device)
        if self.learnable:
            return self.eps + F.softplus(raw)
        # If fixed, interpret raw as already-positive gamma.
        return torch.clamp(raw, min=self.eps)


class BoundsParam(nn.Module):
    """
    Bounds parameterization for m and M with modes:
    - fixed: m, M provided (buffers)
    - free:  m unconstrained, M = m + eps + softplus(width_raw)
    - unit:  m in (0,1-eps), M in (m,1) via nested sigmoids
    """

    def __init__(
        self,
        n: int,
        mode: Literal["fixed", "free", "unit"] = "fixed",
        m_init: TensorLike = 0.0,
        M_init: TensorLike = 1.0,
        learnable: bool = False,
        eps_width: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.mode = mode
        self.learnable = bool(learnable)
        self.eps_width = float(eps_width)

        m0 = torch.as_tensor(m_init, dtype=torch.float32).expand(self.n).clone()
        M0 = torch.as_tensor(M_init, dtype=torch.float32).expand(self.n).clone()

        if mode == "fixed":
            self.register_buffer("m_buf", m0)
            self.register_buffer("M_buf", M0)
        elif mode == "free":
            # m is free; M = m + eps + softplus(width_raw)
            if learnable:
                self.m_raw = nn.Parameter(m0)
                width0 = torch.clamp(M0 - m0 - self.eps_width, min=1e-12)
                w_raw0 = torch.log(torch.expm1(width0))
                self.w_raw = nn.Parameter(w_raw0)
            else:
                self.register_buffer("m_raw", m0)
                width0 = torch.clamp(M0 - m0, min=self.eps_width)
                self.register_buffer("w_raw", width0)
        elif mode == "unit":
            # m = sigmoid(a) in (0,1), M = m + (1-m)*sigmoid(b)
            # Initialize a,b from m0,M0 (clipped to (0,1)).
            m0c = torch.clamp(m0, 1e-4, 1 - 1e-4)
            M0c = torch.clamp(M0, 1e-4, 1 - 1e-4)
            a0 = torch.log(m0c) - torch.log1p(-m0c)

            frac = torch.clamp((M0c - m0c) / (1 - m0c + 1e-12), 1e-4, 1 - 1e-4)
            b0 = torch.log(frac) - torch.log1p(-frac)

            if learnable:
                self.a_raw = nn.Parameter(a0)
                self.b_raw = nn.Parameter(b0)
            else:
                self.register_buffer("a_raw", a0)
                self.register_buffer("b_raw", b0)
        else:
            raise ValueError(f"Unknown bounds mode: {mode}")

    def forward(self, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "fixed":
            m = self.m_buf.to(dtype=dtype, device=device)
            M = self.M_buf.to(dtype=dtype, device=device)
            return m, M

        if self.mode == "free":
            m_raw = self.m_raw.to(dtype=dtype, device=device) if self.learnable else self.m_raw.to(dtype=dtype, device=device)
            w_raw = self.w_raw.to(dtype=dtype, device=device)
            if self.learnable:
                width = self.eps_width + F.softplus(w_raw)
            else:
                width = torch.clamp(w_raw, min=self.eps_width)
            m = m_raw
            M = m_raw + width
            return m, M

        # unit
        a = self.a_raw.to(dtype=dtype, device=device)
        b = self.b_raw.to(dtype=dtype, device=device)

        m = torch.sigmoid(a)
        # ensure there is always room for M > m numerically:
        m = torch.clamp(m, max=1.0 - self.eps_width)

        M = m + (1.0 - m) * torch.sigmoid(b)
        # strictness
        M = torch.maximum(M, m + self.eps_width)
        M = torch.clamp(M, max=1.0)
        return m, M


# -----------------------------
# Core autograd Function
# -----------------------------

class _SeparableQProjectionFn(torch.autograd.Function):
    """
    Project z onto {x: sum x = xi, m <= x <= M} under weighted L2 with weights gamma:
        x = argmin_x sum_i gamma_i (x_i - z_i)^2
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        z: torch.Tensor,       # (B,N)
        gamma: torch.Tensor,   # (B,N), gamma>0
        m: torch.Tensor,       # (B,N)
        M: torch.Tensor,       # (B,N)
        xi: torch.Tensor,      # (B,)
        bisection_iters: int,
        active_tol: float,
    ) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"z must be (B,N), got {tuple(z.shape)}")
        B, N = z.shape
        if gamma.shape != (B, N) or m.shape != (B, N) or M.shape != (B, N):
            raise ValueError("gamma, m, M must all be (B,N) after broadcasting.")
        if xi.shape not in [(B,), (B, 1)]:
            raise ValueError("xi must be (B,) or (B,1) after broadcasting.")
        xi = xi.reshape(B)

        if torch.any(M <= m):
            raise ValueError("All bounds must satisfy M > m.")

        two_gamma = 2.0 * gamma

        # Bracket lambda using z-form:
        # x_i=m_i when lambda <= 2*gamma_i*(m_i - z_i)
        # x_i=M_i when lambda >= 2*gamma_i*(M_i - z_i)
        lam_low = torch.min(two_gamma * (m - z), dim=-1).values
        lam_high = torch.max(two_gamma * (M - z), dim=-1).values

        # Bisection: g(lambda) = sum clip(z + lambda/(2gamma), [m,M]) - xi
        for _ in range(int(bisection_iters)):
            lam_mid = 0.5 * (lam_low + lam_high)
            x_mid = torch.clamp(
                z + lam_mid.unsqueeze(-1) / two_gamma,
                min=m,
                max=M,
            )
            g_mid = x_mid.sum(dim=-1) - xi
            # g increasing in lambda: if g_mid > 0, lambda too big -> move high down
            lam_high = torch.where(g_mid > 0, lam_mid, lam_high)
            lam_low = torch.where(g_mid > 0, lam_low, lam_mid)

        lam = 0.5 * (lam_low + lam_high)
        x = torch.clamp(z + lam.unsqueeze(-1) / two_gamma, min=m, max=M)

        # Active set masks
        free = (x > (m + active_tol)) & (x < (M - active_tol))
        lower = x <= (m + active_tol)
        upper = x >= (M - active_tol)

        s = free.to(dtype=z.dtype) / two_gamma  # s_i = 1/(2gamma_i) on free set else 0
        S = s.sum(dim=-1)  # (B,)

        ctx.save_for_backward(free, lower, upper, s, S, gamma, x, m, M, lam)
        return x

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        free, lower, upper, s, S, gamma, x, m, M, lam = ctx.saved_tensors
        _ = x, m, M  # kept for clarity; not used in formulas below

        # Safe divide for degenerate case S=0 (all clamped): treat coupled gradients as zero.
        S_safe = torch.where(S > 0, S, torch.ones_like(S))
        # c = (sum_{free} grad_out_i * s_i) / (sum_{free} s_i)
        c = (grad_out * s).sum(dim=-1) / S_safe  # (B,)
        c = torch.where(S > 0, c, torch.zeros_like(c))

        # Grad w.r.t z: free*(g_i - c)
        grad_z = free.to(grad_out.dtype) * (grad_out - c.unsqueeze(-1))

        # Grad w.r.t xi: c
        grad_xi = c  # (B,)

        # Grad w.r.t gamma on free set:
        # dL/dgamma_i = (lambda/(2*gamma_i^2)) * (c - g_i) for i in free
        grad_gamma = free.to(grad_out.dtype) * (lam.unsqueeze(-1) / (2.0 * gamma**2)) * (c.unsqueeze(-1) - grad_out)

        # Grad w.r.t bounds for active constraints:
        # if x_i=m_i (lower active): dL/dm_i = g_i - c
        # if x_i=M_i (upper active): dL/dM_i = g_i - c
        grad_m = lower.to(grad_out.dtype) * (grad_out - c.unsqueeze(-1))
        grad_M = upper.to(grad_out.dtype) * (grad_out - c.unsqueeze(-1))

        # Non-tensor args: bisection_iters, active_tol
        return grad_z, grad_gamma, grad_m, grad_M, grad_xi, None, None


# -----------------------------
# User-facing layers
# -----------------------------

class SeparableQPProjection(nn.Module):
    """
    Weighted projection layer for:
        x = argmin_x sum_i gamma_i (x_i - z_i)^2
            s.t. sum_i x_i = xi,  m_i <= x_i <= M_i

    Defaults are sensible for simplex-like activations:
        m=0, M=1, xi=1, gamma=1
    """

    def __init__(
        self,
        n: int,
        *,
        xi: TensorLike = 1.0,
        learn_xi: bool = False,
        xi_mode: Literal["fixed", "feasible_sigmoid"] = "fixed",
        bounds_mode: Literal["fixed", "free", "unit"] = "fixed",
        m_init: TensorLike = 0.0,
        M_init: TensorLike = 1.0,
        learn_bounds: bool = False,
        gamma_init: TensorLike = 1.0,
        learn_gamma: bool = False,
        gamma_per_dim: bool = True,
        bisection_iters: int = 40,
        eps: Epsilons = Epsilons(),
        clamp_xi_to_feasible: bool = False,
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.bisection_iters = int(bisection_iters)
        self.eps = eps
        self.clamp_xi_to_feasible = bool(clamp_xi_to_feasible)

        self.bounds = BoundsParam(
            n=self.n,
            mode=bounds_mode,
            m_init=m_init,
            M_init=M_init,
            learnable=learn_bounds,
            eps_width=eps.width,
        )
        self.gamma = GammaParam(
            n=self.n,
            init=gamma_init,
            learnable=learn_gamma,
            per_dim=gamma_per_dim,
            eps=eps.gamma,
        )

        self.learn_xi = bool(learn_xi)
        self.xi_mode = xi_mode
        xi0 = torch.as_tensor(xi, dtype=torch.float32).reshape(-1)
        if self.learn_xi:
            if xi_mode == "fixed":
                # learn unconstrained xi directly (may become infeasible unless clamp_xi_to_feasible)
                self.xi_raw = nn.Parameter(xi0.clone())
            elif xi_mode == "feasible_sigmoid":
                # learn t, then xi = sum m + sigmoid(t) * (sum M - sum m)
                self.t_raw = nn.Parameter(torch.zeros_like(xi0))
            else:
                raise ValueError(f"Unknown xi_mode: {xi_mode}")
        else:
            self.register_buffer("xi_buf", xi0.clone())

    def _compute_xi(
        self,
        m: torch.Tensor,
        M: torch.Tensor,
        batch: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if not self.learn_xi:
            xi = self.xi_buf.to(dtype=dtype, device=device)
            if xi.numel() == 1:
                return xi.expand(batch)
            if xi.numel() == batch:
                return xi.reshape(batch)
            raise ValueError("Fixed xi must be scalar or (B,) for current batch.")
        # learnable xi
        if self.xi_mode == "fixed":
            xi = self.xi_raw.to(dtype=dtype, device=device).reshape(-1)
            if xi.numel() == 1:
                return xi.expand(batch)
            if xi.numel() == batch:
                return xi.reshape(batch)
            raise ValueError("Learned xi (fixed) must be scalar or (B,).")

        # feasible_sigmoid
        t = self.t_raw.to(dtype=dtype, device=device).reshape(-1)
        if t.numel() == 1:
            t = t.expand(batch)
        elif t.numel() != batch:
            raise ValueError("t_raw must be scalar or (B,).")

        sum_m = m.sum(dim=-1)
        sum_M = M.sum(dim=-1)
        return sum_m + torch.sigmoid(t) * (sum_M - sum_m)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B,N) scores to be projected.

        Returns:
            x: (B,N) feasible solution.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.dim() != 2 or z.shape[-1] != self.n:
            raise ValueError(f"Expected z shape (B,{self.n}) or ({self.n},), got {tuple(z.shape)}")

        dtype = z.dtype
        device = z.device
        B, N = z.shape

        gamma = self.gamma(dtype=dtype, device=device)  # (N,) or (1,)
        if gamma.numel() == 1:
            gamma = gamma.expand(N)
        gamma = gamma.unsqueeze(0).expand(B, N)

        m, M = self.bounds(dtype=dtype, device=device)
        m = m.unsqueeze(0).expand(B, N)
        M = M.unsqueeze(0).expand(B, N)

        xi = self._compute_xi(m=m, M=M, batch=B, dtype=dtype, device=device)

        if self.clamp_xi_to_feasible:
            sum_m = m.sum(dim=-1)
            sum_M = M.sum(dim=-1)
            xi = torch.clamp(xi, min=sum_m, max=sum_M)

        return _SeparableQProjectionFn.apply(
            z, gamma, m, M, xi, self.bisection_iters, float(self.eps.active_set)
        )


class BetaToZ(nn.Module):
    """
    Convert (beta, gamma) into z = -beta/(2gamma).
    Useful when upstream predicts beta and/or gamma.
    """

    def __init__(self, eps_gamma: float = 1e-6) -> None:
        super().__init__()
        self.eps_gamma = float(eps_gamma)

    def forward(self, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        gamma = torch.clamp(gamma, min=self.eps_gamma)
        return -beta / (2.0 * gamma)


class SeparableQPSolveFromBeta(nn.Module):
    """
    Convenience wrapper for the original form:
        min_x sum_i (gamma_i x_i^2 + beta_i x_i) s.t. constraints
    using z = -beta/(2gamma) and projecting.
    """

    def __init__(self, projection: SeparableQPProjection) -> None:
        super().__init__()
        self.projection = projection
        self.beta_to_z = BetaToZ(eps_gamma=projection.eps.gamma)

    def forward(self, beta: torch.Tensor, gamma: Optional[torch.Tensor] = None) -> torch.Tensor:
        if beta.dim() == 1:
            beta_ = beta.unsqueeze(0)
        else:
            beta_ = beta

        if gamma is None:
            B, N = beta_.shape
            g = self.projection.gamma(dtype=beta_.dtype, device=beta_.device)
            if g.numel() == 1:
                g = g.expand(N)
            g = g.unsqueeze(0).expand(B, N)
            z = self.beta_to_z(beta_, g)
            return self.projection(z)

        z = self.beta_to_z(beta_, gamma)
        return self.projection(z)


class FixedZProjection(nn.Module):
    """If z is known and constant, ignore input and output the projected solution."""

    def __init__(self, z_const: TensorLike, projection: SeparableQPProjection) -> None:
        super().__init__()
        z0 = torch.as_tensor(z_const, dtype=torch.float32).reshape(-1)
        if z0.numel() != projection.n:
            raise ValueError("z_const must have shape (N,).")
        self.register_buffer("z_const", z0)
        self.projection = projection

    def forward(
        self,
        batch_size: int = 1,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        z = self.z_const
        if device is not None:
            z = z.to(device=device)
        if dtype is not None:
            z = z.to(dtype=dtype)
        z = z.unsqueeze(0).expand(batch_size, -1)
        return self.projection(z)
