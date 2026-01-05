# Separable Box-Constrained Sum-Equality QP Layer

A lightweight PyTorch layer for projecting vectors onto a box with a single sum-equality constraint. The projection is solved in closed form via a 1D root find, making it fast, deterministic, and fully differentiable. This README condenses the full demo report in [`demo_outputs_v2/separable_QP_demo.md`](demo_outputs_v2/separable_QP_demo.md) and links to key visualizations produced by `demo_separable_qp_layer.py`.

## Problem statement
For each row $x \in \mathbb{R}^N$ we solve

$$
\begin{aligned}
\min_{x\in\mathbb{R}^N}\;& \sum_{i=1}^N \left(\gamma_i x_i^2 + \beta_i x_i\right) \\
\text{s.t.}\;& \sum_{i=1}^N x_i = \xi,\qquad m_i \le x_i \le M_i,\qquad \gamma_i>0.
\end{aligned}
$$

Ignoring constants, this is a **weighted projection** of the unconstrained minimizer $z_i = -\tfrac{\beta_i}{2\gamma_i}$ onto the feasible polytope:

$$
\min_x\; \sum_{i=1}^N \gamma_i\,(x_i - z_i)^2\quad\text{s.t.}\quad x \in \mathcal{C},\qquad
\mathcal{C} = \{x: \sum_i x_i = \xi,\; m_i \le x_i \le M_i\}.
$$
Because $\gamma_i>0$, the objective is strictly convex and the solution is unique.

## Closed-form structure
The KKT conditions yield a scalar Lagrange multiplier $\lambda$ with elementwise solution

$$
 x_i^*(\lambda) = \operatorname{clip}\!\left(z_i + \frac{\lambda}{2\gamma_i},\; [m_i, M_i]\right),\qquad
 g(\lambda) = \sum_{i=1}^N x_i^*(\lambda) - \xi = 0.
$$

Solving the 1D root $g(\lambda)=0$ via bisection produces $x^*$ in $O(N \log \tfrac{1}{\varepsilon})$ time. The backward pass is analytic: gradients propagate through the free coordinates while active bounds receive subgradient-style signals.

## What the layer provides
- **Projection layer:** `SeparableQPProjection` implements the forward/ backward identity above with configurable, optionally learnable parameters $\gamma$, $m$, $M$, and $\xi$.
- **Beta-to-solution shortcut:** `SeparableQPSolveFromBeta` solves the quadratic directly from coefficients $(\beta,\gamma,m,M,\xi)$ without forming $z$.
- **Deterministic autograd:** Custom `torch.autograd.Function` keeps gradients exact and avoids iterative solvers in the backward pass.
- **Safe parameterizations:** `GammaParam` and `BoundsParam` ensure positivity of $\gamma$ and bound widths; `xi_mode="feasible_sigmoid"` keeps learned sums inside the feasible interval.

## Usage
```python
import torch
from separable_qp_layer import SeparableQPProjection

layer = SeparableQPProjection(
    n=8,
    xi=1.0,                     # target sum
    bounds_mode="fixed",       # or "free" / "unit" for learnable bounds
    m_init=0.0,
    M_init=1.0,
    gamma_init=1.0,
)

z = torch.randn(4, 8)          # unconstrained minimizers
x = layer(z)                   # projected outputs, sum-to-xi and within [m,M]
loss = (x**2).mean()
loss.backward()
```

To explore additional modes (learnable $\xi$, non-uniform $\gamma$, k-hot budgets, etc.), see the runnable examples in `demo_separable_qp_layer.py`.

## Visual intuition
The demo script generates figures under [`demo_outputs_v2/`](demo_outputs_v2/) that illustrate common regimes:

- **Simplex-like activation:** compares the QP projection to softmax on $m=0$, $M=1$, $\xi=1$, $\gamma=1$ (e.g., [`simplex_hist_linear.png`](demo_outputs_v2/simplex_hist_linear.png)).
- **k-hot / budgeted gating:** enforces $\sum x = k$ with box bounds (e.g., [`khot_sorted_profiles_quantiles.png`](demo_outputs_v2/khot_sorted_profiles_quantiles.png)).
- **Geometry (2D):** shows contours, feasible segment, and solution for $N=2$ ([`contour_n2_improved.png`](demo_outputs_v2/contour_n2_improved.png)).
- **Effect of stiffness $\gamma$:** redistributes mass according to per-dimension weights ([`gamma_group_mass_compare.png`](demo_outputs_v2/gamma_group_mass_compare.png)).

Place the README next to the images to keep relative links valid.

## Installation
Set up a Python environment (Python 3.9+ recommended) and install the dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` pins the runtime needs for both the layer and the demo (PyTorch, NumPy, Matplotlib, and PyTest for tests).

## Running the demo
```bash
python demo_separable_qp_layer.py --help
```
The script emits PNGs and printed statistics (row sums, support sizes, entropy, feasibility checks) to `demo_outputs_v2/`. The report [`demo_outputs_v2/separable_QP_demo.md`](demo_outputs_v2/separable_QP_demo.md) explains every figure and statistic in detail and includes troubleshooting tips for blank plots or feasibility errors.

## When to use this layer
- **Deterministic constrained activations:** enforce exact budgets in attention, gating, or allocation modules.
- **Learnable bounds:** keep parameters inside $[m,M]$ during training without ad-hoc clipping.
- **Replacement for softmax/sigmoid:** recover sparse simplex-like outputs with predictable sparsity via box constraints and $\gamma$ shaping.

Because the solver is analytical and batched, it is well-suited for fast inference or as a drop-in differentiable projection in larger models.
