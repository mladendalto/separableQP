# Separable Box-Constrained Sum-Equality QP Layer

A lightweight PyTorch layer for projecting vectors onto a box with a single sum-equality constraint. The projection is solved in closed form via a 1D root find, making it fast, deterministic, and fully differentiable. This README highlights the figures produced by `demo_separable_qp_layer.py` (saved under `demo_outputs/` after you run the script locally).

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

The demo script generates figures under `demo_outputs/` that illustrate common regimes (run the script to populate the folder):

- **Simplex-like activation:** compares the QP projection to softmax on $m=0$, $M=1$, $\xi=1$, $\gamma=1$ (e.g., the consolidated histogram panels in `simplex_value_panels.png`, sorted-profile + support overlay in `simplex_sorted_profiles_panel.png`, and the combined view in `simplex_support_topk_panel.png`).
- **k-hot / budgeted gating:** enforces $\sum x = k$ with box bounds (e.g., `khot_sorted_profiles_quantiles.png` and `khot_support_topk_panel.png`).
- **Geometry (2D):** shows contours, feasible segment, and solution for $N=2$ (`contour_n2_geometry.png`).
- **Effect of stiffness $\gamma$:** redistributes mass according to per-dimension weights (`gamma_group_mass_compare.png`).
- **Activation family benchmark:** compares sparsity and runtime for softmax, sparsemax, entmax (configurable $\alpha$), and this layer on the same logits (value/log panels in `activation_family_value_panels.png`, sorted profiles in `activation_family_sorted_profiles.png`, support vs top-k mass in `activation_family_support_topk.png`, runtime bar chart in `activation_family_runtime.png`, sparsity bar chart in `activation_family_support.png`, and the underlying CSV `activation_family_profile.csv`).
- **Box constraints in action:** demonstrates how explicit upper bounds spread allocations compared to simplex-style normalization (`box_constraint_sorted_profiles.png` and `box_constraint_support_topk.png`).

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

The script emits PNGs and printed statistics (row sums, support sizes, entropy, feasibility checks) to `demo_outputs/`.

## Comparing with other sparse activations

The demo now produces a side-by-side comparison of softmax, sparsemax, entmax-$\alpha$ (including $\alpha=1$ for softmax-like and $\alpha=2$ for sparsemax-like behavior), and this layer (all on the same logits). The plots above show both the sparsity profile and a quick CPU runtime bar chart. Raw measurements live in `demo_outputs/activation_family_profile.csv` so you can reproduce the simple complexity table in the README and compare against your own hardware.

| Method | Similarity | Where SeparableQP Wins |
| --- | --- | --- |
| [Sparsemax (Martins et al., 2016)](https://arxiv.org/abs/1602.02068) | High. It is equivalent to this layer with $\gamma=1$, $m=0$, $M=1$. | Flexibility. Sparsemax is homoscedastic (assumes equal variance); this layer adapts curvature per class. |
| [OptNet (Amos & Kolter, 2017)](https://arxiv.org/abs/1703.00443) | High. Solves generic QPs differentiably. | Speed. OptNet is $O(N^3)$ or iterative; this layer is $O(N\log N)$. In a Transformer, OptNet is unusable; this layer is feasible. |
| [Constrained Sparsemax (Malaviya et al., 2018)](https://aclanthology.org/W18-5409) | Very high. Adds upper bounds to Sparsemax. | Learnable $\gamma$. They usually fix the quadratic penalty; learning $\gamma$ prevents dead gradients by softening curvature dynamically. |
| [Entmax-$\alpha$ (Peters et al., 2019)](https://arxiv.org/abs/1905.05702) | Moderate–high. Controls sparsity with $\alpha$. | Exact feasibility. Entmax only enforces non-negativity; this layer hits the budget and box bounds exactly. |

The runtime and sparsity figures in `demo_outputs/` highlight that SeparableQP matches or exceeds the sparsity of entmax/sparsemax while staying close to softmax in wall-clock time on CPU (see [`activation_family_runtime.png`](demo_outputs/activation_family_runtime.png) and [`activation_family_support.png`](demo_outputs/activation_family_support.png)).

## Box constraints in practice

To illustrate why explicit bounds matter, the demo includes a small allocation toy example. With only the simplex constraint ($m=0$, $M=1$) a handful of coordinates can dominate; adding $M=0.25$ forces mass to spread while still obeying the sum target. The contrast is visible once you regenerate `box_constraint_sorted_profiles.png` and the top-k mass/support panel `box_constraint_support_topk.png` via the demo.

## When to use this layer

- **Deterministic constrained activations:** enforce exact budgets in attention, gating, or allocation modules.
- **Learnable bounds:** keep parameters inside $[m,M]$ during training without ad-hoc clipping.
- **Replacement for softmax/sigmoid:** recover sparse simplex-like outputs with predictable sparsity via box constraints and $\gamma$ shaping.

Because the solver is analytical and batched, it is well-suited for fast inference or as a drop-in differentiable projection in larger models.
