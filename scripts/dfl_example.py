"""Decision-focused learning (DFL), a small self-contained example (Phase 5, step 5.4).

A forecaster is usually trained to be accurate (least squares). DFL trains it on the cost of
the decisions it leads to: the decision (commit tomorrow's energy) is a convex optimization
problem, made differentiable with cvxpylayers, so the gradient of the realized cost flows back
into the forecaster. Warm-starting from the least-squares model matters: from a random start
the forecast can get stuck at zero, where gradients vanish.

    uv run python scripts/dfl_example.py     # needs the dfl extra: `make setup` installs it

Toy data, no battery: with a shortfall penalty of 3x the price, the best decision commits less
than the expected output, and DFL learns that on its own. Next step: replace the toy data with
your backtest's features and the battery rule from dispatch.py.
"""

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

PRICE, PENALTY, EPS = 1.0, 3.0, 0.1
pv_hat = cp.Parameter(24)
commit = cp.Variable(24)
objective = cp.Maximize(
    PRICE * cp.sum(commit)
    - PENALTY * cp.sum(cp.pos(commit - pv_hat))
    - EPS * cp.sum_squares(commit - pv_hat)
)
layer = CvxpyLayer(
    cp.Problem(objective, [commit >= 0, commit <= 1]), parameters=[pv_hat], variables=[commit]
)

torch.manual_seed(0)
rng = np.random.default_rng(0)
days = 300
shape = torch.tensor(
    np.clip(np.sin((np.arange(24) - 6) / 12 * np.pi), 0, None), dtype=torch.float32
)
x = torch.tensor(rng.uniform(0.3, 1.0, (days, 1)), dtype=torch.float32)
y = (x + 0.15 * torch.randn(days, 1)).clamp(0, 1) * shape
train, test = slice(0, 200), slice(200, 300)


def realized_cost(c, actual):
    return (-PRICE * c.sum(1) + PENALTY * torch.relu(c - actual).sum(1)).mean()


# 1. the usual model: least squares on the daily level
X = torch.cat([x, torch.ones_like(x)], 1)
coef = torch.linalg.lstsq(X[train], (y[train].sum(1, keepdim=True) / shape.sum())).solution
# 2. decision-focused: start from the least-squares model (warm start), then minimize realized cost
model = torch.nn.Linear(1, 1)
with torch.no_grad():
    model.weight.fill_(float(coef[0]))
    model.bias.fill_(float(coef[1]))
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for _epoch in range(150):
    (c,) = layer((model(x[train]) * shape).clamp(0, 1))
    loss = realized_cost(c, y[train])
    opt.zero_grad()
    loss.backward()
    opt.step()
with torch.no_grad():
    (c_mse,) = layer(((X[test] @ coef) * shape).clamp(0, 1))
    (c_dfl,) = layer((model(x[test]) * shape).clamp(0, 1))
    print(
        f"test days, realized profit per day  least-squares: {-realized_cost(c_mse, y[test]):.3f}   "
        f"decision-focused: {-realized_cost(c_dfl, y[test]):.3f}"
    )
    print(
        f"slope/intercept  least-squares: {float(coef[0]):.3f}/{float(coef[1]):.3f}   "
        f"decision-focused: {float(model.weight):.3f}/{float(model.bias):.3f}"
    )
