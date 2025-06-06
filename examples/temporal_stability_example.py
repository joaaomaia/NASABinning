"""Example demonstrating temporal stability with NASABinning."""

import numpy as np
import pandas as pd

from nasabinning import NASABinner
from nasabinning.temporal_stability import (
    temporal_separability_score,
    event_rate_by_time,
    ks_over_time,
)

# synthetic dataset -------------------------------------------------------
rng = np.random.default_rng(0)
n = 800

# feature with slight drift across months
X = pd.DataFrame({"x": rng.normal(size=n)})
X["month"] = rng.choice([202301, 202302, 202303, 202304], size=n)

proba = 0.2 + 0.15 * X["x"] + 0.02 * (X["month"] - 202301)
y = (rng.random(n) < proba).astype(int)

# fit NASABinner using Optuna to maximise temporal separability
binner = NASABinner(
    strategy="supervised",
    check_stability=True,
    use_optuna=True,
    time_col="month",
    strategy_kwargs={"n_trials": 10},
)

binner.fit(X, y, time_col="month")

# stability metrics ------------------------------------------------------
pivot = binner.stability_over_time(X, y, time_col="month")
ks = ks_over_time(pivot)

bins = binner.transform(X)["x"]
sep = temporal_separability_score(
    pd.DataFrame({"bin": bins, "target": y, "time": X["month"]}),
    "x",
    "bin",
    "target",
    "time",
)

print(f"Temporal separability: {sep:.3f}")
print(f"IV: {binner.iv_:.3f}")
print(f"KS over time: {ks:.3f}")

# plot curves ------------------------------------------------------------
# Exibe gráfico de event rate por safra para cada bin
binner.plot_event_rate_stability(pivot)
