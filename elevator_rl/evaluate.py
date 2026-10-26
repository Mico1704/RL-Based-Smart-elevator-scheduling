"""Evaluate all controllers on identical fresh seeds. Usage: python -m elevator_rl.evaluate --model runs/lam0.1/dqn.npz"""
import argparse, os
import numpy as np, pandas as pd
from .env import ElevatorEnv, PATTERNS
from .baselines import Random, NearestCar, Scan
from .dqn import DQNAgent

EVAL_SEED0 = 10_000   # training uses other seeds


def run_episode(ctrl, pattern, seed, lam=0.10):
    env = ElevatorEnv(pattern, lam=lam); env.reset(seed=seed); ctrl.reset(env); done, ret = False, 0.0
    while not done:
        _, r, _, done, _ = env.step(ctrl.act(env)); ret += r
    return {**env.metrics(), "ret": ret}


def evaluate(controllers, n_seeds=20, lam=0.10):
    rows = []
    for pi, pat in enumerate(PATTERNS):
        for k in range(n_seeds):
            seed = EVAL_SEED0 + 100 * pi + k
            for c in controllers:
                if hasattr(c, "rng") and c.name == "Random": c.rng = np.random.default_rng(seed)
                rows.append(dict(pattern=pat, controller=c.name, seed=seed, **run_episode(c, pat, seed, lam)))
    return pd.DataFrame(rows)


def summary(df):
    g = df.groupby(["pattern", "controller"])[["avg_wait", "max_wait", "avg_travel", "energy", "unserved"]].agg(["mean", "std"])
    return g


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--out", default="results"); a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    ag = DQNAgent(ElevatorEnv().obs_dim, hidden=(128, 128)); ag.load(a.model)
    df = evaluate([Random(), NearestCar(), Scan(), ag], a.seeds); df.to_csv(f"{a.out}/evaluation_results.csv", index=False)
    pd.set_option("display.width", 200); print(summary(df).round(1))
