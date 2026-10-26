"""Lambda sweep: evaluate DQN models trained with different energy weights and plot wait vs energy.
Usage: python -m elevator_rl.sweep --models 0.02:runs/lam0.02/dqn.npz 0.1:models/dqn_v3.npz 0.3:runs/lam0.3/dqn.npz"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from .env import ElevatorEnv, PATTERNS
from .baselines import NearestCar, Scan
from .dqn import DQNAgent
from .evaluate import evaluate

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--seeds", type=int, default=20); p.add_argument("--out", default="results"); a = p.parse_args()
    os.makedirs(a.out, exist_ok=True); rows = []
    for item in a.models:
        lam, path = item.split(":"); ag = DQNAgent(ElevatorEnv().obs_dim, hidden=(128, 128)); ag.load(path); ag.name = f"DQN lam={lam}"
        df = evaluate([ag], a.seeds); df["lam"] = float(lam); rows.append(df)
    base = evaluate([NearestCar(), Scan()], a.seeds); df = pd.concat(rows + [base]); df.to_csv(f"{a.out}/lambda_sweep.csv", index=False)
    m = df.groupby("controller")[["avg_wait", "max_wait", "energy", "unserved"]].mean().round(1); print(m); m.to_csv(f"{a.out}/lambda_sweep_summary.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for c, g in m.iterrows():
        ax.scatter(g.energy, g.avg_wait, marker="o" if c.startswith("DQN") else "s", label=c)
    ax.set_xlabel("energy / episode"); ax.set_ylabel("avg wait (ticks)"); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{a.out}/lambda_tradeoff.png", dpi=150)
