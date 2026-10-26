"""Train the DQN agent on mixed traffic. Usage: python -m elevator_rl.train --steps 300000 --lam 0.1 --out runs/lam0.1"""
import argparse, json, os, time
import numpy as np
from .env import ElevatorEnv
from .dqn import DQNAgent
from .baselines import NearestCar, Scan


def train(steps=300_000, lam=0.10, seed=0, out="runs/default", hidden=(256, 256), lr=5e-4, log=True, guide=0.8, gamma=0.95, nstep=8, guide_floor=0.3):
    os.makedirs(out, exist_ok=True)
    env = ElevatorEnv("mixed", lam=lam)
    ag = DQNAgent(env.obs_dim, hidden=hidden, lr=lr, gamma=gamma, seed=seed)
    s, _ = env.reset(seed=seed); ep, ep_ret, hist, t0 = 0, 0.0, [], time.time()
    helpers = [NearestCar(), Scan()]
    rng = np.random.default_rng(seed + 1)

    def new_episode_policy(t):
        """Guided exploration: early on, many episodes are driven by a rule-of-thumb controller (+noise) so that
        useful transitions (board, deliver) reach the replay buffer; the agent still learns only from the reward."""
        p = max(guide_floor, guide * (1.0 - t / (0.6 * steps)))
        if rng.random() < p:
            h = helpers[int(rng.integers(len(helpers)))]; h.reset(env); return h
        return None
    helper = new_episode_policy(0); win = []

    def flush(final):
        while win and (len(win) >= nstep or final):
            R = sum(gamma ** k * w[2] for k, w in enumerate(win)); k = len(win)
            ag.store(win[0][0], win[0][1], R, win[-1][3], 0.0 if final else gamma ** k)
            win.pop(0)
    for t in range(1, steps + 1):
        eps = max(0.05, 1.0 - 0.95 * t / (0.5 * steps))
        if helper is not None:
            a = helper.act(env)
            if rng.random() < 0.10:                       # per-step noise on the guide
                acts = env.decode(a); k = int(rng.integers(env.C)); acts[k] = int(rng.integers(3)); a = env.encode(acts)
        else:
            a = ag.act(env, eps)
        s2, r, _, done, _ = env.step(a)
        win.append((s, np.array(env.decode(a)), r, s2)); flush(done); s = s2; ep_ret += r
        if t > 2000: ag.learn()
        if t % 4 == 0: ag.tq.copy_from(ag.q, tau=0.03)
        if done:
            guided_flag = helper is not None
            m = env.metrics(); hist.append(dict(step=t, episode=ep, pattern=env.pattern, ret=ep_ret, guided=guided_flag, **m))
            ep += 1; ep_ret = 0.0; s, _ = env.reset(); helper = new_episode_policy(t)
            if log and ep % 50 == 0:
                print(f"step {t} ep {ep} eps {eps:.2f} ret(last50) {np.mean([h['ret'] for h in hist[-50:]]):.1f} "
                      f"wait {np.mean([h['avg_wait'] for h in hist[-50:]]):.1f} E {np.mean([h['energy'] for h in hist[-50:]]):.0f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    ag.save(os.path.join(out, "dqn.npz")); json.dump(hist, open(os.path.join(out, "train_log.json"), "w"))
    return ag, hist


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300_000); p.add_argument("--lam", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--gamma", type=float, default=0.95); p.add_argument("--nstep", type=int, default=8); p.add_argument("--hidden", type=int, default=128); p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--out", default="runs/default")
    a = p.parse_args(); train(a.steps, a.lam, a.seed, a.out, hidden=(a.hidden, a.hidden), lr=a.lr, gamma=a.gamma, nstep=a.nstep)
