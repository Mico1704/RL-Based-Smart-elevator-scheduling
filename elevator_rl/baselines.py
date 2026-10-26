"""Rule-based controllers: Random, Nearest-car, SCAN-style. Each maps env -> joint action."""
import numpy as np
from .env import ElevatorEnv


class Random:
    name = "Random"
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def reset(self, env): pass
    def act(self, env): return int(self.rng.integers(env.action_space.n))


def _requests(env, i):
    return {p[1] for p in env.riders[i]}


class NearestCar:
    """Each hall call goes to the closest car that has room; cars serve their riders + assigned calls."""
    name = "Nearest-car"
    def reset(self, env): pass
    def act(self, env):
        C, F = env.C, env.F
        assigned = [set() for _ in range(C)]
        calls = sorted((f for f in range(F) if env.waiting[f]), key=lambda f: env.waiting[f][0][2])
        for f in calls:
            cand = [i for i in range(C) if len(env.riders[i]) < env.cap] or list(range(C))
            i = min(cand, key=lambda j: (abs(env.floor[j] - f), len(assigned[j])))
            assigned[i].add(f)
        acts = []
        for i in range(C):
            f = env.floor[i]
            if f in _requests(env, i) or (env.waiting[f] and len(env.riders[i]) < env.cap):
                acts.append(1); continue
            targets = _requests(env, i) | assigned[i]
            if not targets: acts.append(1); continue
            t = min(targets, key=lambda x: abs(x - f))
            acts.append(2 if t > f else 0)
        return ElevatorEnv.encode(acts)


class Scan:
    """Elevator (SCAN) algorithm: sweep in one direction serving requests, reverse when none ahead."""
    name = "SCAN-style"
    def reset(self, env): self.d = [1 if i % 2 == 0 else -1 for i in range(env.C)]
    def act(self, env):
        C, F = env.C, env.F
        acts = []
        for i in range(C):
            f = env.floor[i]
            req = _requests(env, i) | {x for x in range(F) if env.waiting[x]}
            if f in _requests(env, i) or (env.waiting[f] and len(env.riders[i]) < env.cap):
                acts.append(1); continue
            if not req: acts.append(1); continue
            ahead = [x for x in req if (x - f) * self.d[i] > 0]
            if not ahead:
                self.d[i] = -self.d[i]
                ahead = [x for x in req if (x - f) * self.d[i] > 0]
                if not ahead: acts.append(1); continue
            acts.append(2 if self.d[i] > 0 else 0)
        return ElevatorEnv.encode(acts)
