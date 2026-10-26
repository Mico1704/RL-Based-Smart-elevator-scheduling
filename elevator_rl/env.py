"""Custom Gymnasium elevator-group-control simulator (10 floors, 3 cars, capacity 8).

One decision step = one simulator tick. Every car performs one primitive action:
0 = move down one floor, 1 = stop (open doors: unload + load), 2 = move up one floor.
The central agent chooses all three at once -> 3^3 = 27 joint actions.
"""
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

PATTERNS = ["up_peak", "down_peak", "lunch", "quiet"]
RATES = {"up_peak": 0.50, "down_peak": 0.50, "lunch": 0.55, "quiet": 0.12}  # Poisson arrivals / tick
START_COST, STOP_COST = 0.5, 0.5   # extra energy for (re)starting / decelerating a car
START_FLOORS = (0, 3, 6)


def sample_od(pattern, rng, F):
    """Origin/destination distribution of one new passenger for a traffic pattern."""
    u = rng.random()

    def inter():
        o = int(rng.integers(0, F)); d = int(rng.integers(0, F - 1)); d += d >= o
        return o, d

    if pattern == "up_peak":
        if u < 0.75: return 0, int(rng.integers(1, F))
        if u < 0.90: return inter()
        return int(rng.integers(1, F)), 0
    if pattern == "down_peak":
        if u < 0.75: return int(rng.integers(1, F)), 0
        if u < 0.90: return inter()
        return 0, int(rng.integers(1, F))
    if pattern == "lunch":
        if u < 0.275: return 0, int(rng.integers(1, F))
        if u < 0.55: return int(rng.integers(1, F)), 0
        return inter()
    return inter()  # quiet: uniform inter-floor traffic


class ElevatorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, pattern="mixed", n_floors=10, n_cars=3, capacity=8, horizon=200,
                 lam=0.10, kappa=0.10, beta=0.25, seed=None):
        super().__init__()
        self.F, self.C, self.cap, self.H = n_floors, n_cars, capacity, horizon
        self.pattern_cfg, self.lam, self.kappa, self.beta = pattern, lam, kappa, beta
        self.action_space = spaces.Discrete(3 ** n_cars)
        self.obs_dim = n_cars * (n_floors + 2 + n_floors + 12) + 3 * n_floors + len(PATTERNS) + 1
        self.observation_space = spaces.Box(0.0, 1.0, (self.obs_dim,), np.float32)
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def decode(a, C=3):
        out = []
        for _ in range(C):
            out.append(a % 3); a //= 3
        return out

    @staticmethod
    def encode(acts):
        return int(sum(int(x) * 3 ** i for i, x in enumerate(acts)))

    def reset(self, seed=None, options=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.pattern = self.pattern_cfg if self.pattern_cfg != "mixed" else \
            PATTERNS[int(self.rng.integers(len(PATTERNS)))]
        self.t = 0
        self.floor = [START_FLOORS[i % len(START_FLOORS)] for i in range(self.C)]
        self.prev = [1] * self.C                    # previous action (1 = stopped)
        self.dir = [0] * self.C
        self.riders = [[] for _ in range(self.C)]   # passenger = [origin, dest, t_arrive, t_board]
        self.waiting = [deque() for _ in range(self.F)]
        self.waits, self.travels = [], []
        self.generated = 0
        self.energy = 0.0
        return self._obs(), {}

    # ------------------------------------------------------------------ dynamics
    def step(self, action):
        acts = self.decode(int(action), self.C)
        self.t += 1
        E = 0.0
        for i, a in enumerate(acts):
            f = self.floor[i]
            if (a == 0 and f == 0) or (a == 2 and f == self.F - 1):
                a = 1                                    # invalid move -> stop
            if a == 1:
                if self.prev[i] != 1: E += STOP_COST
                keep = []
                for p in self.riders[i]:
                    if p[1] == f: self.travels.append(self.t - p[3])
                    else: keep.append(p)
                self.riders[i] = keep
                q = self.waiting[f]
                while q and len(self.riders[i]) < self.cap:
                    p = q.popleft(); p[3] = self.t
                    self.waits.append(self.t - p[2]); self.riders[i].append(p)
            else:
                E += 1.0 + (START_COST if self.prev[i] != a else 0.0)
                self.floor[i] = f + a - 1
            self.prev[i] = a; self.dir[i] = a - 1
        for _ in range(self.rng.poisson(RATES[self.pattern])):
            o, d = sample_od(self.pattern, self.rng, self.F)
            self.waiting[o].append([o, d, self.t, -1]); self.generated += 1
        self.energy += E
        n_wait = sum(len(q) for q in self.waiting)
        n_ride = sum(len(r) for r in self.riders)
        reward = -self.kappa * (n_wait + self.beta * n_ride) - self.lam * E
        done = self.t >= self.H
        return self._obs(), float(reward), False, done, {"energy": E, "n_wait": n_wait}

    def _rel(self, i):
        """12 relational features of car i: pickup/drop targets here, below and above, and whether another car is closer."""
        F, f = self.F, self.floor[i]
        drops = {p[1] for p in self.riders[i]}
        calls = [x for x in range(F) if self.waiting[x]]
        room = len(self.riders[i]) < self.cap
        feat = [float(f in drops), float(bool(self.waiting[f]) and room)]
        for pool in (drops, calls):
            for sign in (-1, 1):
                c = [x for x in pool if (x - f) * sign > 0]
                if c:
                    x = min(c, key=lambda y: abs(y - f)); feat += [1.0, 1.0 - abs(x - f) / (F - 1)]
                    if pool is calls:
                        other = min((abs(self.floor[j] - x) for j in range(self.C) if j != i), default=99)
                        feat.append(float(other < abs(x - f)))
                else:
                    feat += [0.0, 0.0] + ([0.0] if pool is calls else [])
        return feat

    def _obs(self):
        F, o = self.F, []
        for i in range(self.C):
            oh = np.zeros(F); oh[self.floor[i]] = 1
            btn = np.zeros(F)
            for p in self.riders[i]: btn[p[1]] = 1
            o += [oh, [(self.dir[i] + 1) / 2, len(self.riders[i]) / self.cap], btn, self._rel(i)]
        up = np.zeros(F); dn = np.zeros(F); age = np.zeros(F)
        for f, q in enumerate(self.waiting):
            for p in q:
                if p[1] > f: up[f] += 1
                else: dn[f] += 1
            if q: age[f] = (self.t - q[0][2]) / 60.0
        pat = np.zeros(len(PATTERNS)); pat[PATTERNS.index(self.pattern)] = 1
        o += [np.minimum(up / 4, 1), np.minimum(dn / 4, 1), np.minimum(age, 1), pat, [self.t / self.H]]
        return np.concatenate([np.asarray(x, dtype=np.float32).ravel() for x in o])

    # ------------------------------------------------------------------ metrics
    def metrics(self):
        """All waits count: boarded passengers' real waits + current age of those still queued."""
        waits = list(self.waits) + [self.t - p[2] for q in self.waiting for p in q]
        return {
            "avg_wait": float(np.mean(waits)) if waits else 0.0,
            "max_wait": float(np.max(waits)) if waits else 0.0,
            "avg_travel": float(np.mean(self.travels)) if self.travels else float("nan"),
            "energy": float(self.energy),
            "generated": self.generated, "boarded": len(self.waits), "completed": len(self.travels),
            "unserved": sum(len(q) for q in self.waiting),
        }
