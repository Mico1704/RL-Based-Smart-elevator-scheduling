"""Double DQN with a factorised (value-decomposition) Q-head, implemented in pure NumPy.

Q_joint(s, a1..a3) = sum_i Q_i(s, a_i): the 27-action argmax splits into 3 independent argmaxes,
and the network only needs 9 outputs. Trained on the shared team reward with Adam + Huber loss.
"""
import numpy as np


class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]; self.v = [np.zeros_like(p) for p in self.W + self.b]; self.k = 0

    def forward(self, x, cache=False):
        hs = [x]
        for j, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if j < len(self.W) - 1: x = np.maximum(x, 0)
            hs.append(x)
        return (x, hs) if cache else x

    def step(self, hs, gout, lr, clip=10.0):
        gW, gb, g = [], [], gout
        for j in reversed(range(len(self.W))):
            gW.insert(0, hs[j].T @ g); gb.insert(0, g.sum(0))
            if j > 0: g = (g @ self.W[j].T) * (hs[j] > 0)
        grads = gW + gb
        n = np.sqrt(sum((x ** 2).sum() for x in grads))
        if n > clip: grads = [x * clip / n for x in grads]
        self.k += 1; params = self.W + self.b
        for p, g_, m, v in zip(params, grads, self.m, self.v):
            m *= 0.9; m += 0.1 * g_; v *= 0.999; v += 0.001 * g_ * g_
            p -= lr * (m / (1 - 0.9 ** self.k)) / (np.sqrt(v / (1 - 0.999 ** self.k)) + 1e-8)

    def copy_from(self, o, tau=1.0):
        for p, q in zip(self.W + self.b, o.W + o.b): p *= (1 - tau); p += tau * q


class DQNAgent:
    name = "DQN"
    def __init__(self, obs_dim, n_cars=3, hidden=(256, 256), lr=5e-4, gamma=0.99, buf=100_000, batch=128, seed=0):
        self.rng = np.random.default_rng(seed); self.C = n_cars
        sizes = [obs_dim, *hidden, 3 * n_cars]
        self.q, self.tq = MLP(sizes, self.rng), MLP(sizes, self.rng); self.tq.copy_from(self.q)
        self.lr, self.gamma, self.batch, self.N, self.n, self.ptr = lr, gamma, batch, buf, 0, 0
        self.S = np.zeros((buf, obs_dim), np.float32); self.S2 = np.zeros_like(self.S)
        self.A = np.zeros((buf, n_cars), np.int64); self.R = np.zeros(buf, np.float32); self.D = np.zeros(buf, np.float32)  # D = gamma^k bootstrap multiplier (0 if terminal)

    def greedy_acts(self, obs):
        return self.q.forward(obs[None])[0].reshape(self.C, 3).argmax(1)

    def act(self, env, eps=0.0):
        a = self.greedy_acts(env._obs())
        if eps > 0:
            m = self.rng.random(self.C) < eps; a = np.where(m, self.rng.integers(0, 3, self.C), a)
        return env.encode(a)

    def reset(self, env): pass

    def store(self, s, a, r, s2, d):
        i = self.ptr; self.S[i], self.A[i], self.R[i], self.S2[i], self.D[i] = s, a, r, s2, d
        self.ptr = (i + 1) % self.N; self.n = min(self.n + 1, self.N)

    def learn(self):
        idx = self.rng.integers(0, self.n, self.batch)
        s, a, r, s2, d = self.S[idx], self.A[idx], self.R[idx], self.S2[idx], self.D[idx]
        a_star = self.q.forward(s2).reshape(-1, self.C, 3).argmax(2)                    # double-DQN action choice
        qt = self.tq.forward(s2).reshape(-1, self.C, 3)
        y = r + d * np.take_along_axis(qt, a_star[..., None], 2).sum((1, 2))
        out, hs = self.q.forward(s, cache=True)
        o3 = out.reshape(-1, self.C, 3)
        qsa = np.take_along_axis(o3, a[..., None], 2).sum((1, 2))
        err = qsa - y; g = np.clip(err, -5, 5) / self.batch                             # Huber gradient
        gout = np.zeros_like(o3); np.put_along_axis(gout, a[..., None], g[:, None, None] * np.ones((1, self.C, 1), np.float32), 2)
        self.q.step(hs, gout.reshape(-1, 3 * self.C), self.lr)
        return float((err ** 2).mean())

    def save(self, path):
        np.savez(path, **{f"W{i}": w for i, w in enumerate(self.q.W)}, **{f"b{i}": b for i, b in enumerate(self.q.b)})

    def load(self, path):
        z = np.load(path)
        for i in range(len(self.q.W)): self.q.W[i][:] = z[f"W{i}"]; self.q.b[i][:] = z[f"b{i}"]
