# RL-Based Smart Elevator Scheduling (AI Assignment 2)
J. Arjun Naik (B23CM1018)  |  https://github.com/Mico1704/RL-Based-Smart-elevator-scheduling

Central RL agent (Double DQN with value-decomposed head, pure NumPy) schedules 3 elevators in a 10-floor building
to reduce waiting time and energy. Custom Gymnasium simulator; baselines: Random, Nearest-car, SCAN-style.

## Layout
- `elevator_rl/env.py` simulator (state, 27 joint actions, reward, energy accounting)
- `elevator_rl/baselines.py` Random / Nearest-car / SCAN-style
- `elevator_rl/dqn.py` Double DQN + factorised Q head (NumPy)
- `elevator_rl/train.py`, `elevator_rl/evaluate.py`
- `models/dqn_v3.npz` trained agent, `results/` CSVs + GIF, `figures/` plots, `report/` LaTeX + PDF

## Run (local or Google Colab)
```
pip install -r requirements.txt
python -m elevator_rl.train --steps 250000 --lam 0.1 --out runs/v3          # ~6 min CPU
python -m elevator_rl.evaluate --model models/dqn_v3.npz --seeds 20          # needs --hidden 128 (see evaluate.py)
```
Lambda sweep: rerun train with `--lam 0.02 / 0.3` and plot wait vs energy.

## Honest status
The trained agent serves most passengers but is slower than Nearest-car/SCAN on every pattern (see report); it does
NOT yet satisfy "beat best baseline on both objectives". Reproduce numbers with the commands above.
