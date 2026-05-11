# 2048 Reinforcement Learning

Reinforcement learning agents for the game 2048, implementing three approaches:

| Agent | Algorithm | Requires training? |
|---|---|---|
| **SARSA** | Semi-gradient SARSA with linear function approximation | Yes (`train.py`) |
| **Beam Search** | Greedy tree search with handcrafted heuristics | No |
| **DQN** | Deep Q-Network with experience replay | Yes (`dqn.py`) |

Based on [Li & Peng, "Playing 2048 With Reinforcement Learning" (2021)](https://arxiv.org/abs/2110.10374).

---

## Setup

```bash
git clone https://github.com/<your-username>/RL2048.git
cd RL2048
pip install numpy scipy matplotlib torch
```

PyTorch and numpy are pre-installed on Kaggle and Google Colab — no extra installs needed there.

---

## File structure

```
RL2048/
├── game.py          # Game logic, state encoding, feature extraction, heuristic_score
├── environment.py   # Abstract Environment base class
├── agent.py         # SARSA agent, all policies including beam_search_policy
├── episode.py       # Episode rollout helper
├── train.py         # Train SARSA → saves w_star.npy
├── dqn.py           # DQN agent + all standalone visualizations
├── 2048_DQN.ipynb   # Colab/Kaggle notebook (train, eval, compare, plot)
├── benchmark.py     # Local comparison script for all methods
├── test.py          # Tkinter visualizer (SARSA)
├── w_star.npy       # SARSA weights (zero-initialized; overwritten by train.py)
└── constants.py     # UI constants for the tkinter visualizer
```

---

## Training

### SARSA

```bash
python train.py
```

Runs semi-gradient SARSA indefinitely until convergence (‖Δw‖ < tolerance). Stop early with `Ctrl+C` — weights are saved to `w_star.npy` automatically. Progress is logged to `agent.log` and stdout every 10 episodes. Expect ~20–50k episodes for a reasonable policy.

**Resume from checkpoint:**
`train.py` loads `w_star.npy` at startup if it exists and the shape matches. Just re-run `python train.py` to continue.

**Hyperparameters** (edit `train.py`):
- `alpha=1e-11` — step size (learning rate)
- `tolerance=1e-20` — convergence threshold for ‖Δw‖
- `alpha_decay=True` — divides α by update count over time

### DQN

**Local:**
```bash
python dqn.py                            # 2000 games with defaults
python dqn.py --games 5000 --hidden 256  # deeper network, more training
python dqn.py --load dqn_weights.pth     # resume from checkpoint
```

Saves `dqn_weights.pth` and `dqn_weights_train_stats.json` (training curve data).

**Google Colab / Kaggle:**
```python
# Cell 1 — get the code
!git clone https://github.com/<your-username>/RL2048.git
import sys
sys.path.insert(0, 'RL2048')

# Cell 2 — train
from dqn import DQNAgent
agent = DQNAgent()
stats = agent.train(n_games=5000, print_every=100)

# Cell 3 — save
agent.save('dqn_weights.pth')

# Cell 4 — plot training curve
import matplotlib.pyplot as plt
import numpy as np
window = 200
steps = [s['steps'] for s in stats]
smoothed = np.convolve(steps, np.ones(window)/window, mode='valid')
plt.plot(smoothed)
plt.xlabel('Episode')
plt.ylabel(f'Moves (rolling avg {window})')
plt.title('DQN Training Curve')
plt.show()
```

**DQN hyperparameters** (CLI flags or constructor kwargs):

| Flag | Default | Meaning |
|---|---|---|
| `--games` | 2000 | Number of training episodes |
| `--hidden` | 128 | Hidden layer size |
| `--lr` | 1e-4 | Adam learning rate |
| `--epsilon-start` | 0.3 | Initial exploration rate |
| `--epsilon-end` | 0.05 | Minimum exploration rate |
| `--epsilon-decay` | 0.995 | Multiplicative decay per episode |
| `--batch` | 128 | Replay buffer batch size |
| `--buffer` | 10000 | Replay buffer capacity |

---

## Benchmarking & comparison

### Quick benchmark (random vs SARSA only — fast)

```bash
python benchmark.py --games 200
```

### Full benchmark including beam search and DQN

```bash
# Requires: w_star.npy (trained SARSA), dqn_weights.pth (trained DQN)
python benchmark.py --games 100 --all
```

> **Note:** beam search is slow (~2–10 seconds per game at depth=20, k=10). Use fewer games or reduce depth/k for quick tests.

### Tune beam search speed vs. quality

```bash
python benchmark.py --beam --beam-depth 10 --beam-k 5 --games 50   # fast
python benchmark.py --beam --beam-depth 20 --beam-k 10 --games 100  # paper settings
```

### Outputs

- Printed summary table (avg moves, % reaching 2048 / 1024 / 512 / 256)
- `benchmark_results.png` — side-by-side bar chart + box plot
- `benchmark_results.json` — raw per-game data (max_tile, steps) for all methods

### Plot only DQN training curve

```bash
python benchmark.py --dqn-curve
# Reads dqn_weights_train_stats.json, saves dqn_training_curve.png
```

---

## Visualization (live game replay)

The tkinter visualizer in `test.py` shows the SARSA agent playing in real time:

```bash
python test.py
```

Requires a display (won't work in headless Colab/Kaggle). On macOS you may need `python3 -m tkinter` to confirm tkinter is installed.

**To watch the beam search agent play**, swap the policy in `test.py`:

```python
# In test.py main(), replace:
action = agent.softmax_policy(current_state)
# with:
action = agent.beam_search_policy(current_state, depth=20, k=10)
```

**To watch the DQN agent play**, replace the agent block in `test.py main()`:

```python
from dqn import DQNAgent
dqn_agent = DQNAgent()
dqn_agent.load('dqn_weights.pth')
# then in the loop:
action = dqn_agent.select_action(game_grid.board, epsilon=0.0)
```

---

## Feature vector (SARSA)

13 features extracted from each (state, action) pair, computed on the post-action board before a new tile spawns:

| # | Feature | Description |
|---|---|---|
| 1 | `points` | Score gained by this action |
| 2 | `emptiness` | Number of blank tiles |
| 3 | `roughness` | Std dev of adjacent tile differences |
| 4 | `monotonicity` | Longest non-decreasing tile run |
| 5 | `std_vertical_dif` | Std dev of row-to-row differences |
| 6 | `std_horizontal_dif` | Std dev of col-to-col differences |
| 7 | `tile_delta` | Min sum of directional differences |
| 8 | `mean` | Mean tile value |
| 9 | `std` | Std dev of tile values |
| 10 | `distance_to_corner` | Manhattan distance of max tile to nearest corner |
| 11 | `center_sum` | Sum of 4 center tiles |
| 12 | `perimeter_sum` | Sum of 12 perimeter tiles |
| 13 | `max_tile` | log2 of the maximum tile |

---

## References

- Li & Peng, [Playing 2048 With Reinforcement Learning](https://arxiv.org/abs/2110.10374), NeurIPS 2021
- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.
- Szubert & Jaskowski, [Temporal Difference Learning of N-Tuple Networks for 2048](http://www.cs.put.poznan.pl/wjaskowski/pub/papers/Szubert2014_2048.pdf), IEEE 2014
