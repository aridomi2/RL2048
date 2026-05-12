'''
Deep Q-Network agent for 2048.

Implements the CNN + Double DQN model from:
  Li & Peng, "Playing 2048 With Reinforcement Learning" (2021)
  https://arxiv.org/abs/2110.10374

Key design choices (all from or motivated by the paper):
  - CNN encoder: two Conv2d layers process the (16, 4, 4) one-hot board
    spatially before the MLP head, capturing tile-adjacency patterns that
    a flat MLP cannot see.
  - Double DQN: online net selects actions, target net evaluates them,
    reducing Q-value overestimation.
  - Hard target update every N steps (instead of soft τ update) gives a
    stable regression target during early training.
  - Reward = raw merge score only (no max-tile bonus) — clean, stationary
    signal aligned with the actual game objective.
  - Gradient clipping (max norm 10) guards against exploding gradients from
    the exponentially-scaled merge scores.

Self-contained: only requires numpy, torch, and game.py from this repo.
Both PyTorch and numpy are pre-installed on Kaggle and Google Colab —
no additional pip installs needed.

Quick-start (local or notebook):
    from game import TwntyFrtyEight
    from dqn import DQNAgent
    agent = DQNAgent()
    stats = agent.train(n_games=2000)
    agent.save('dqn_weights.pth')
'''

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from game import TwntyFrtyEight


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def board_to_tensor(board: np.ndarray) -> torch.Tensor:
    '''
    Encode a 4x4 numpy board as a flat (256,) float32 tensor.

    Uses the one-hot scheme from Li & Peng 2021:
      - Create a (16, 4, 4) binary array.
      - Channel i is 1 wherever board[r,c] == 2^i  (channel 0 = empty tile).
      - Flatten to 256 values.
    This representation lets the network distinguish tile magnitudes cleanly.
    '''
    one_hot = np.zeros((16, 4, 4), dtype=np.float32)
    for r in range(4):
        for c in range(4):
            val = board[r, c]
            channel = 0 if val == 0 else int(np.log2(val))
            one_hot[channel, r, c] = 1.0
    return torch.tensor(one_hot.flatten(), dtype=torch.float32)


def _valid_actions(board: np.ndarray) -> list:
    '''Return list of actions (0-3) that actually change the board.'''
    return [a for a in range(4) if np.any(TwntyFrtyEight.move(board, a)[0] != board)]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class DQNNet(nn.Module):
    '''
    CNN model from Li & Peng 2021.

    Architecture:
        Input : (batch, 256) — flattened (16, 4, 4) one-hot board
        Reshape: (batch, 16, 4, 4)
        Conv1 : Conv2d(16 → 128, 3×3, padding=1) → ReLU   # (batch, 128, 4, 4)
        Conv2 : Conv2d(128 → 128, 3×3, padding=1) → ReLU  # (batch, 128, 4, 4)
        Flatten: (batch, 128*4*4 = 2048)
        FC1   : Linear(2048 → hidden) → ReLU → Dropout
        FC2   : Linear(hidden → 4)
        Output: Q-value for each of the 4 actions

    Two conv layers let the network learn spatial patterns (corner stacking,
    monotone rows, merging opportunities) that a flat MLP cannot represent.
    The input is kept as a flat 256-vector so board_to_tensor() is unchanged;
    the reshape to (16, 4, 4) happens inside forward().
    '''
    def __init__(self, hidden_size: int = 256, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(16, 128, kernel_size=3, padding=1),   # (16,4,4) → (128,4,4)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # (128,4,4) → (128,4,4)
            nn.ReLU(),
        )
        conv_out = 128 * 4 * 4  # 2048
        self.fc = nn.Sequential(
            nn.Linear(conv_out, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 256) — reshape to spatial (batch, 16, 4, 4) for conv layers
        x = x.view(x.size(0), 16, 4, 4)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten back
        return self.fc(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    '''Fixed-capacity FIFO experience replay buffer.'''

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: torch.Tensor, action: int, reward: float,
             next_state: torch.Tensor, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    '''
    Deep Q-learning agent for 2048.

    Uses:
      - CNN encoder (two Conv2d layers) for spatial board understanding
      - Double DQN: online net selects actions, target net evaluates them
      - Experience replay (ReplayBuffer)
      - Hard target network update every `target_update_freq` optimisation steps
      - Epsilon-greedy exploration over valid actions only
      - MSE loss with gradient clipping (max norm 10)
      - Reward = raw merge score (no auxiliary bonuses)
    '''

    def __init__(self, lr: float = 1e-4, gamma: float = 0.99,
                 hidden_size: int = 256, buffer_capacity: int = 100_000,
                 dropout: float = 0.2, target_update_freq: int = 1000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self._opt_steps = 0  # counts optimisation steps for hard target update

        self.online_net = DQNNet(hidden_size=hidden_size, dropout=dropout).to(self.device)
        self.target_net = DQNNet(hidden_size=hidden_size, dropout=dropout).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Use all available GPUs via DataParallel when more than one is present.
        # save() / load() unwrap the module automatically so weights are portable.
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            self.online_net = nn.DataParallel(self.online_net)
            self.target_net = nn.DataParallel(self.target_net)

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        print(f'DQNAgent ready | device={self.device} | GPUs={max(n_gpus,1)} | '
              f'hidden={hidden_size} | buffer={buffer_capacity} | '
              f'target_update_freq={target_update_freq}')

    # ------------------------------------------------------------------
    # DataParallel helper
    # ------------------------------------------------------------------

    def _unwrap(self, net: nn.Module) -> nn.Module:
        '''Return the underlying module, unwrapping DataParallel if needed.'''
        return net.module if isinstance(net, nn.DataParallel) else net

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, board: np.ndarray, epsilon: float) -> int:
        '''Epsilon-greedy policy restricted to valid actions.'''
        actions = _valid_actions(board)
        if not actions:
            return 0
        if random.random() < epsilon:
            return random.choice(actions)
        with torch.no_grad():
            state_t = board_to_tensor(board).unsqueeze(0).to(self.device)
            q_vals = self.online_net(state_t).squeeze(0).cpu().numpy()
        # Mask invalid actions to -inf before argmax
        masked = np.full(4, -np.inf)
        for a in actions:
            masked[a] = q_vals[a]
        return int(np.argmax(masked))

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def _hard_update_target(self):
        '''Copy online network weights directly into target network.'''
        self._unwrap(self.target_net).load_state_dict(
            self._unwrap(self.online_net).state_dict()
        )

    def optimize(self, batch_size: int = 128) -> float | None:
        '''
        Sample one batch from the replay buffer and do one gradient step.

        Uses Double DQN to decouple action selection from action evaluation:
          - Online net picks the best next action: a* = argmax_a Q_online(s', a)
          - Target net evaluates it:  Q_target(s', a*)
        This reduces overestimation bias vs. vanilla max Q_target(s', a).

        Gradient clipping (max_norm=10) prevents exploding updates from
        large merge-score rewards late in training.
        '''
        if len(self.replay_buffer) < batch_size:
            return None

        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.stack(states).to(self.device)
        actions     = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards     = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones       = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Q(s, a) from online network
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target: r + γ · Q_target(s', argmax_a Q_online(s', a))
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(1, keepdim=True)
            next_q       = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q     = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Hard target update every target_update_freq optimisation steps
        self._opt_steps += 1
        if self._opt_steps % self.target_update_freq == 0:
            self._hard_update_target()

        return loss.item()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, n_games: int = 1000, epsilon_start: float = 0.3,
              epsilon_end: float = 0.05, epsilon_decay: float = 0.995,
              batch_size: int = 128, print_every: int = 100,
              n_envs: int = 1) -> list[dict]:
        '''
        Train for `n_games` episodes.

        Parameters
        ----------
        n_games        : total games to play
        epsilon_start  : initial exploration rate
        epsilon_end    : minimum exploration rate
        epsilon_decay  : multiplicative decay per completed game
        batch_size     : replay buffer sample size per optimisation step
        print_every    : log a summary line every N games
        n_envs         : number of parallel game environments.
                         Set n_envs=16 (or higher) on GPU to batch forward
                         passes and dramatically increase GPU utilisation.
                         n_envs=1 uses the original sequential loop.

        Returns a list of per-game dicts with keys:
            game, max_tile, steps, total_reward
        '''
        if n_envs > 1:
            return self._train_parallel(
                n_games=n_games, n_envs=n_envs,
                epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay, batch_size=batch_size,
                print_every=print_every,
            )

        epsilon = epsilon_start
        all_stats = []

        for game_num in range(1, n_games + 1):
            board = TwntyFrtyEight.new_board(4, 4)
            total_reward = 0.0
            steps = 0

            while True:
                action = self.select_action(board, epsilon)
                next_board, points = TwntyFrtyEight.move(board, action)

                # Apply the move only if it changes the board
                if np.any(next_board != board):
                    next_board = TwntyFrtyEight.add_two(next_board)

                status = TwntyFrtyEight.get_board_status(next_board)
                done = status in ('win', 'lose')

                # Reward: raw merge score only — clean, stationary signal
                reward = float(points)

                self.replay_buffer.push(
                    board_to_tensor(board),
                    action,
                    reward,
                    board_to_tensor(next_board),
                    done,
                )
                self.optimize(batch_size)

                total_reward += reward
                steps += 1
                board = next_board

                if done:
                    break

            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            game_stats = {
                'game':         game_num,
                'max_tile':     int(np.max(board)),
                'steps':        steps,
                'total_reward': total_reward,
            }
            all_stats.append(game_stats)

            if game_num % print_every == 0:
                recent = all_stats[-print_every:]
                avg_steps  = np.mean([s['steps'] for s in recent])
                win_rate   = sum(1 for s in recent if s['max_tile'] >= 2048) / print_every
                reach_1024 = sum(1 for s in recent if s['max_tile'] >= 1024) / print_every
                print(f'Game {game_num:>6}/{n_games} | '
                      f'avg_steps={avg_steps:>7.1f} | '
                      f'2048_rate={win_rate:>5.1%} | '
                      f'1024_rate={reach_1024:>5.1%} | '
                      f'ε={epsilon:.3f}')

        return all_stats

    def _train_parallel(self, n_games: int, n_envs: int,
                        epsilon_start: float, epsilon_end: float,
                        epsilon_decay: float, batch_size: int,
                        print_every: int) -> list[dict]:
        '''
        Train across n_envs simultaneous game environments.

        All boards are batched into a single GPU forward pass per step,
        giving the network (n_envs × avg_steps_per_game) samples per second
        instead of one — dramatically increasing GPU utilisation.

        Called automatically by train() when n_envs > 1.
        '''
        epsilon  = epsilon_start
        all_stats: list[dict] = []
        games_done = 0

        boards          = [TwntyFrtyEight.new_board(4, 4) for _ in range(n_envs)]
        episode_rewards = [0.0] * n_envs
        episode_steps   = [0]   * n_envs

        while games_done < n_games:
            # ── Batch forward pass ─────────────────────────────────────
            state_batch = torch.stack(
                [board_to_tensor(b) for b in boards]
            ).to(self.device)                          # (n_envs, 256)

            with torch.no_grad():
                q_batch = self.online_net(state_batch).cpu().numpy()  # (n_envs, 4)

            # ── Select one action per environment ──────────────────────
            actions = []
            for i, board in enumerate(boards):
                valid = _valid_actions(board)
                if not valid:
                    actions.append(0)
                elif random.random() < epsilon:
                    actions.append(random.choice(valid))
                else:
                    masked = np.full(4, -np.inf)
                    for a in valid:
                        masked[a] = q_batch[i][a]
                    actions.append(int(np.argmax(masked)))

            # ── Step every environment ─────────────────────────────────
            for i in range(n_envs):
                if games_done >= n_games:
                    break

                board  = boards[i]
                action = actions[i]
                next_board, points = TwntyFrtyEight.move(board, action)
                if np.any(next_board != board):
                    next_board = TwntyFrtyEight.add_two(next_board)

                status = TwntyFrtyEight.get_board_status(next_board)
                done   = status in ('win', 'lose')

                # Reward: raw merge score only — clean, stationary signal
                reward = float(points)

                self.replay_buffer.push(
                    board_to_tensor(board),
                    action,
                    reward,
                    board_to_tensor(next_board),
                    done,
                )

                episode_rewards[i] += reward
                episode_steps[i]   += 1
                boards[i]           = next_board

                if done:
                    games_done += 1
                    game_stats = {
                        'game':         games_done,
                        'max_tile':     int(np.max(next_board)),
                        'steps':        episode_steps[i],
                        'total_reward': episode_rewards[i],
                    }
                    all_stats.append(game_stats)

                    if games_done % print_every == 0:
                        recent     = all_stats[-print_every:]
                        avg_steps  = np.mean([s['steps'] for s in recent])
                        win_rate   = sum(1 for s in recent if s['max_tile'] >= 2048) / print_every
                        reach_1024 = sum(1 for s in recent if s['max_tile'] >= 1024) / print_every
                        print(f'Game {games_done:>6}/{n_games} | '
                              f'avg_steps={avg_steps:>7.1f} | '
                              f'2048_rate={win_rate:>5.1%} | '
                              f'1024_rate={reach_1024:>5.1%} | '
                              f'ε={epsilon:.3f}')

                    # Reset this environment
                    boards[i]          = TwntyFrtyEight.new_board(4, 4)
                    episode_rewards[i] = 0.0
                    episode_steps[i]   = 0
                    epsilon = max(epsilon_end, epsilon * epsilon_decay)

            # One optimisation step per parallel round
            self.optimize(batch_size)

        return all_stats

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def play_game(self, epsilon: float = 0.0) -> dict:
        '''Play one full game greedily. Returns {max_tile, steps}.'''
        board = TwntyFrtyEight.new_board(4, 4)
        steps = 0
        while True:
            action = self.select_action(board, epsilon)
            next_board, _ = TwntyFrtyEight.move(board, action)
            if np.any(next_board != board):
                next_board = TwntyFrtyEight.add_two(next_board)
            status = TwntyFrtyEight.get_board_status(next_board)
            board = next_board
            steps += 1
            if status in ('win', 'lose'):
                break
        return {'max_tile': int(np.max(board)), 'steps': steps}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = 'dqn_weights.pth'):
        # Unwrap DataParallel so weights are portable across machines / single-GPU
        torch.save(self._unwrap(self.online_net).state_dict(), path)
        print(f'Model saved → {path}')

    def load(self, path: str = 'dqn_weights.pth'):
        state_dict = torch.load(path, map_location=self.device)
        self._unwrap(self.online_net).load_state_dict(state_dict)
        self._unwrap(self.target_net).load_state_dict(state_dict)
        self._unwrap(self.target_net).eval()
        print(f'Model loaded ← {path}')

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, n_games: int = 200, epsilon: float = 0.0) -> list[dict]:
        '''
        Play n_games greedily and return per-game stats.
        Use after training to measure policy quality.

        Example:
            stats = agent.evaluate(200)
            plot_tile_distribution(stats)
        '''
        results = []
        for i in range(n_games):
            results.append(self.play_game(epsilon=epsilon))
            if (i + 1) % 50 == 0:
                print(f'  evaluated {i+1}/{n_games} games')
        return results


# ---------------------------------------------------------------------------
# Standalone visualizations
# (no benchmark.py needed — all plots work from dqn.py alone)
# ---------------------------------------------------------------------------

def plot_training_curve(stats: list[dict], window: int = 100,
                        save_path: str | None = 'dqn_training_curve.png'):
    '''
    Plot moves-per-game and cumulative reward over training episodes.

    Parameters
    ----------
    stats     : list returned by agent.train()
    window    : rolling-average window size
    save_path : file to save to (set None to skip saving)

    Notebook usage:
        stats = agent.train(n_games=2000)
        plot_training_curve(stats)
    '''
    import matplotlib.pyplot as plt

    games   = [s['game']         for s in stats]
    steps   = [s['steps']        for s in stats]
    rewards = [s['total_reward'] for s in stats]
    tiles   = [s['max_tile']     for s in stats]

    w = min(window, len(stats))
    kernel = np.ones(w) / w
    sm_steps   = np.convolve(steps,   kernel, mode='valid')
    sm_rewards = np.convolve(rewards, kernel, mode='valid')
    x = games[w - 1:]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(f'DQN Training  ({len(stats)} episodes)', fontsize=13, fontweight='bold')

    # Panel 1 — moves per episode
    axes[0].plot(x, sm_steps, color='#4C72B0', linewidth=1.4, label=f'rolling avg ({w})')
    axes[0].scatter(games, steps, alpha=0.08, s=4, color='#4C72B0')
    axes[0].set_ylabel('Moves per game')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    # Panel 2 — total reward per episode
    axes[1].plot(x, sm_rewards, color='#DD8452', linewidth=1.4, label=f'rolling avg ({w})')
    axes[1].scatter(games, rewards, alpha=0.08, s=4, color='#DD8452')
    axes[1].set_ylabel('Total reward')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    # Panel 3 — max tile reached per episode (scatter coloured by tile)
    tile_vals = sorted(set(tiles))
    cmap = plt.cm.get_cmap('viridis', len(tile_vals))
    tile_to_idx = {t: i for i, t in enumerate(tile_vals)}
    colors = [cmap(tile_to_idx[t]) for t in tiles]
    axes[2].scatter(games, tiles, c=colors, s=5, alpha=0.5)
    axes[2].set_ylabel('Max tile reached')
    axes[2].set_xlabel('Episode')
    axes[2].set_yscale('log', base=2)
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: str(int(v))))
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Training curve saved → {save_path}')
    plt.show()


def plot_tile_distribution(stats: list[dict],
                           label: str = 'DQN',
                           save_path: str | None = 'dqn_tile_distribution.png'):
    '''
    Bar chart showing how often the agent reached each max-tile milestone,
    plus a cumulative reach-rate panel.

    Parameters
    ----------
    stats     : list of {max_tile, steps} dicts  (from agent.evaluate() or agent.train())
    label     : legend label / title suffix
    save_path : file to save to (set None to skip saving)

    Notebook usage:
        eval_stats = agent.evaluate(200)
        plot_tile_distribution(eval_stats)
    '''
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    n = len(stats)
    milestones = [2**i for i in range(1, 12)]   # 2 … 2048
    counts  = [sum(1 for s in stats if s['max_tile'] == t) for t in milestones]
    cumul   = [sum(1 for s in stats if s['max_tile'] >= t) / n * 100 for t in milestones]
    labels  = [str(t) for t in milestones]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Tile Distribution — {label}  (n={n})', fontsize=13, fontweight='bold')

    # Left: exact distribution (counts)
    bar_color = '#4C72B0'
    bars = ax1.bar(labels, counts, color=bar_color, alpha=0.85, edgecolor='white')
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3,
                     str(cnt), ha='center', va='bottom', fontsize=8)
    ax1.set_xlabel('Max tile reached')
    ax1.set_ylabel('Number of games')
    ax1.set_title('Exact tile reached')
    ax1.tick_params(axis='x', rotation=45)

    # Right: cumulative reach rate
    ax2.bar(labels, cumul, color='#55A868', alpha=0.85, edgecolor='white')
    for i, (pct, lbl) in enumerate(zip(cumul, labels)):
        if pct > 2:
            ax2.text(i, pct + 0.5, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    ax2.set_xlabel('Reached at least this tile')
    ax2.set_ylabel('% of games')
    ax2.set_title('Cumulative reach rate')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis='x', rotation=45)

    # Print summary to console too
    print(f'\n{label} — {n} games')
    print(f"  avg moves : {np.mean([s['steps'] for s in stats]):.1f}")
    for t in [512, 1024, 2048]:
        r = sum(1 for s in stats if s['max_tile'] >= t) / n * 100
        print(f'  ≥{t:<6}   : {r:.1f}%')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Tile distribution saved → {save_path}')
    plt.show()


def compare_agents(results: dict[str, list[dict]],
                   save_path: str | None = 'comparison.png'):
    '''
    Four-panel comparison of multiple agents (2 × 2 grid).

    Panels:
      1 (top-left)  Cumulative reach rate   — % of games reaching ≥ each milestone
      2 (top-right) Exact tile distribution — % of games whose max tile = each value
      3 (bot-left)  Game length box plot    — distribution of moves per game
      4 (bot-right) Mean moves per agent    — bar chart of average game length ± std

    Pass a dict mapping agent name → list of {max_tile, steps} dicts.

    Notebook usage:
        compare_agents({'DQN': eval_stats, 'SARSA': sarsa_stats, 'Beam': beam_stats})
    '''
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    methods    = list(results.keys())
    n_methods  = len(methods)
    colors     = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2'][:n_methods]

    cum_milestones  = [256, 512, 1024, 2048]
    exact_tiles     = [32, 64, 128, 256, 512, 1024, 2048]
    bar_width       = 0.7 / n_methods

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Agent Comparison', fontsize=14, fontweight='bold')
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ── Panel 1: Cumulative reach rate ────────────────────────────────────────
    x1 = np.arange(len(cum_milestones))
    for i, (method, stats) in enumerate(results.items()):
        n    = len(stats)
        pcts = [sum(1 for s in stats if s['max_tile'] >= t) / n * 100
                for t in cum_milestones]
        offset = x1 + i * bar_width - (n_methods - 1) * bar_width / 2
        bars = ax1.bar(offset, pcts, bar_width, label=method,
                       color=colors[i], alpha=0.85)
        for bar, pct in zip(bars, pcts):
            if pct > 4:
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.7,
                         f'{pct:.0f}%', ha='center', va='bottom', fontsize=7)
    ax1.set_xticks(x1)
    ax1.set_xticklabels([str(t) for t in cum_milestones])
    ax1.set_xlabel('Reached at least this tile')
    ax1.set_ylabel('% of games')
    ax1.set_title('Cumulative reach rate')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_ylim(0, 108)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.25)

    # ── Panel 2: Exact tile distribution ─────────────────────────────────────
    x2 = np.arange(len(exact_tiles))
    for i, (method, stats) in enumerate(results.items()):
        n    = len(stats)
        pcts = [sum(1 for s in stats if s['max_tile'] == t) / n * 100
                for t in exact_tiles]
        offset = x2 + i * bar_width - (n_methods - 1) * bar_width / 2
        bars = ax2.bar(offset, pcts, bar_width, label=method,
                       color=colors[i], alpha=0.85)
        for bar, pct in zip(bars, pcts):
            if pct > 3:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.5,
                         f'{pct:.0f}%', ha='center', va='bottom', fontsize=7)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([str(t) for t in exact_tiles])
    ax2.set_xlabel('Max tile reached (exact)')
    ax2.set_ylabel('% of games')
    ax2.set_title('Exact tile distribution')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_ylim(0, 108)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.25)

    # ── Panel 3: Game length box plot ─────────────────────────────────────────
    groups = [[s['steps'] for s in results[m]] for m in methods]
    bp = ax3.boxplot(groups, labels=methods, patch_artist=True,
                     medianprops=dict(color='black', linewidth=1.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax3.set_ylabel('Number of moves')
    ax3.set_title('Game length distribution')
    ax3.grid(axis='y', alpha=0.25)

    # ── Panel 4: Mean moves per agent (bar + error bar) ───────────────────────
    means = [np.mean([s['steps'] for s in results[m]]) for m in methods]
    stds  = [np.std( [s['steps'] for s in results[m]]) for m in methods]
    x4    = np.arange(n_methods)
    bars4 = ax4.bar(x4, means, color=colors, alpha=0.85,
                    yerr=stds, capsize=5,
                    error_kw=dict(elinewidth=1.2, ecolor='#333333'))
    for bar, mean, std in zip(bars4, means, stds):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + std + max(means) * 0.01,
                 f'{mean:.0f}', ha='center', va='bottom', fontsize=9,
                 fontweight='bold')
    ax4.set_xticks(x4)
    ax4.set_xticklabels(methods)
    ax4.set_ylabel('Average moves per game')
    ax4.set_title('Mean game length  (± 1 std)')
    ax4.grid(axis='y', alpha=0.25)

    # ── Console summary ───────────────────────────────────────────────────────
    header = f"{'Method':<18} {'n':>5} {'Avg steps':>10} {'≥2048':>8} {'≥1024':>8} {'≥512':>8}"
    print('\n' + header)
    print('─' * len(header))
    for method, stats in results.items():
        n   = len(stats)
        avg = np.mean([s['steps'] for s in stats])
        r2k = sum(1 for s in stats if s['max_tile'] >= 2048) / n * 100
        r1k = sum(1 for s in stats if s['max_tile'] >= 1024) / n * 100
        r5  = sum(1 for s in stats if s['max_tile'] >=  512) / n * 100
        print(f'{method:<18} {n:>5} {avg:>10.1f} {r2k:>7.1f}% {r1k:>7.1f}% {r5:>7.1f}%')
    print()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Comparison plot saved → {save_path}')
    plt.show()


def _run_one(game, policy) -> dict:
    '''Helper: run one episode with any state-based policy. Used by compare_agents example.'''
    s = game.get_initial_state()
    steps = 0
    while not game.is_terminal_state(s):
        a = policy(s)
        s = game.transition(s, a)
        steps += 1
    # WINNING_STATE is a sentinel outside the normal state range, so state_to_board()
    # returns None for it — handle it explicitly instead of calling get_state_status().
    max_tile = (TwntyFrtyEight.WINNING_VALUE
                if s == TwntyFrtyEight.WINNING_STATE
                else int(game.get_state_status(s)))
    return {'max_tile': max_tile, 'steps': steps}


# ---------------------------------------------------------------------------
# CLI / notebook entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import json, argparse

    parser = argparse.ArgumentParser(description='Train DQN agent for 2048')
    parser.add_argument('--games',               type=int,   default=15_000)
    parser.add_argument('--hidden',              type=int,   default=256)
    parser.add_argument('--lr',                  type=float, default=1e-4)
    parser.add_argument('--epsilon-start',       type=float, default=0.3)
    parser.add_argument('--epsilon-end',         type=float, default=0.05)
    parser.add_argument('--epsilon-decay',       type=float, default=0.9998)
    parser.add_argument('--batch',               type=int,   default=512)
    parser.add_argument('--buffer',              type=int,   default=100_000)
    parser.add_argument('--target-update-freq',  type=int,   default=1000)
    parser.add_argument('--n-envs',              type=int,   default=16,
                        help='Parallel environments for GPU training (default 16)')
    parser.add_argument('--print-every',         type=int,   default=100)
    parser.add_argument('--save',                type=str,   default='dqn_weights.pth')
    parser.add_argument('--load',                type=str,   default=None,
                        help='Path to existing weights to resume from')
    args = parser.parse_args()

    agent = DQNAgent(lr=args.lr, hidden_size=args.hidden,
                     buffer_capacity=args.buffer,
                     target_update_freq=args.target_update_freq)

    if args.load:
        agent.load(args.load)

    stats = agent.train(
        n_games       = args.games,
        epsilon_start = args.epsilon_start,
        epsilon_end   = args.epsilon_end,
        epsilon_decay = args.epsilon_decay,
        batch_size    = args.batch,
        print_every   = args.print_every,
        n_envs        = args.n_envs,
    )

    agent.save(args.save)

    stats_path = args.save.replace('.pth', '_train_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    print(f'Training stats saved → {stats_path}')
