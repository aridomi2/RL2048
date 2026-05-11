'''
Deep Q-Network agent for 2048.

Based on Model 1 from:
  Li & Peng, "Playing 2048 With Reinforcement Learning" (2021)
  https://arxiv.org/abs/2110.10374

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
    Model 1 from Li & Peng 2021.
    Linear(256 → hidden) → ReLU → Dropout → Linear(hidden → 4)
    Output: Q-value for each of the 4 actions.
    '''
    def __init__(self, input_size: int = 256, hidden_size: int = 128,
                 output_size: int = 4, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
      - Experience replay (ReplayBuffer)
      - Soft-update target network (τ = 0.01 per step)
      - Epsilon-greedy exploration over valid actions only
      - MSE loss:  (r + γ · max_a Q_target(s',a) - Q(s,a))²
    '''

    def __init__(self, lr: float = 1e-4, gamma: float = 0.99,
                 hidden_size: int = 128, buffer_capacity: int = 10_000,
                 dropout: float = 0.2, tau: float = 0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.tau = tau

        self.online_net = DQNNet(hidden_size=hidden_size, dropout=dropout).to(self.device)
        self.target_net = DQNNet(hidden_size=hidden_size, dropout=dropout).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        print(f'DQNAgent ready | device={self.device} | '
              f'hidden={hidden_size} | buffer={buffer_capacity}')

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

    def _soft_update_target(self):
        for t_p, o_p in zip(self.target_net.parameters(),
                             self.online_net.parameters()):
            t_p.data.copy_(self.tau * o_p.data + (1 - self.tau) * t_p.data)

    def optimize(self, batch_size: int = 128) -> float | None:
        '''Sample one batch from replay buffer and do one gradient step.'''
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

        # Target: r + γ · max_a Q_target(s', a)  (0 if terminal)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._soft_update_target()

        return loss.item()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, n_games: int = 1000, epsilon_start: float = 0.3,
              epsilon_end: float = 0.05, epsilon_decay: float = 0.995,
              batch_size: int = 128, print_every: int = 100) -> list[dict]:
        '''
        Train for `n_games` episodes.

        Returns a list of per-game dicts with keys:
            game, max_tile, steps, total_reward
        Useful for plotting the training curve afterwards.
        '''
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

                # Reward: merge score + bonus when a new max tile appears
                bonus = int(np.max(next_board)) if np.max(next_board) > np.max(board) else 0
                reward = float(points + bonus)

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
        torch.save(self.online_net.state_dict(), path)
        print(f'Model saved → {path}')

    def load(self, path: str = 'dqn_weights.pth'):
        state_dict = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.target_net.eval()
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
    Side-by-side comparison of multiple agents.
    Pass a dict mapping agent name → list of {max_tile, steps} dicts.

    Notebook usage (after training DQN and running beam search):
        from game import TwntyFrtyEight
        from agent import Agent

        dqn_stats  = dqn_agent.evaluate(100)

        game   = TwntyFrtyEight()
        sarsa  = Agent(game)
        sarsa.w = np.load('w_star.npy')
        sarsa_stats = [_run_one(game, sarsa.greedy_policy) for _ in range(100)]
        beam_stats  = [_run_one(game, lambda s: sarsa.beam_search_policy(s, depth=10, k=5))
                       for _ in range(50)]

        compare_agents({'DQN': dqn_stats, 'SARSA': sarsa_stats, 'Beam': beam_stats})
    '''
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    methods  = list(results.keys())
    colors   = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2'][:len(methods)]
    milestones = [256, 512, 1024, 2048]
    x = np.arange(len(milestones))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Agent Comparison', fontsize=13, fontweight='bold')

    # Left: cumulative reach rates
    width = 0.7 / len(methods)
    for i, (method, stats) in enumerate(results.items()):
        n    = len(stats)
        pcts = [sum(1 for s in stats if s['max_tile'] >= t) / n * 100 for t in milestones]
        offset = x + i * width - (len(methods) - 1) * width / 2
        bars = ax1.bar(offset, pcts, width, label=method, color=colors[i], alpha=0.85)
        for bar, pct in zip(bars, pcts):
            if pct > 4:
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.6,
                         f'{pct:.0f}%', ha='center', va='bottom', fontsize=7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(t) for t in milestones])
    ax1.set_xlabel('Reached at least this tile')
    ax1.set_ylabel('% of games')
    ax1.set_title('Cumulative reach rate')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=9)

    # Right: game length box plot
    groups = [[s['steps'] for s in results[m]] for m in methods]
    bp = ax2.boxplot(groups, labels=methods, patch_artist=True,
                     medianprops=dict(color='black', linewidth=1.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax2.set_ylabel('Number of moves')
    ax2.set_title('Game length distribution')

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
    parser.add_argument('--games',         type=int,   default=2000)
    parser.add_argument('--hidden',        type=int,   default=128)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--epsilon-start', type=float, default=0.3)
    parser.add_argument('--epsilon-end',   type=float, default=0.05)
    parser.add_argument('--epsilon-decay', type=float, default=0.995)
    parser.add_argument('--batch',         type=int,   default=128)
    parser.add_argument('--buffer',        type=int,   default=10_000)
    parser.add_argument('--print-every',   type=int,   default=100)
    parser.add_argument('--save',          type=str,   default='dqn_weights.pth')
    parser.add_argument('--load',          type=str,   default=None,
                        help='Path to existing weights to resume from')
    args = parser.parse_args()

    agent = DQNAgent(lr=args.lr, hidden_size=args.hidden, buffer_capacity=args.buffer)

    if args.load:
        agent.load(args.load)

    stats = agent.train(
        n_games       = args.games,
        epsilon_start = args.epsilon_start,
        epsilon_end   = args.epsilon_end,
        epsilon_decay = args.epsilon_decay,
        batch_size    = args.batch,
        print_every   = args.print_every,
    )

    agent.save(args.save)

    stats_path = args.save.replace('.pth', '_train_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    print(f'Training stats saved → {stats_path}')
