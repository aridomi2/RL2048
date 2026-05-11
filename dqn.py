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
