'''
Benchmark all 2048 RL agents and compare their performance.

Usage:
    python benchmark.py                        # 100 games per method (skips beam/DQN by default for speed)
    python benchmark.py --games 200            # 200 games per method
    python benchmark.py --all                  # include beam search + DQN (requires trained weights)
    python benchmark.py --skip-sarsa           # skip SARSA
    python benchmark.py --beam-depth 10 --beam-k 5  # faster beam search params

Output:
    - Summary table printed to stdout
    - benchmark_results.png  (tile distribution + game length plots)
    - benchmark_results.json (raw per-game data for further analysis)
'''

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from collections import defaultdict

from game import TwntyFrtyEight
from agent import Agent


# ---------------------------------------------------------------------------
# Runners — each returns a list of {max_tile, steps} dicts
# ---------------------------------------------------------------------------

def run_random(n: int) -> list[dict]:
    game = TwntyFrtyEight()
    stats = []
    for _ in range(n):
        s = game.get_initial_state()
        steps = 0
        while not game.is_terminal_state(s):
            a = Agent(game).random_policy(s)
            s = game.transition(s, a)
            steps += 1
        stats.append({'max_tile': int(game.get_state_status(s)), 'steps': steps})
    return stats


def run_sarsa(n: int, weight_path: str = 'w_star.npy') -> list[dict] | None:
    game = TwntyFrtyEight()
    agent = Agent(game)
    try:
        loaded = np.load(weight_path)
        if loaded.shape == agent.w.shape:
            agent.w = loaded
        else:
            print(f'[SARSA] Weight shape mismatch ({loaded.shape} vs {agent.w.shape}). '
                  'Run train.py first. Using zero weights.')
    except FileNotFoundError:
        print(f'[SARSA] {weight_path} not found. Run train.py first. Using zero weights.')

    stats = []
    for _ in range(n):
        s = game.get_initial_state()
        steps = 0
        while not game.is_terminal_state(s):
            a = agent.greedy_policy(s)
            s = game.transition(s, a)
            steps += 1
        stats.append({'max_tile': int(game.get_state_status(s)), 'steps': steps})
    return stats


def run_beam(n: int, depth: int = 20, k: int = 10) -> list[dict]:
    game = TwntyFrtyEight()
    agent = Agent(game)
    stats = []
    for i in range(n):
        s = game.get_initial_state()
        steps = 0
        while not game.is_terminal_state(s):
            a = agent.beam_search_policy(s, depth=depth, k=k)
            s = game.transition(s, a)
            steps += 1
        stats.append({'max_tile': int(game.get_state_status(s)), 'steps': steps})
        if (i + 1) % 10 == 0:
            print(f'  beam search: {i+1}/{n} games done')
    return stats


def run_dqn(n: int, weight_path: str = 'dqn_weights.pth') -> list[dict] | None:
    try:
        from dqn import DQNAgent
    except ImportError:
        print('[DQN] Could not import dqn.py. Skipping.')
        return None
    agent = DQNAgent()
    try:
        agent.load(weight_path)
    except FileNotFoundError:
        print(f'[DQN] {weight_path} not found. Run "python dqn.py" first. Skipping.')
        return None
    return [agent.play_game(epsilon=0.0) for _ in range(n)]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

TILE_MILESTONES = [128, 256, 512, 1024, 2048]
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']


def plot_results(results: dict[str, list[dict]], n: int, save_path: str = 'benchmark_results.png'):
    methods = list(results.keys())
    n_methods = len(methods)
    colors = COLORS[:n_methods]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'2048 RL Agent Benchmark  ({n} games each)', fontsize=13, fontweight='bold')

    # ── Plot 1: Max tile distribution (grouped bar chart) ──────────────────
    ax = axes[0]
    tile_labels = [str(t) for t in TILE_MILESTONES]
    x = np.arange(len(TILE_MILESTONES))
    width = 0.7 / n_methods

    for i, (method, stats) in enumerate(results.items()):
        pcts = [sum(1 for s in stats if s['max_tile'] >= t) / n * 100
                for t in TILE_MILESTONES]
        bars = ax.bar(x + i * width - (n_methods - 1) * width / 2,
                      pcts, width, label=method, color=colors[i], alpha=0.88)
        for bar, pct in zip(bars, pcts):
            if pct > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{pct:.0f}%', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(tile_labels)
    ax.set_xlabel('Reached at least this tile')
    ax.set_ylabel('% of games')
    ax.set_title('Tile Reach Rate (cumulative)')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)

    # ── Plot 2: Game length box plot ───────────────────────────────────────
    ax = axes[1]
    data   = [s['steps'] for stats in results.values() for s in stats]  # unused — just for ref
    groups = [[s['steps'] for s in stats] for stats in results.values()]
    bp = ax.boxplot(groups, labels=methods, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel('Number of moves')
    ax.set_title('Game Length Distribution')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Plot saved → {save_path}')


def plot_dqn_training_curve(stats_path: str = 'dqn_weights_train_stats.json',
                             window: int = 100,
                             save_path: str = 'dqn_training_curve.png'):
    '''
    Plot the DQN training curve from the JSON file saved by dqn.py.
    Call this separately after DQN training:
        python benchmark.py --dqn-curve
    '''
    try:
        with open(stats_path) as f:
            stats = json.load(f)
    except FileNotFoundError:
        print(f'{stats_path} not found. Train DQN first.')
        return

    steps   = [s['steps'] for s in stats]
    rewards = [s['total_reward'] for s in stats]
    games   = [s['game'] for s in stats]

    smoothed_steps   = np.convolve(steps,   np.ones(window) / window, mode='valid')
    smoothed_rewards = np.convolve(rewards, np.ones(window) / window, mode='valid')
    x = games[window - 1:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle('DQN Training Curve', fontsize=13, fontweight='bold')

    ax1.plot(x, smoothed_steps, color='#4C72B0', linewidth=1.5)
    ax1.set_ylabel(f'Moves per game\n(rolling avg {window})')
    ax1.grid(alpha=0.3)

    ax2.plot(x, smoothed_rewards, color='#DD8452', linewidth=1.5)
    ax2.set_ylabel(f'Total reward\n(rolling avg {window})')
    ax2.set_xlabel('Episode')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'DQN training curve saved → {save_path}')


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: dict[str, list[dict]]):
    header = f"{'Method':<18} {'Avg Steps':>10} {'≥2048':>8} {'≥1024':>8} {'≥512':>8} {'≥256':>8}"
    print('\n' + header)
    print('─' * len(header))
    for method, stats in results.items():
        n = len(stats)
        avg  = np.mean([s['steps'] for s in stats])
        r2k  = sum(1 for s in stats if s['max_tile'] >= 2048) / n * 100
        r1k  = sum(1 for s in stats if s['max_tile'] >= 1024) / n * 100
        r512 = sum(1 for s in stats if s['max_tile'] >=  512) / n * 100
        r256 = sum(1 for s in stats if s['max_tile'] >=  256) / n * 100
        print(f'{method:<18} {avg:>10.1f} {r2k:>7.1f}% {r1k:>7.1f}% {r512:>7.1f}% {r256:>7.1f}%')
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark 2048 RL agents')
    parser.add_argument('--games',        type=int,   default=100,
                        help='Games per method (default 100)')
    parser.add_argument('--all',          action='store_true',
                        help='Include beam search and DQN (requires trained weights / time)')
    parser.add_argument('--skip-random',  action='store_true')
    parser.add_argument('--skip-sarsa',   action='store_true')
    parser.add_argument('--beam',         action='store_true',
                        help='Include beam search (slow; use --beam-depth/k to tune)')
    parser.add_argument('--beam-depth',   type=int,   default=20)
    parser.add_argument('--beam-k',       type=int,   default=10)
    parser.add_argument('--dqn',          action='store_true',
                        help='Include DQN (requires dqn_weights.pth)')
    parser.add_argument('--dqn-weights',  type=str,   default='dqn_weights.pth')
    parser.add_argument('--dqn-curve',    action='store_true',
                        help='Plot DQN training curve and exit')
    parser.add_argument('--dqn-stats',    type=str,   default='dqn_weights_train_stats.json')
    parser.add_argument('--out',          type=str,   default='benchmark_results.png')
    args = parser.parse_args()

    if args.dqn_curve:
        plot_dqn_training_curve(args.dqn_stats)
        exit(0)

    results = {}
    n = args.games

    if not args.skip_random:
        print(f'Running Random ({n} games)…')
        results['Random'] = run_random(n)

    if not args.skip_sarsa:
        print(f'Running SARSA ({n} games)…')
        results['SARSA'] = run_sarsa(n)

    if args.beam or args.all:
        print(f'Running Beam Search depth={args.beam_depth} k={args.beam_k} ({n} games)…')
        results['Beam Search'] = run_beam(n, args.beam_depth, args.beam_k)

    if args.dqn or args.all:
        print(f'Running DQN ({n} games)…')
        dqn_stats = run_dqn(n, args.dqn_weights)
        if dqn_stats:
            results['DQN'] = dqn_stats

    if not results:
        print('Nothing to benchmark. Run with --all or specific flags.')
        exit(1)

    print_summary(results)
    plot_results(results, n, save_path=args.out)

    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f)
    print('Raw results saved → benchmark_results.json')
