import numpy as np
from game import TwntyFrtyEight
from agent import Agent

def main():
    g = TwntyFrtyEight()
    # g.play()
    
    agent = Agent(environ=g)
    try:
        loaded_w = np.load('w_star.npy')
        if loaded_w.shape == agent.w.shape:
            agent.w = loaded_w
            print(f'Loaded weights from w_star.npy  shape={loaded_w.shape}')
        else:
            print(f'Warning: w_star.npy shape {loaded_w.shape} != expected '
                  f'{agent.w.shape}. Starting from zeros.')
    except FileNotFoundError:
        print('No w_star.npy found. Starting from zeros.')
    try:
        agent.find_optimal_weight(alpha=1e-11, tolerance=1e-20, alpha_decay=True)
    except KeyboardInterrupt:
        pass
    
    w_star = agent.w
    print(f'Optimal Weight: {w_star}')
    np.save('w_star.npy', w_star)

if __name__ == '__main__':
    main()