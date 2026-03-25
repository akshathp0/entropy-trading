import numpy as np
import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

BLOCK_SIZE = config['block_size']
N_SIMULATIONS = config['n_simulations']

def run_monte_carlo(portfolio, spy, block_size = BLOCK_SIZE, n = N_SIMULATIONS):
    np.random.seed(42)
    returns = portfolio['Blend Return'].values
    spy_returns = spy['Pct Return'].values

    strat_sim = []
    spy_sim = []

    for _ in range(n):
        indices = np.concatenate([np.arange(i, min(i + block_size, len(returns)))
                                  for i in np.random.randint(0, len(returns) - block_size,
                                  size = len(returns) // block_size)])
        
        strat_sim.append(np.prod(1 + returns[indices]))
        spy_sim.append(np.prod(1 + spy_returns[indices]))
        
    return strat_sim, spy_sim
