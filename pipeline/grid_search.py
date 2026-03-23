import run
import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

ALT_GAMMAS = config['alt_gammas']
ALT_ENTROPY_MODES = config['alt_entropy_modes']
ALT_ENTROPY_WINDOWS = config['alt_entropy_windows']

def run_grid_search(tickers, gamma, entropy_mode, entropy_window):
    config['entropy_confidence'] = gamma
    config['rolling_window'] = entropy_window
    config['entropy_mode'] = entropy_mode

    results = {}
    for ticker in tickers:
        results[ticker] = run.run_pipeline(ticker)
    
    return results