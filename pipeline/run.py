from backtests import blend_signal, mean_reversion, momentum_backtest, exposure
from data import data_loader
from features import momentum, tstat, volatility, zscore
from regime import entropy, matrix, state_labels

def run_pipeline(ticker):
    df = data_loader.get_data(ticker)
    df = tstat.compute_tstat(df)
    df = momentum.calculate_momentum(df)
    df = zscore.calculate_zscore(df)
    df = volatility.compute_volatility(df)
    df = state_labels.label_volatility(df)
    df = state_labels.label_regime(df)
    df = state_labels.smooth_regime(df)
    df = matrix.initialize_state(df)
    df = entropy.build_expanding_entropy(df)
    df = exposure.normalize_entropy(df)
    df = mean_reversion.generate_signal(df)
    df = momentum_backtest.generate_signal(df)
    df = blend_signal.generate_signal(df)
    df = blend_signal.blended_backtest(df)
