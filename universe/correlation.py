import numpy as np
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

CORR_THRESHOLD = config['corr_threshold']

def corr_matrix(returns):
    return returns.corr()

def correlation_check(top_pcas, matrix, loadings):
    flagged_pairs = flag_pairs(matrix, top_pcas)

    while(len(flagged_pairs) != 0):
        for ticker1, ticker2 in flagged_pairs:
            if ticker1 not in top_pcas or ticker2 not in top_pcas:
                continue
            print(f'checking: {ticker1} and {ticker2}')
            idx1 = loadings.loc[ticker1].abs().idxmax()
            idx2 = loadings.loc[ticker2].abs().idxmax()
            if (loadings[idx1].loc[ticker1] > loadings[idx2].loc[ticker2]):
                top_pcas = iterate_pairs(loadings, top_pcas, idx2, ticker2, matrix)
            else:
                top_pcas = iterate_pairs(loadings, top_pcas, idx1, ticker1, matrix)
            
        flagged_pairs = flag_pairs(matrix, top_pcas)

    return top_pcas

def flag_pairs(matrix, top_pcas, corr_threshold = CORR_THRESHOLD):
    subset = matrix.loc[top_pcas, top_pcas]
    flagged_pairs = [(subset.index[i], subset.index[j]) for i,j in zip(*np.where(np.abs(subset.values) > corr_threshold)) if i < j]

    return flagged_pairs

def iterate_pairs(loadings, top_pcas, idx, ticker, matrix, threshold = CORR_THRESHOLD):
    col = loadings[idx]
    was_positive = col.loc[ticker] > 0
    top_pcas.remove(ticker)

    if was_positive:
        ranked_pcas = col[col > 0].sort_values(ascending = False)
    else:
        ranked_pcas = col[col > 0].sort_values(ascending = False)

    for ticker in ranked_pcas.index:
        if ticker not in top_pcas:
            correlations = matrix.loc[ticker, top_pcas].abs()
            if (correlations < threshold).all():
                top_pcas.append(ticker)
                break
    
    return top_pcas