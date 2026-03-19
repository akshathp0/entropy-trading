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
            idx1 = loadings.loc[ticker1].abs().idxmax()
            idx2 = loadings.loc[ticker2].abs().idxmax()
            if (loadings[idx1].loc[ticker1] > loadings[idx2].loc[ticker2]):
                top_pcas = iterate_pairs(loadings, top_pcas, idx2, ticker2)
            else:
                top_pcas = iterate_pairs(loadings, top_pcas, idx1, ticker1)
            
        flagged_pairs = flag_pairs(matrix, top_pcas)

    return top_pcas

def flag_pairs(matrix, top_pcas, corr_threshold = CORR_THRESHOLD):
    subset = matrix.loc[top_pcas, top_pcas]
    flagged_pairs = [(subset.index[i], subset.index[j]) for i,j in zip(*np.where(np.abs(subset.values) > corr_threshold)) if i < j]

    return flagged_pairs

def iterate_pairs(loadings, top_pcas, idx, ticker):
    n = 2
    top_pcas.remove(ticker)
    top_pca = loadings[idx].abs().nlargest(n).index[-1]

    while(top_pca in top_pcas):
        n += 1
        top_pca = loadings[idx].abs().nlargest(n).index[-1]
    
    top_pcas.append(top_pca)
    
    return top_pcas