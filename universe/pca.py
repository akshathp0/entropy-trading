from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

VARIANCE_THRESHOLD = config['variance_threshold']
CORR_THRESHOLD = config['corr_threshold']

def fit_pca(returns, tickers):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(returns)

    pca = PCA()
    principal_components = pca.fit_transform(scaled)

    df = pd.DataFrame(pca.components_.T, index = tickers, columns = [f'PC{i+1}' for i in range(len(tickers))])

    return df, pca

def significant_components(p_comp, variance_threshold = VARIANCE_THRESHOLD):
    for i in range(len(p_comp.explained_variance_ratio_)):
        if p_comp.explained_variance_ratio_.cumsum()[i] >= variance_threshold:
            return i + 1

def top_pcas(loadings, n_components):
    top_pcas = []

    for i in range(n_components):
        col = loadings[f'PC{i + 1}']

        ranked_pcas = col[col > 0].sort_values(ascending = False)
        for ticker in ranked_pcas.index:
            if ticker not in top_pcas:
                top_pcas.append(ticker)
                break

        ranked_pcas = col[col > 0].sort_values(ascending = False)
        for ticker in ranked_pcas.index:
            if ticker not in top_pcas:
                top_pcas.append(ticker)
                break

    return top_pcas