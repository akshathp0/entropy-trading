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
        n = 1
        top_pca = loadings[f'PC{i + 1}'].abs().idxmax()
        while(top_pca in top_pcas):
            top_pca = loadings[f'PC{i + 1}'].abs().nlargest(n).index[-1]
            n += 1

        top_pcas.append(top_pca)

    return top_pcas