from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

VARIANCE_THRESHOLD = config['variance_threshold']

def fit_pca(returns, tickers):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(returns)

    pca = PCA()
    principal_components = pca.fit_transform(scaled)

    df = pd.DataFrame(pca.components_.T, index = tickers, columns = [f'PC{i+1}' for i in range(len(tickers))])

    return df, pca

def significant_components(p_comp, variance_threshold = VARIANCE_THRESHOLD):
    for i, total in enumerate(p_comp.explained_variance_ratio_.cumsum()):
        if total >= variance_threshold:
            return i + 1