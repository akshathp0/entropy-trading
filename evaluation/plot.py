import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(matrix):
    fig, ax = plt.subplots(figsize = (12, 10))
    sns.heatmap(matrix, annot = False, cmap = 'coolwarm', ax = ax)
    ax.set_title('Initial ETF Universe Heatmap')
    plt.tight_layout()

    return fig

def plot_pca_bars(loadings, n_components):
    fig, axes = plt.subplots(n_components, 1, figsize = (12, n_components * 5))
    for i, ax in enumerate(axes):
        sns.barplot(loadings[f'PC{i + 1}'], ax = ax)
        ax.set_title(f'PC{i + 1} Bars')

    plt.tight_layout()

    return fig