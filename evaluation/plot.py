import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(matrix):
    fig, ax = plt.subplots(figsize = (12, 10))
    sns.heatmap(matrix, annot = False, cmap = 'coolwarm', ax = ax)
    ax.set_title('Initial ETF Universe Heatmap')
    plt.tight_layout()

    return fig

