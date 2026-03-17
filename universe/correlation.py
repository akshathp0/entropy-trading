import pandas as pd

def corr_matrix(returns):
    return returns.corr()