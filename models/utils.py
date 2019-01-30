import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import numpy as np

def plot_ACF(X, nlags):
    X = np.array(X)
    N = X.shape[1]
    plt.figure(figsize=(10,N*4))
    for i in range(N):
        X_acf = acf(X[:, i], nlags = nlags) # Compute the ACF
        plt.subplot(N, 1, i+1)
        plt.plot(X_acf)
        plt.title("ACF for theta[{}]".format(i))
        plt.ylabel("ACF(theta[{}])".format(i))
        plt.xlabel("Lags")
        plt.grid(True)
        plt.ylim(X_acf.mean() - X_acf.std(),X_acf.mean() + X_acf.std()) # Close limit
        plt.xlim(0, nlags)
        plt.hlines(0, 0, nlags)
        plt.tight_layout()
    plt.subplots_adjust(hspace = 0.3)
