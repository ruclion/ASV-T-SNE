import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA


def tsne_plotter(data, label, save_png, title):
    n_labels = len(set(label))

    # tsne
    tsne = TSNE(n_components=2, init='pca', learning_rate=10, perplexity=12, n_iter=1000)
    transformed_data = tsne.fit_transform(data)

    # LDA
    # lda = LDA(n_components=2)
    # transformed_data = lda.fit_transform(data, label)

    # PCA
    # pca = PCA(n_components=2)
    # transformed_data = pca.fit_transform(data)

    plt.figure()
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], 10, c=label, cmap=plt.cm.Spectral, alpha=0.5)
    #plt.title(title)
    plt.savefig(save_png)


