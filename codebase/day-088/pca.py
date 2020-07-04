# https://deeplearningcourses.com/c/unsupervised-deep-learning-in-python
# https://www.udemy.com/unsupervised-deep-learning-in-python
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import tensorflow as tf

def main():
    (Xtrain, Ytrain), (Xtest, Ytest) =  tf.keras.datasets.mnist.load_data()
    
    (Xtrain,Ytrain), (Xtest,Ytest) = (Xtrain[:1000].flatten().reshape(1000,784),Ytrain[:1000]),(Xtest[:1000].flatten().reshape(1000,784),Ytest[:1000])
    pca = PCA()
    reduced = pca.fit_transform(Xtrain)
    plt.scatter(reduced[:,0], reduced[:,1], s=100, c=Ytrain, alpha=0.5)
    plt.show()

    plt.plot(pca.explained_variance_ratio_)
    plt.show()

    # cumulative variance
    # choose k = number of dimensions that gives us 95-99% variance
    cumulative = []
    last = 0
    for v in pca.explained_variance_ratio_:
        cumulative.append(last + v)
        last = cumulative[-1]
    plt.plot(cumulative)
    plt.show()

if __name__ == '__main__':
    main()
