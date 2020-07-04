import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_xor_data():
    x1 = np.random.random((100,2))
    x2 = np.random.random((100,2)) - np.array([1,1])
    x3 = np.random.random((100,2)) - np.array([1,0])
    x4 = np.random.random((100,2)) - np.array([0,1])
    x = np.vstack([x1,x2,x3,x4])
    Y = np.array([0]*200 + [1]*200)
    return x,Y

def main():
    X, Y = get_xor_data()
    plt.scatter(X[:,0],X[:,1],s=100,c=Y,alpha=0.5)
    plt.show()

    tsne = TSNE(perplexity=40)
    Z = tsne.fit_transform(X)
    plt.scatter(Z[:,0],Z[:,1],s=100,c=Y,alpha=0.5)
    plt.show()


if __name__ == '__main__':
    main()
