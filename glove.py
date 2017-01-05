from scipy.sparse import *
import numpy as np
import pickle
import random
from sklearn.utils import shuffle

def weight_func(nval, alpha, nmax):
    ratio = np.power(nval/nmax, alpha);
    return min(1, ratio)

def main():
    print("loading cooccurrence matrix")
    with open('modelData/cooc.pkl', 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = 40
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))
    bs = np.random.normal(size=(cooc.shape[0], 1))
    ds = np.random.normal(size=(cooc.shape[1], 1))
    gradsq_x = np.ones((cooc.shape[0], embedding_dim), dtype=np.float64)
    gradsq_y = np.ones((cooc.shape[1], embedding_dim), dtype=np.float64)
    gradsq_b = np.ones((cooc.shape[0], 1),dtype=np.float64)
    gradsq_d = np.ones((cooc.shape[1], 1),dtype=np.float64)


    print("xs shape: ", xs.shape)

    eta = 0.05
    alpha = 0.75

    epochs = 15
    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        global_cost = 0;
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):

            inner_prod = (np.log(n) - np.dot(xs[ix, :], ys[jy, :]) - bs[ix] - ds[jy]);
            global_cost += weight_func(n, alpha, nmax)*(inner_prod**2);

            # Gradient calculations
            grad_x = weight_func(n, alpha, nmax)*inner_prod*ys[jy, :];
            grad_y = weight_func(n, alpha, nmax)*inner_prod*xs[ix, :];
            grad_bias = weight_func(n, alpha, nmax)*inner_prod

            # Variable updates (AdaGrad)
            xs[ix, :] += eta*grad_x/gradsq_x[ix, :];
            ys[jy, :] += eta*grad_y/gradsq_y[jy, :];

            bs[ix, :] += eta*grad_bias/gradsq_b[ix, :];
            ds[jy, :] += eta*grad_bias/gradsq_d[jy, :];

            # squared gradient updates
            gradsq_x[ix, :] += np.square(grad_x)
            gradsq_y[jy, :] += np.square(grad_y)
            gradsq_b[ix, :] += grad_bias ** 2
            gradsq_d[jy, :] += grad_bias ** 2

        print("Cost: ", global_cost);

    res_x = np.hstack((xs, bs));
    res_y = np.hstack((ys, ds));

    np.save('modelData/embeddings_nostop_'+str(embedding_dim), res_x)
    np.save('modelData/embeddings_nostop_'+str(embedding_dim)+'_sum', res_x+res_y)
    # np.save('modelData/embeddings_40_avg', (res_x+res_y)/2)




if __name__ == '__main__':
    main()
