import numpy as np
import pickle


def load_data_label():
    X = pickle.load(open("save_X.p", "rb"))
    Y = pickle.load(open("save_Y.p", "rb"))
    X = np.array(X)
    Y = np.array(Y)
    Y = Y.reshape(562, 6901)
    for x in X:
        x = np.reshape(x, (5, 6901, 1))
    # train: 404
    # val: 46
    # test: 112
    X_train = X[:404]
    Y_train = Y[:404]
    X_val = X[404:450]
    Y_val = Y[404:450]
    X_test = X[450:]
    Y_test = Y[450:]
    Y_train = Y_train.reshape(404, 6901)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


if __name__ == '__main__':
    load_data_label('')
