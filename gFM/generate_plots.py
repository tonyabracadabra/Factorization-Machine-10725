import glob, os
import scipy.io as sio
import numpy as np

os.chdir("../simulated_data/")

numtrain = 400
iterations = 100
rank_k = 10


def get_data(file):
    data = sio.loadmat(file)['data'][0]
    x, y, w0, w, W = data[0], data[1].reshape(-1), data[2], data[3], data[4]
    xtrain, ytrain, xtest, ytest = x[:numtrain,:], y[:numtrain], x[numtrain:,:], y[numtrain:]
    
    return xtrain, ytrain, xtest, ytest, w0, w, W


for i, file in enumerate(glob.glob("*.mat")):
    print file
    the_estimator = gFM.BatchRegression(rank_k=rank_k, max_init_iter=max_init_iter, learning_rate=0.001, init_tol=1e-4, tol=0, diag_zero=False)

    xtrain, ytrain, xtest, ytest, w0, w, W = get_data(file)
    ytruetrain = w0 + np.dot(xtrain, w) + np.expand_dims(np.diagonal(xtrain.dot(W).dot(xtrain.T)),axis=1)
    ytruetest = w0 + np.dot(xtest, w) + np.expand_dims(np.diagonal(xtest.dot(W).dot(xtest.T)),axis=1)
    
    y = (X.dot(w_true) + gFM.A_diag0(U_true, U_true, X.T)-0.5).flatten()
    
    trainset_error_record, testset_error_record = [], []
    for i in xrange(iterations):
        the_estimator.fit(xtrain, ytrain, n_more_iter=10)
        y_hat_grad_train = the_estimator.predict(xtrain)
        y_hat_grad_test = the_estimator.predict(xtest)

        trainerror = sklearn.metrics.mean_absolute_error(ytrain, y_hat_grad_train) / sklearn.metrics.mean_absolute_error(ytrain, numpy.zeros((len(ytrain),)))
        testerror = sklearn.metrics.mean_absolute_error(ytest, y_hat_grad_test) / sklearn.metrics.mean_absolute_error(ytest, numpy.zeros((len(ytest),)))
        trainset_error_record.append(trainerror)
        testset_error_record.append(testerror)
        
    plt.semilogy(range(100), trainset_error_record, '-b', label='train error')
    plt.semilogy(range(100), testset_error_record, '-r', label='test error')
    plt.legend()
    plt.savefig('../results/gFM/noisey/'+file+'_train_test_curve'+'.png')
    plt.close()