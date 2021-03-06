from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from scipy.io.matlab.mio import loadmat

import numpy as np
import seaborn as sns
from stat_test.quadratic_time import GaussianQuadraticTest
from tools.latex_plot_init import plt


from modshogun import QuadraticTimeMMD, GaussianKernel, RealFeatures

sns.set_style("whitegrid")

def prepare_dataset(X, y):
    N = len(X)
    train_test_ind = int(0.9 * N)
    
    inds = np.random.permutation(N)
    X = X[inds]
    y = y[inds]
    
    # spit into train and test
    X_test = X[train_test_ind:]
    y_test = y[train_test_ind:]
    X_train = X[:train_test_ind]
    y_train = y[:train_test_ind]
    N = len(X_train)
    N_test = len(X_test)
    
    # sort for easy plotting
    temp = X_test[:, 0].argsort()
    y_test = y_test[temp]
    X_test = X_test[temp]
    temp = X[:, 0].argsort()
    y = y[temp]
    X = X[temp]
    
    # normalise by training data statistics
    X_mean = np.mean(X_train)
    print X_mean
    X_std = np.std(X_train)
    print X_std
    X_train -= X_mean
    X_train /= X_std
    X_test -= X_mean
    X_test /= X_std
    
    
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train -= y_mean
    y_train /= y_std
    y_test -= y_mean
    y_test /= y_std
    
    return X_train, y_train, X_test, y_test, N, N_test

def bootstrap_null(U_matrix, num_bootstrap=1000):
    bootstrapped_stats = np.empty(num_bootstrap)
    N = U_matrix.shape[0]
    
    for i in range(num_bootstrap):
        W = np.sign(np.random.randn(N))
        WW = np.outer(W, W)
        st = np.mean(U_matrix * WW)
        bootstrapped_stats[i] = N * st
    
    return bootstrapped_stats

def compute_gp_regression_gradients(y_test, pred_mean, pred_std):
    return -(y_test - pred_mean) / pred_std ** 2

def sample_null_simulated_gp(s, pred_mean, pred_std, num_samples=1000):
    # this is slow and doesnt work better than bootstrapping
    samples = np.empty(num_samples)
    N = len(pred_mean)
    for i in range(num_samples):
        # simulate from predictive distribution and evaluate gradients at those points
        fake_y_test = np.random.randn(N) * pred_std + pred_mean
        fake_gradients = compute_gp_regression_gradients(fake_y_test, pred_mean, pred_std)
        
        # compute test statistic under this alt
        _, samples[i] = s.get_statistic_multiple_custom_gradient(fake_y_test[:, 0], fake_gradients[:, 0])
    
    return samples

def compare_against_mmd_test():
    data = loadmat("../data/02-solar.mat")
    X = data['X']
    y = data['y']
    
    X_train, y_train, X_test, y_test, N, N_test = prepare_dataset(X, y)
    
    kernel = RBF(input_dim=1, variance=0.608, lengthscale=0.207)
    m = GPRegression(X_train, y_train, kernel, noise_var=0.283)
    m.optimize()
    pred_mean, pred_std = m.predict(X_test)
    
    s = GaussianQuadraticTest(None)
    gradients = compute_gp_regression_gradients(y_test, pred_mean, pred_std)
    U_matrix, stat = s.get_statistic_multiple_custom_gradient(y_test[:, 0], gradients[:, 0])
    num_test_samples = 10000
    null_samples = bootstrap_null(U_matrix, num_bootstrap=num_test_samples)
#     null_samples = sample_null_simulated_gp(s, pred_mean, pred_std, num_test_samples)
    p_value_ours = 1.-np.mean(null_samples<=stat)

    y_rep = np.random.randn(len(X_test))*pred_std.flatten() + pred_mean.flatten()
    y_rep = np.atleast_2d(y_rep).T
    A = np.hstack((X_test, y_test))
    B = np.hstack((X_test, y_rep))
    feats_p = RealFeatures(A.T)
    feats_q = RealFeatures(B.T)
    width=1
    kernel=GaussianKernel(10, width);
    mmd=QuadraticTimeMMD();
    mmd.set_kernel(kernel)
    mmd.set_p(feats_p)
    mmd.set_q(feats_q)
    mmd_stat=mmd.compute_statistic()
    
    # sample from null
    num_null_samples = 10000
    mmd_null_samples = np.zeros(num_null_samples)
    for i in range(num_null_samples):
        # fix y_rep from above, and change the other one (that would replace y_test)
        y_rep2 = np.random.randn(len(X_test))*pred_std.flatten() + pred_mean.flatten()
        y_rep2 = np.atleast_2d(y_rep2).T
        A = np.hstack((X_test, y_rep2))
        feats_p = RealFeatures(A.T)
        width=1
        kernel=GaussianKernel(10, width);
        mmd=QuadraticTimeMMD();
        mmd.set_kernel(kernel)
        mmd.set_p(feats_p)
        mmd.set_q(feats_q)
        mmd_null_samples[i]=mmd.compute_statistic()
    
    p_value_mmd = 1.-np.mean(mmd_null_samples<=mmd_stat)
    
    return p_value_ours, p_value_mmd

if __name__ == '__main__':
    # repeat computing p.value multiple times for both methods
    num_repetitions = 100
    results = np.zeros((num_repetitions, 2))
    for i in range(num_repetitions):
        results[i,0], results[i,1] = compare_against_mmd_test()
        if i>1:
            print i
            print np.mean(results[:i], axis=0)
            print np.std(results[:i], axis=0)
     
    
    data = loadmat("../data/02-solar.mat")
    X = data['X']
    y = data['y']
    
    X_train, y_train, X_test, y_test, N, N_test = prepare_dataset(X, y)
    
    print "num_train:", len(X_train)
    print "num_test:", len(X_test)
    
    kernel = RBF(input_dim=1, variance=0.608, lengthscale=0.207)
    m = GPRegression(X_train, y_train, kernel, noise_var=0.283)
    m.optimize()
    
    res = 100
    pred_mean, pred_std = m.predict(X_test)
    X_test_plot = X_test[:,0]* 116.502738394 + 1815.93213296
    
    fig, ax = plt.subplots()
    plt.plot(X_test_plot, pred_mean, 'r-')
#     plt.plot(X_test, pred_mean + 2 * pred_std, 'b--')
#     plt.plot(X_test, pred_mean - 2 * pred_std, 'b--')
    # some hacks to make x axis ok again
    lower = (pred_mean - 2 * pred_std)[:,0]
    upper = (pred_mean + 2 * pred_std)[:,0]
    plt.fill_between(X_test_plot, lower, upper, color='r', alpha=0.3)
    plt.plot(X_train* 116.502738394 + 1815.93213296,  y_train,  'b.', markersize=3)
    plt.plot(X_test_plot, y_test, '*', color='black', markersize=5)
    plt.grid(True)
    plt.xlabel(r"Year")
    plt.ylabel(r"Solar activity (normalised)")
    
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 100))
    
    plt.savefig("gp_regression_data_fit.eps", bbox_inches='tight')
    plt.savefig("gp_regression_data_fit.pdf", bbox_inches='tight')
    exit()
    
    
    s = GaussianQuadraticTest(None)
    gradients = compute_gp_regression_gradients(y_test, pred_mean, pred_std)
    U_matrix, stat = s.get_statistic_multiple_custom_gradient(y_test[:, 0], gradients[:, 0])
    
    num_test_samples = 10000
    null_samples = bootstrap_null(U_matrix, num_bootstrap=num_test_samples)
    print "p-value:", 1.-np.mean(null_samples<=stat)
    
    plt.figure()
    sns.distplot(null_samples, kde=False, norm_hist=True)
    plt.plot([stat, stat], [0, .012], 'black')
    plt.legend([r"$V_n$ test", r"Bootstrapped $B_n$"])
    plt.xlabel(r"$V_n$")
    plt.ylabel(r"Frequency")
    plt.savefig("gp_regression_bootstrap_hist.eps", bbox_inches='tight')
    
    # compare to Lloyd & Gharamani
    # sample from GP, and perform MMD two sample test between test data and sampled data
    y_rep = np.random.randn(len(X_test))*pred_std.flatten() + pred_mean.flatten()
    y_rep = np.atleast_2d(y_rep).T
    
    # stack together (X_test,y_test) and (X_test, y_pred)
    A = np.hstack((X_test, y_test))
    B = np.hstack((X_test, y_rep))
    
    # compute MMD between (X_test,y_test) and (X_test, y_pred)
    feats_p = RealFeatures(A.T)
    feats_q = RealFeatures(B.T)
    width=1
    kernel=GaussianKernel(10, width);
    mmd=QuadraticTimeMMD();
    mmd.set_kernel(kernel)
    mmd.set_p(feats_p)
    mmd.set_q(feats_q)
    mmd_stat=mmd.compute_statistic()
    
    # sample from null
    num_null_samples = 10000
    mmd_null_samples = np.zeros(num_null_samples)
    for i in range(num_null_samples):
        # fix y_rep from above, and change the other one (that would replace y_test)
        y_rep2 = np.random.randn(len(X_test))*pred_std.flatten() + pred_mean.flatten()
        y_rep2 = np.atleast_2d(y_rep2).T
         
        A = np.hstack((X_test, y_rep2))
         
        feats_p = RealFeatures(A.T)
        width=1
        kernel=GaussianKernel(10, width);
        mmd=QuadraticTimeMMD();
        mmd.set_kernel(kernel)
        mmd.set_p(feats_p)
        mmd.set_q(feats_q)
        mmd_null_samples[i]=mmd.compute_statistic()
    
    print "p-value mmd:", 1.-np.mean(mmd_null_samples<=mmd_stat)
    plt.figure()
    sns.distplot(mmd_null_samples, kde=False, norm_hist=True)
    plt.plot([mmd_stat, mmd_stat], [0, 5], 'black')
    plt.legend([r"MMD$^2$ test", r"Null distribution"])
    plt.xlabel(r"MMD$^2$")
    plt.ylabel(r"Frequency")
    plt.savefig("gp_lloyd_gharamani_mmd.eps", bbox_inches='tight')
    
    plt.show()
