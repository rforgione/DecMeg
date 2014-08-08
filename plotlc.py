from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import numpy as np
import benchmark_pooling as bp

def cost_size_rel(X_train, X_test, y_train, y_test, score_fn):
    clf = SGDClassifier(loss='log', alpha=1.05)

    train_errs, test_errs = [], []
    subset_sizes = np.exp(np.linspace(3, np.log(X_train.shape[0]), 20)).astype(int)

    index = 0

    for m in subset_sizes:
        
        clf.fit(X_train[:m], y_train[:m])

        train_err = 1 - clf.score(X_train[:m], y_train[:m])
        test_err = 1 - clf.score(X_test, y_test)

        train_errs.append(train_err)
        test_errs.append(test_err)

        index += 1

    return subset_sizes, train_errs, test_errs

def compute_error(y, yfit):
    m = y.shape[0]
    yfit = yfit[:,1]
    assert y.shape == yfit.shape
    J = -(1/m)*sum(y*np.log(yfit) + (1-y)*np.log(1 - yfit))
    return J

def plot_response(subset_sizes, train_errs, test_errs):
    print "Plotting the learning curves..."
    plt.plot(subset_sizes, train_errs, lw=2)
    plt.plot(subset_sizes, test_errs, lw=2)
    plt.legend(['Training Error', 'CV Error'])
    plt.xscale('log')
    plt.xlabel('Dataset Size')
    plt.ylabel('Error')
    plt.title('Model response to dataset size')
    plt.show()

if __name__ == "__main__":

    subjects = range(1,17)   

    X_all = []
    y_all = []

    X_train = []
    X_cv = []

    y_train = []
    y_cv = []

    subset_sizes = np.array([])
    train_errs = np.array([])
    test_errs = np.array([])

    tmin = 0
    tmax = 0.5

    for subject in subjects:
        filename = 'data/train_subject%02d.mat' % subject
        print "Loading and processing %s" % filename
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        yy = data['y']
        sfreq = data['sfreq']
        tmin_original = data['tmin']

        X_all.append(XX)
        y_all.append(yy)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    X_all = bp.create_features(X_all, tmin, tmax, sfreq, tmin_original)

    X_all, y_all = shuffle(X_all, y_all)

    X_train, X_cv, y_train, y_cv = train_test_split(X_all, y_all, train_size=.7)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_cv = scaler.transform(X_cv)

    score_fn = compute_error
    subset_sizes, train_errs, test_errs = cost_size_rel(X_train, X_cv, y_train, y_cv, score_fn)
    plot_response(subset_sizes, train_errs, test_errs)
    