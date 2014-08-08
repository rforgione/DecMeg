# Copyright Emanuele Olivetti 2014, BSD license, 3 clauses.

# import numpy
import numpy as np
# import the LogisticRegression module
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
# import the loadmat module for loading the data
from scipy.io import loadmat
from sklearn.utils import shuffle
from sklearn.svm import SVC
import time
import matplotlib as plt


def create_features(XX, tmin, tmax, sfreq, tmin_original=-0.5):
    beginning = np.round((tmin - tmin_original) * sfreq).astype(np.int)
    end = np.round((tmax - tmin_original) * sfreq).astype(np.int)
    XX = XX[:, :, beginning:end].copy()

    XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])

    return XX


if __name__ == '__main__':

    start = time.time()

    print "DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain"
    all_subjects = range(1,17)

    # We throw away all the MEG data outside the first 0.5sec from when
    # the visual stimulus start:
    tmin = 0.0
    tmax = 0.5
    # print "Restricting MEG data to the interval [%s, %s]sec." % (tmin, tmax)

    X_all = []
    y_all = []

    print "Importing all training data."
    for subject in all_subjects:
        filename = 'data/train_subject%02d.mat' % subject
        print "Loading and processing %s" % filename
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        yy = data['y']
        sfreq = data['sfreq']
        tmin_original = data['tmin']

        XX = create_features(XX, tmin, tmax, sfreq)

        X_all.append(XX)
        y_all.append(yy)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)


    print "Performing mean normalization and feature scaling."
    scaler = StandardScaler()
    scaler.fit(X_all)
    X_all = scaler.transform(X_all)

    alpha = np.array([.0001, .0003, .001, .003, .01, .03, .1, .3, 1, 3, 10, 30])
    accuracy = []

    for i in alpha:
        clf = SGDClassifier(alpha=i, n_iter=100, loss='log')
        clf.fit(X_all, y_all)
        scores = cross_val_score(clf, X_all, y_all, cv=5)
        accuracy.append(scores.mean())

    best_score = max(accuracy)
    best_score_index = accuracy.index(best_score)
    best_alpha = alpha[best_score_index]

    print "The best score was %0.4f, and was achieved by alpha=%0.4f" % (best_score, best_alpha)

    summary = np.vstack((alpha, accuracy)).T

    print summary

    end = time.time()
    elapsed = end - start
    print "The elapsed time for the script was %i seconds." % elapsed
    print "Done."
    
