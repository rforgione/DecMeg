# This script imports our training data and uses it to create a classification 
# model using the stochastic gradient descent algorithm.


import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from scipy.io import loadmat
from sklearn.utils import shuffle
from sklearn.svm import SVC
import time


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
    test_subjects = range(17,24)

    # We throw away all the MEG data outside the first 0.5sec from when
    # the visual stimulus start:
    tmin = 0.0
    tmax = 0.5
    # print "Restricting MEG data to the interval [%s, %s]sec." % (tmin, tmax)

    X_all = []
    y_all = []

    X_test = []
    ids_test = []

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

    print X_all

    print "Importing test data."
    for subject in test_subjects:
        filename = 'data/test_subject%02d.mat' % subject
        print "Loading and processing %s" % filename
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        ids = data['Id']
        sfreq = data['sfreq']
        tmin_original = data['tmin']

        XX = create_features(XX, tmin, tmax, sfreq)

        X_test.append(XX)
        ids_test.append(ids)

    X_test = np.vstack(X_test)
    ids_test = np.concatenate(ids_test)

    print "Testset:", X_test.shape

    print "Performing mean normalization and feature scaling."
    scaler = StandardScaler()
    scaler.fit(X_all)
    X_all = scaler.transform(X_all)
    X_test = scaler.transform(X_test)

    clf = SGDClassifier(alpha=1.075, n_iter=100)

    # comment out the below two lines if you are only looking to produce a submission.
    print "Training and cross-validating."
    clf.fit(X_all, y_all)
    train_score = clf.score(X_all, y_all)
    cv_score = cross_val_score(clf, X_all, y_all, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2))

    alpha_performance = "alpha_performance.csv"
    cf = open(alpha_performance, 'a')
    # write to the alpha performance data file with the alpha value, the train set score,
    # and the cv score. Comment out the following lines if you do not want to store data
    # on alpha performance.
    print >> cf, str(clf.alpha) + "," + str(train_score) + "," + str(cv_score.mean())
    cf.close()

    y_pred = clf.predict(X_test)
    
    print y_pred

    filename_submission = "submission.csv"
    print "Creating submission file."
    f = open(filename_submission, "w")
    print >> f, "Id,Prediction"
    for i in range(len(y_pred)):
        print >> f, str(ids_test[i]) + "," + str(y_pred[i])
    f.close()

    end = time.time()
    elapsed = end - start
    print "The elapsed time for the script was %i seconds." % elapsed
    print "Done."
