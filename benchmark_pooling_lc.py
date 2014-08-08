# Copyright Emanuele Olivetti 2014, BSD license, 3 clauses.

# import numpy
import numpy as np
import matplotlib.pyplot as plt
# import the LogisticRegression module
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation
# import the loadmat module for loading the data
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
    all_subjects = range(1,2)
    test_subjects = range(17,18)

    # We throw away all the MEG data outside the first 0.5sec from when
    # the visual stimulus start:
    tmin = 0.0
    tmax = 0.5
    # print "Restricting MEG data to the interval [%s, %s]sec." % (tmin, tmax)

    X_all = []
    y_all = []

    X_train = []
    y_train = []

    X_cv = []
    y_cv = []

    X_cv1 = []
    X_cv2 = []

    y_cv1 = []
    y_cv2 = []

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

    X_train, y_train = shuffle(X_all, y_all)

    # X_train, X_cv, y_train, y_cv = train_test_split(X_all_rand, y_all_rand, train_size=0.6)
    # X_cv1, X_cv2, y_cv1, y_cv2 = train_test_split(X_cv, y_cv, train_size=0.5)


    print "Trainset:", X_train.shape
    # print "CV Set:", X_cv.shape

    # print "Importing test data."
    # for subject in test_subjects:
    #     filename = 'data/test_subject%02d.mat' % subject
    #     print "Loading and processing %s" % filename
    #     data = loadmat(filename, squeeze_me=True)
    #     XX = data['X']
    #     ids = data['Id']
    #     sfreq = data['sfreq']
    #     tmin_original = data['tmin']

    #     XX = create_features(XX, tmin, tmax, sfreq)

    #     X_test.append(XX)
    #     ids_test.append(ids)

    # X_test = np.vstack(X_test)
    # ids_test = np.concatenate(ids_test)

    # print "Testset:", X_test.shape

    print "Performing mean normalization and feature scaling."
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # X_cv = scaler.transform(X_cv)
    # X_cv2 = scaler.transform(X_cv2)
    # X_test = scaler.transform(X_test)

    clf = SGDClassifier(alpha=1, n_iter=100)
    learning_curve(clf, X_train, y_train, cv=3, scoring="f1", exploit_incremental_learning=True)
    # comment out the below two lines if you are only looking to produce a submission.
    # print "Training on training set, testing against cv set."
    # clf.fit(X_train, y_train)
    # train_score = clf.score(X_train, y_train)
    # cv_score = clf.score(X_cv, y_cv)

    # print "The training accuracy is %0.4f%%." % (train_score*100)
    # print "The cross validation accuracy is %0.4f%%." % (cv_score*100)

    # SVC_cost_filename = 'svc_cost.csv'
    # cf = open(SVC_cost_filename, 'a')
    # print >> cf, str(clf.C) + "," + str(train_score) + "," + str(cv_score)
    # cf.close()

    # param_file = 'param_file.csv'
    # pf = open(param_file, 'a')
    # print >> pf, str(clf.C) + "," + str(clf.get_params())
    # pf.close()

    # X_train_2 = np.vstack((X_train, X_cv1))
    # y_train_2 = np.vstack((y_train, y_cv1))

    # clf.fit(X_train_2, y_train_2)

    # cv_score_2 = clf.score(X_train_2, y_train_2)

    # print "The final cross validation accuracy is $0.4f%%." % (cv_score_2*100)
    # print "Retraining on entire set (training + cv)."
    # clf.fit(X_all_rand, y_all_rand)

    # y_pred = clf.predict(X_test)

    # filename_submission = "submission.csv"
    # print "Creating submission file."
    # f = open(filename_submission, "w")
    # print >> f, "Id,Prediction"
    # for i in range(len(y_pred)):
    #     print >> f, str(ids_test[i]) + "," + str(y_pred[i])
    # f.close()
    end = time.time()
    elapsed = end - start
    print "The elapsed time for the script was %i seconds." % elapsed
    print "Done."
    
