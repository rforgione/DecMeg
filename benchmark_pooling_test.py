# DecMeg2014 example code.

# Simple prediction of the class labels of the test set by:
# - pooling all the training trials of all subjects in one dataset.
# - Extracting the MEG data in the first 500ms from when the
#   stimulus starts.
# - Using a linear classifier (logistic regression).

# Copyright Emanuele Olivetti 2014, BSD license, 3 clauses.

# import numpy
import numpy as np
# import the LogisticRegression module
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
# import the loadmat module for loading the data
from scipy.io import loadmat
import time


def create_features(XX, tmin, tmax, sfreq, tmin_original=-0.5):
    """Creation of the feature space:
    - restricting the time window of MEG data to [tmin, tmax]sec.
    - Concatenating the 306 channels of each trial in one long
      vector.
    - Normalizing each feature independently (z-scoring).
    """
    # print "Applying the desired time window."
    # beginning = 0.0 - (-0.5) * 250 = 0.5 * 250 = entry 125
    beginning = np.round((tmin - tmin_original) * sfreq).astype(np.int)
    # end = 0.5 - (-0.5) * 250 = 1.0 * 250 = entry 250
    end = np.round((tmax - tmin_original) * sfreq).astype(np.int)
    # restricts the z-dimension to 125 elements long
    XX = XX[:, :, beginning:end].copy()

    # print "2D Reshaping: concatenating all 306 timeseries."
    # reshape XX to a 2D data matrix of 590 x 38250
    # 590 x (306 channels x 125 time series) = 590 x 38250
    # This takes the 306 vectors on the z-axis and 'straights them out'
    # i.e. time 1 sensor 1, time 1 sensor 2, time 1 sensor 3, time 2 sensor 1, time 2 sensor 2, etc...
    XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])

    # print "Features Normalization."
    # XX -= XX.mean(0)
    # XX = np.nan_to_num(XX / XX.std(0))

    return XX


if __name__ == '__main__':

    start = time.time()

    print "DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain"
    print
    subjects_train = range(1, 17) # we use 1-11 as our subjects in the train set
    # subjects_cv = range(12, 17) # we use 12-16 as our subjects in the crossval set
    subjects_test = range(17, 24) # we use 17-23 as our subjects in the test set
    print "Training on subjects", subjects_train 

    # We throw away all the MEG data outside the first 0.5sec from when
    # the visual stimulus start:
    tmin = 0.0
    tmax = 0.5
    # print "Restricting MEG data to the interval [%s, %s]sec." % (tmin, tmax)

    X_train = []
    y_train = []

    X_cv = []
    y_cv = []

    X_test = []
    ids_test = []

    print "Creating the trainset."
    for subject in subjects_train:
        filename = 'data/train_subject%02d.mat' % subject
        print "Loading and processing %s\r" % filename
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        yy = data['y']
        sfreq = data['sfreq']
        tmin_original = data['tmin']
        # print "Dataset summary:",

        XX = create_features(XX, tmin, tmax, sfreq)

        X_train.append(XX)
        y_train.append(yy)
    # at this point we have a list of np.ndarray objects
    # X_train = [train_subject1.mat,...,train_subjectm.mat]
    # we use np.vstack(X_train) to stack these into one cohesive m x n dataset
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    print "Trainset:", X_train.shape

    # print "Creating the cross validation set."
    # for subject in subjects_cv:
    #     filename = 'data/train_subject%02d.mat' % subject
    #     print "Loading and processing %s\r" % filename
    #     data = loadmat(filename, squeeze_me=True)
    #     XX = data['X']
    #     yy = data['y']
    #     sfreq = data['sfreq']
    #     tmin_original = data['tmin']

    #     XX = create_features(XX, tmin, tmax, sfreq)

    #     X_cv.append(XX)
    #     y_cv.append(yy)

    # X_cv = np.vstack(X_cv)
    # y_cv = np.concatenate(y_cv)
    # print "CV Set:", X_train.shape

    print "Creating the testset."
    for subject in subjects_test:
        filename = 'data/test_subject%02d.mat' % subject
        print "Loading", filename
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        ids = data['Id']
        sfreq = data['sfreq']
        tmin_original = data['tmin']
        # print "Dataset summary:"
        # print "XX:", XX.shape
        # print "ids:", ids.shape
        # print "sfreq:", sfreq

        XX = create_features(XX, tmin, tmax, sfreq)

        X_test.append(XX)
        ids_test.append(ids)

    X_test = np.vstack(X_test)
    ids_test = np.concatenate(ids_test)
    print "Testset:", X_test.shape

    print "Performing mean normalization and feature scaling."
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = SGDClassifier(alpha=3, n_iter=100, loss="log", shuffle=True)
    # comment out the below two lines if you are only looking to produce a submission.
    print "Training the classifier."
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    cv_score = clf.score(X_cv, y_cv).mean()

    print "The training accuracy is %0.4f%%." % (train_score*100)
    print "The cross validation accuracy is %0.4f%%." % (cv_score*100)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

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