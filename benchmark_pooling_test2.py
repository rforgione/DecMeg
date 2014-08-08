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
from sklearn.cross_validation import train_test_split
# import the loadmat module for loading the data
from scipy.io import loadmat
from sklearn.utils import shuffle
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
    all_subjects = range(1,17)

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

    X_all_rand, y_all_rand = shuffle(X_all, y_all)

    X_train, X_cv, y_train, y_cv = train_test_split(X_all_rand, y_all_rand, train_size=0.8)

    print "Trainset:", X_train.shape
    print "CV Set:", X_cv.shape

    print "Performing mean normalization and feature scaling."
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_cv = scaler.transform(X_cv)

    clf = SGDClassifier(alpha=1, n_iter=100, loss="log", shuffle=False)
    # comment out the below two lines if you are only looking to produce a submission.
    print "Training the classifier."
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    cv_score = clf.score(X_cv, y_cv)

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
    
