## ReadMe

### DecMeg2014 : Decoding the Human Brain

### Objective: Predict visual stimuli from MEG recordings of human brain activity

From the Kaggle.com competition page:

Understanding how the human brain works is a primary goal in neuroscience research. Non-invasive functional neuroimaging techniques, such as magnetoencephalography (MEG), are able to capture the brain activity as multiple timeseries. When a subject is presented a stimulus and the concurrent brain activity is recorded, the relation between the pattern of recorded signal and the category of the stimulus may provide insights on the underlying mental process. Among the approaches to analyse the relation between brain activity and stimuli, the one based on predicting the stimulus from the concurrent brain recording is called brain decoding.

The goal of this competition is to predict the category of a visual stimulus presented to a subject from the concurrent brain activity. The brain activity is captured with an MEG device which records 306 timeseries at 1KHz of the magnetic field associated with the brain currents. The categories of the visual stimulus for this competition are two: face and scrambled face. A stimulus and the concurrent MEG recording is called trial and thousands of randomized trials were recorded from multiple subjects. The trials of some of the subjects, i.e. the train set, are provided to create prediction models. The remaining trials, i.e. the test set, belong to different subjects and they will be used to score the prediction models. Because of the variability across subjects in brain anatomy and in the patterns of brain activity, a certain degree of difference is expected between the data of different subjects and thus between the train set and the test set.


### Entry by Rob Forgione:

To run the code, you first need to download the data from the following link:

http://www.kaggle.com/c/decoding-the-human-brain/data

Extract all of the zip files into the data folder in the root directory.

Then, from the root directory, run the 'benchmark_pooling.py' script. The script will import the data, restrict the time segment, fit the data using Stochastic Gradient Descent with a 'hinge' loss function (i.e. the same one used for Support Vector Machines), then perform 5-fold cross validation on the training set. The function will print the cross validation accuracy and the elapsed time for the script. It will write the binary predictions to a file called 'submission.csv'

The benchmark_pooling.py script is partially based on the starter code provided by Emmanuel Olivetti. His code was specifically used to take the original 3D matrix provided with the script, reduce it down to 2 dimensions using the np.reshape command. 

The code was able to product ~67% accuracy on the test data, good for 103rd place, placing me in the top 38% of competitors. If presented with a similar competition in the future, I would try to implement an ensemble method (such as stacked generalization) to this data, as the structure of the data (17 participants, many trials per participant) makes for a natural structuring of data into 16 'mini-classifiers' which could then me combined into one 'super-classifier'. Perhaps I will use this strategy in future kaggle competitions. 