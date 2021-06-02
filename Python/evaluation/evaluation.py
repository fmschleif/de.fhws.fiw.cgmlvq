from math import sqrt
from sklearn.metrics import confusion_matrix

import csv
import numpy as np
import os


# start script in same directory

KNN_DATASET = "voting_sim.csv"
CGMLVQ_DATASET = "voting_sim_embedded.csv"

START_TEST = 348  # 80/20


def cgmlvq_euclid( X, W ):

    return np.linalg.norm( np.array([X-W]).T )**2


def cgmlvq_compute_costs( fvec, lbl, prot, plbl ):

    nfv = fvec.shape[0]
    npp = len( plbl )
    crout = np.zeros( (1,nfv) )

    for i in range( 0, nfv ):  # loop through examples

        fvi = fvec[i,:]  # actual example
        lbi = lbl[i]     # actual example

        # calculate squared distances to all prototypes
        dist = np.empty( (npp, 1) )  # define squared distances
        dist[:] = np.nan

        for j in range( 0, npp ):  # distances from all prototypes
            dist[j] = cgmlvq_euclid( fvi, prot[j,:] )

        # find the two winning prototypes
        correct   = np.where( np.array([plbl]) == lbi )[1]  # all correct prototype indices
        incorrect = np.where( np.array([plbl]) != lbi )[1]  # all wrong   prototype indices

        dJ, JJ = dist[correct].min(0), dist[correct].argmin(0)      # correct winner
        dK, KK = dist[incorrect].min(0), dist[incorrect].argmin(0)  # wrong winner

        # winner indices
        jwin = correct[JJ][0]
        kwin = incorrect[KK][0]

        # the class label according to nearest prototype
        crout[0, i] = plbl[jwin] * (dJ <= dK) + plbl[kwin] * (dJ > dK)

    return crout


def cgmlvq_classify( X_test, X_train, y_train ):

    lbl = np.ones( (1, X_test.shape[0]) ).T  # fake labels

    return cgmlvq_compute_costs( X_test, lbl, X_train, y_train )


def knn_euclidean_distance( row1, row2 ):

    distance = 0.0

    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2

    return sqrt(distance)


def knn_get_neighbors( train, test_row, num_neighbors ):

    # Locate the most similar neighbors

    distances = list()

    for train_row in train:
        dist = knn_euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))

    distances.sort(key=lambda tup: tup[1])

    neighbors = list()

    for i in range(num_neighbors):
        neighbors.append(distances[i][0])

    return neighbors


def knn_predict_classification( train, test_row, num_neighbors ):

    # Make a classification prediction with neighbors

    neighbors = knn_get_neighbors(train, test_row, num_neighbors)

    output_values = [row[-1] for row in neighbors]

    prediction = max(set(output_values), key=output_values.count)

    return prediction


def load_dataset( dataset ):

    data = []

    csv_file = open( f"../data sets/{dataset}" )

    csv_reader = csv.reader( csv_file, delimiter=',' )

    for row in csv_reader:

        data.append( row )

    csv_file.close()

    data = np.array( data, dtype=np.cfloat )

    return data, data.shape[0], data.shape[1]


def knn():

    # kNN source
    # https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/ 

    data, rows, columns = load_dataset( KNN_DATASET )

    train = data[0:START_TEST, :]
    test = data[START_TEST:rows, :]
    y_test = data[START_TEST:rows, -1]

    predicted = []

    for test_row in test:

        predicted.append( knn_predict_classification(train, test_row, int(sqrt(len(train)))) )

    cm = confusion_matrix( y_test, predicted )

    print( "knn confusion matrix" )
    print( cm )


def cgmlvq():

    data, rows, columns = load_dataset( CGMLVQ_DATASET )

    X_train = data[0:START_TEST, 0:columns]
    X_test = data[START_TEST:rows, 0:columns]

    y_train = np.array( data[0:START_TEST, -1], dtype=int )
    y_test = np.array( data[START_TEST:rows, -1], dtype=int )

    predicted = cgmlvq_classify( X_test, X_train, y_train )

    cm = confusion_matrix( y_test, predicted[0] )

    print( "cgmlvq confusion matrix" )
    print( cm )


knn()
cgmlvq()