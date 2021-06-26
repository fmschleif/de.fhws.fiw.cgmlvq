from ..cgmlvq import CGMLVQ
from math import sqrt
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import KFold

import csv
import numpy as np
import os
import unittest


# wrap in test class to start with intellij
class Evaluate_CGMLVQ( unittest.TestCase ):

    KNN_2CLASS_DATASET = "voting_sim.csv"
    CGMLVQ_2CLASS_DATASET = "voting_sim_embedded.csv"

    KNN_MULTICLASS_DATASET = ""
    CGMLVQ_MULTICLASS_DATASET = ""

    kf = KFold( n_splits=5, random_state=None, shuffle=False )


    def test_evaluate( self ):

        self.knn( self.KNN_2CLASS_DATASET )
        self.cgmlvq( self.CGMLVQ_2CLASS_DATASET )


    def knn_euclidean_distance( self, row1, row2 ):

        distance = 0.0

        for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])**2

        return sqrt(distance)


    def knn_get_neighbors( self, train, test_row, num_neighbors ):

        # Locate the most similar neighbors

        distances = list()

        for train_row in train:
            dist = self.knn_euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))

        distances.sort(key=lambda tup: tup[1])

        neighbors = list()

        for i in range(num_neighbors):
            neighbors.append(distances[i][0])

        return neighbors


    def knn_predict_classification( self, train, test_row, num_neighbors ):

        # Make a classification prediction with neighbors

        neighbors = self.knn_get_neighbors(train, test_row, num_neighbors)

        output_values = [row[-1] for row in neighbors]

        prediction = max(set(output_values), key=output_values.count)

        return prediction


    def knn_predict( self, train, test, neighbors ):

        predicted = []

        for test_row in test:
            predicted.append( self.knn_predict_classification(train, test_row, neighbors) )

        return predicted


    def load_dataset( self, dataset ):

        data = []

        csv_file = open( os.path.join(os.getcwd(), "Python\\data sets", dataset) )

        csv_reader = csv.reader( csv_file, delimiter=',' )

        for row in csv_reader:
            data.append( row )

        csv_file.close()

        return data


    def cgmlvq( self, dataset ):

        data = np.array( self.load_dataset(dataset) )

        X = np.array( data[:,:-1], dtype=np.cdouble )
        y = np.array( data[:,-1], dtype=int )

        for train_index, test_index in self.kf.split( X ):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            print( f" Aufteilung Durchgang: {np.asarray(np.unique(y_train,return_counts=True)).T}" )

            self.cgmlvq_run( X_train, X_test, y_train, y_test )


    def knn( self, dataset ):

        data = np.array( self.load_dataset(dataset) )

        for train_index, test_index in self.kf.split( data ):

            train = np.array(data[train_index], dtype=float)
            test = np.array(data[test_index], dtype=float)
            y_test = np.array(data[test_index,-1], dtype=int)

            print( f" Aufteilung Durchgang: {np.asarray(np.unique(train[:,-1],return_counts=True)).T}" )

            self.knn_run( train, test, y_test )


    def knn_run( self, train, test, y_test ):

        # kNN source
        # https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

        predicted = self.knn_predict( train, test, 1 )  # int(sqrt(len(train)))

        fpr, tpr, thresholds = roc_curve( y_test-1, np.array(predicted, dtype=int)-1 )  # must be 0 and 1 labels

        print( f"kNN (k={int(sqrt(len(train)))})" )
        print( f"---" )
        print( f"cm: {confusion_matrix(y_test, predicted)}" )
        print( f"acc: {accuracy_score(y_test, predicted)}" )
        print( f"prec: {precision_score(y_test, predicted)}" )
        print( f"rec: {recall_score(y_test, predicted)}" )
        print( f"f1: {f1_score(y_test, predicted)}" )
        print( f"auc: {auc(fpr, tpr)}" )


    def cgmlvq_run( self, X_train, X_test, y_train, y_test ):

        cgmlvq = CGMLVQ()
        cgmlvq.set_params()
        cgmlvq.fit( X_train, y_train )

        predicted = cgmlvq.predict( X_test )

        fpr, tpr, thresholds = roc_curve( y_test-1, predicted-1 )  # must be 0 and 1 labels

        print( f"CGMLVQ ({cgmlvq.get_params()})" )
        print( f"------" )
        print( f"cm: {confusion_matrix(y_test, predicted)}" )
        print( f"acc: {accuracy_score(y_test, predicted)}" )
        print( f"prec: {precision_score(y_test, predicted)}" )
        print( f"rec: {recall_score(y_test, predicted)}" )
        print( f"f1: {f1_score(y_test, predicted)}" )
        print( f"auc: {auc(fpr, tpr)}" )


    # =========================
    # =========================
    # Data set: voting_sim
    # Datensätze: 435
    # Feature vectors: 435
    # Verteilung: Klasse 1: 267
    #             Klasse 2: 168
    # Kreuzvalidierung: 5
    # =========================
    # =========================
    #
    #
    # Durchgang 1: Klasse 1: 215
    #              Klasse 2: 133
    #
    # kNN (k=1)
    # ---
    # cm: [[50  2]
    #      [ 0 35]]
    # acc: 0.9770114942528736
    # prec: 1.0
    # rec: 0.9615384615384616
    # f1: 0.9803921568627451
    # auc: 0.9807692307692308
    #
    # kNN (k=18)
    # ---
    # cm: [[49  3]
    #      [ 1 34]]
    # acc: 0.9540229885057471
    # prec: 0.98
    # rec: 0.9423076923076923
    # f1: 0.9607843137254902
    # auc: 0.9568681318681319
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[50  2]
    #      [ 2 33]]
    # acc: 0.9540229885057471
    # prec: 0.9615384615384616
    # rec: 0.9615384615384616
    # f1: 0.9615384615384616
    # auc: 0.9521978021978023
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': False, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[50  2]
    #      [ 2 33]]
    # acc: 0.9540229885057471
    # prec: 0.9615384615384616
    # rec: 0.9615384615384616
    # f1: 0.9615384615384616
    # auc: 0.9521978021978023
    #
    #
    # Durchgang 2: Klasse 1: 214
    #              Klasse 2: 134
    #
    # kNN (k=1)
    # ---
    # cm: [[48  5]
    #      [ 1 33]]
    # acc: 0.9310344827586207
    # prec: 0.9795918367346939
    # rec: 0.9056603773584906
    # f1: 0.9411764705882353
    # auc: 0.9381243063263042
    #
    # kNN (k=18)
    # ---
    # cm: [[50  3]
    #      [ 1 33]]
    # acc: 0.9540229885057471
    # prec: 0.9803921568627451
    # rec: 0.9433962264150944
    # f1: 0.9615384615384616
    # auc: 0.956992230854606
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[46  7]
    #      [ 0 34]]
    # acc: 0.9195402298850575
    # prec: 1.0
    # rec: 0.8679245283018868
    # f1: 0.9292929292929293
    # auc: 0.9339622641509434
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': False, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[46  7]
    #      [ 1 33]]
    # acc: 0.9080459770114943
    # prec: 0.9787234042553191
    # rec: 0.8679245283018868
    # f1: 0.9199999999999999
    # auc: 0.9192563817980023
    #
    #
    # Aufteilung Durchgang 3: Klasse 1: 211
    #                         Klasse 2: 137
    #
    # kNN (k=1)
    # ---
    # cm: [[54  2]
    #      [ 3 28]]
    # acc: 0.9425287356321839
    # prec: 0.9473684210526315
    # rec: 0.9642857142857143
    # f1: 0.9557522123893805
    # auc: 0.9337557603686636
    #
    # kNN (k=18)
    # ---
    # cm: [[55  1]
    #      [ 4 27]]
    # acc: 0.9425287356321839
    # prec: 0.9322033898305084
    # rec: 0.9821428571428571
    # f1: 0.9565217391304348
    # auc: 0.9265552995391705
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[55  1]
    #      [ 2 29]]
    # acc: 0.9655172413793104
    # prec: 0.9649122807017544
    # rec: 0.9821428571428571
    # f1: 0.9734513274336283
    # auc: 0.9588133640552995
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': False, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[55  1]
    #      [ 3 28]]
    # acc: 0.9540229885057471
    # prec: 0.9482758620689655
    # rec: 0.9821428571428571
    # f1: 0.9649122807017544
    # auc: 0.942684331797235
    #
    #
    # Aufteilung Durchgang 4: Klasse 1: 214
    #                         Klasse 2: 134
    #
    # kNN (k=1)
    # ---
    # cm: [[52  1]
    #      [ 5 29]]
    # acc: 0.9310344827586207
    # prec: 0.9122807017543859
    # rec: 0.9811320754716981
    # f1: 0.9454545454545454
    # auc: 0.9170366259711432
    #
    # kNN (k=18)
    # ---
    # cm: [[53  0]
    #      [ 1 33]]
    # acc: 0.9885057471264368
    # prec: 0.9814814814814815
    # rec: 1.0
    # f1: 0.9906542056074767
    # auc: 0.9852941176470589
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[51  2]
    #      [ 3 31]]
    # acc: 0.9425287356321839
    # prec: 0.9444444444444444
    # rec: 0.9622641509433962
    # f1: 0.9532710280373832
    # auc: 0.9370144284128745
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': False, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[52  1]
    #      [ 3 31]]
    # acc: 0.9540229885057471
    # prec: 0.9454545454545454
    # rec: 0.9811320754716981
    # f1: 0.9629629629629629
    # auc: 0.9464483906770255
    #
    #
    # Aufteilung Durchgang 5: Klasse 1: 214
    #                         Klasse 2: 134
    #
    # kNN (k=1)
    # ---
    # cm: [[46  7]
    #      [ 3 31]]
    # acc: 0.8850574712643678
    # prec: 0.9387755102040817
    # rec: 0.8679245283018868
    # f1: 0.9019607843137256
    # auc: 0.8898446170921198
    #
    # kNN (k=18)
    # ---
    # cm: [[47  6]
    #      [ 4 30]]
    # acc: 0.8850574712643678
    # prec: 0.9215686274509803
    # rec: 0.8867924528301887
    # f1: 0.9038461538461539
    # auc: 0.8845726970033296
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[45  8]
    #      [ 3 31]]
    # acc: 0.8735632183908046
    # prec: 0.9375
    # rec: 0.8490566037735849
    # f1: 0.8910891089108911
    # auc: 0.8804106548279689
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': False, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[45  8]
    #      [ 4 30]]
    # acc: 0.8620689655172413
    # prec: 0.9183673469387755
    # rec: 0.8490566037735849
    # f1: 0.8823529411764707
    # auc: 0.8657047724750278