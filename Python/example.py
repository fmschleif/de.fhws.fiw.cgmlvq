from cgmlvq import CGMLVQ

import csv
import numpy as np
import os


X = []
y = []

csv_file = open( os.path.join(os.getcwd(), "iris.csv") )

csv_reader = csv.reader( csv_file, delimiter=',' )

for row in csv_reader:

    length = len( row ) - 1

    vec = []

    for i in range( 0, length ):
        vec.append( row[i] )

    X.append( vec )
    y.append( row[length] )


X = np.array( X, dtype=float )
y = np.array( y, dtype=float )

X_train = X[0:120,:]
y_train = y[0:120]

X_test  = X[120:150,:]


cgmlvq = CGMLVQ( 2, 50 )
cgmlvq.fit( X_train, y_train )
cgmlvq.predict( X_test )