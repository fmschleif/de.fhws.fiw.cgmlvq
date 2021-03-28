from cgmlvq import CGMLVQ
from sklearn.metrics import confusion_matrix

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


X = np.array( X )
y = np.array( y )

X_train = X[0:120,:]
y_train = y[0:120]

X_test = X[120:150,:]
y_test = y[120:150]


cgmlvq = CGMLVQ()
cgmlvq.fit( X_train, y_train )
predicted = cgmlvq.predict( X_test )
cm = confusion_matrix( y_test, predicted )

print( predicted )
print( cm )