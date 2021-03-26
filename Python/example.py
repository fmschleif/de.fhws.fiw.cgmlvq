from cgmlvq import CGMLVQ
from sklearn.metrics import confusion_matrix

import csv
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


X_train = X[0:120,:]
y_train = y[0:120]

X_test = X[120:150,:]
y_test = y[120:150]


cgmlvq = CGMLVQ()
cgmlvq.fit( X, y )
predicted = cgmlvq.predict( X )

print( predicted )
print( confusion_matrix(y_test, predicted) )