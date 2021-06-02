from ..cgmlvq import CGMLVQ
from sklearn.metrics import confusion_matrix

import csv
import numpy as np
import os


data = []

csv_file = open( os.path.join(os.getcwd(), "data sets/iris.csv") )

csv_reader = csv.reader( csv_file, delimiter=',' )

for row in csv_reader:

    data.append( row )

csv_file.close()

data = np.array( data, dtype=np.cfloat )

X_train = data[0:120,0:4]
y_train = data[0:120,4]

X_test = data[120:150,0:4]
y_test = data[120:150,4]


cgmlvq = CGMLVQ()
cgmlvq.fit( X_train, y_train )
predicted = cgmlvq.predict( X_test )
cm = confusion_matrix( y_test, predicted )