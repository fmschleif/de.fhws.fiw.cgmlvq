from cgmlvq import CGMLVQ

import csv
import os


X = []
y = []

csv_file = open( os.path.join(os.getcwd(), "twoclass-simple.csv") )

csv_reader = csv.reader( csv_file, delimiter=',' )

for row in csv_reader:

    length = len( row ) - 1

    vec = []

    for i in range( 0, length ):
        vec.append( row[i] )

    X.append( vec )
    y.append( row[length] )


cgmlvq = CGMLVQ( 2, 50 )
cgmlvq.fit( X, y )