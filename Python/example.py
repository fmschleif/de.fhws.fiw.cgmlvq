from CGMLVQ.cgmlvq import CGMLVQ

import csv


X = []
y = []


csv_file = open( "c:/Users/Matthias Nunn/Desktop/Projekte/de.fhws.fiw.cgmlvq/Python/iris-small.csv" )  # only "iris-small.csv" not working?!

csv_reader = csv.reader( csv_file, delimiter=',' )

for row in csv_reader:

    vec = [ row[0], row[1], row[2], row[3] ]

    X.append( vec )
    y.append( row[4] )


cgmlvq = CGMLVQ( 2, 50 )
cgmlvq.fit( X, y )