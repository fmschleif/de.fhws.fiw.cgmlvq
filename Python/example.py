from CGMLVQ.cgmlvq import CGMLVQ

import csv


X = []
y = []


csv_file = open( "c:/Users/Matthias Nunn/Desktop/Projekte/de.fhws.fiw.cgmlvq/Python/iris.csv" )  # only "iris.csv" not working?!

csv_reader = csv.reader( csv_file, delimiter=',' )

for row in csv_reader:

    vec = [ row[0], row[1], row[2], row[3] ]

    X.append( vec )
    y.append( row[4] )


cgmlvq = CGMLVQ( 2, None, None )
cgmlvq.fit( X, y )