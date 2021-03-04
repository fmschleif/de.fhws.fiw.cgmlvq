from scipy.fft import fft, ifft


class CGMLVQ:


    def __init__( self, coefficients, epochs, plbl ):  # TODO: rename pbbl

        self.coefficients = coefficients
        self.epochs = epochs
        self.plbl = plbl


    def fit( self, X, y ):

        rowLength = len( X[0] )

        numberOfClasses = len( set(y) )

        XFrequency = self.__fourier__( X, self.coefficients )

        gmlvq_system, training_curves, param_set = self.__single__( XFrequency, y, self.coefficients, self.plbl );  # TODO: rename return values


    def predict( self, X ):

        pass


    def __fourier__( self, X, coefficients ):

        # Y = fft(x, [], 2);
        # enabled = zeros(1,size(Y,2));
        # enabled(1:r+1) = 1; %preserve DC and r positive frequencies from low to high
        # Y = Y(:, enabled==1);
        pass


    def __single__( self ):

        pass