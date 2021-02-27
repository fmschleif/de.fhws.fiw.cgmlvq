function [ newY, S ] = resampleData( originalX, originalY, newX, mode )
%RESAMPLEDATA Resample function x -> y at new x values.
%   Detailed explanation goes here

%% Check input arguments
if ( nargin < 4 || isempty( mode ) || ~ischar( mode ) )
    mode = 'linear';
end;
if ( ~strcmp( mode, 'repeat' ) && ...
     ~strcmp( mode, 'linear' ) && ...
     ~strcmp( mode, 'chebyshev' ) )
    warning( [ '"' mode '" interpolation mode is not supported yet. ' ...
        'Defaulting to linear mode' ] );
    mode = 'linear';
end;

if ( nargin < 3 || isempty( newX ) || ~isnumeric( newX ) )
    error( 'Incompatible values for resampling' );
end;

if ( nargin < 2 || isempty( originalY ) || ~isnumeric( originalY ) )
    error( 'No or incompatible values for original y' );
end;

if ( nargin < 1 || isempty( originalX ) || ~isnumeric( originalX ) )
    error( 'No or incompatible values for original x' );
end;

%% Check data dimensions
if ( length( originalX ) ~= size( originalY, 2 ) )
    error( 'Dimension mismatch for input x and y' );
end;
% newX has to be a row vector
if ( ~isrow( newX ) );
    newX = newX';
end;


%% Check if data is ordered
if ( ~issorted( originalX ) )
    warning( [ 'Data points are not sorted in ascending order. '...
        'Data is sorted now. The interpolation matrix is only valid ' ...
        'sorted data, not on the original data!' ] );
    [ originalX, orderedIndices ] = sort( originalX );
    originalY = originalY(orderedIndices );    
end;

if ( ~issorted( newX ) )
    warning( [ 'New x values are not sorted in ascending order. '...
        'Values sorted now!' ] );
    newX = sort( newX );
end;

%% Interpolate the data
% We assume, the data is ordered in ascending order.
% In dependence of the chosen interpolation mode, the values of the
% function are calculated on the new x values.
% Till now, only an straight linear approximation is implemented.

newY = zeros( size(originalY, 1 ), length( newX ) );
S = zeros( length( newX ), length( originalX ) );


for x = newX
    % Iff we have the very special case, that the new x is equal to one of
    % the x values we already know, no calculation is neccessary.
    if ( any( x == originalX ) )
        S( x == newX, x == originalX ) = 1.0;
        continue;
    end;

    % The more regular case, x is anywhere in between two known x values.
    % Thus we need to do some interpolation in order to calculate a
    % meaningful value for y.    
    switch ( mode )
        case 'linear'
            % The only one implemented at this point.
            % Interpolation is done using formula
            % $$y = m \cdot x + n $$
            % with 
            % $$ m = \delta_y / \delta_x $$
            % and
            % $$ n = y_1 $$;
            %
            % Having two points $x_1, x_2$ with $x_1 < x_2$ with their
            % corresponding values $y_1, y_2$, and a value $x_n$ (n = new)
            % where the value $y_n$ is unknown, everything cuts down to:
            % $$ y_n = y_1 + \frac{ \left( y_2 - y_1 \right) \cdot  
            % \left( x_n - x_1 \right) }{x_2 - x_1}  $$
            % 
            % For having a matrix representation of the resampling it would
            % be very nice to have an equation of the form:
            % $$ y_n = c_1 \cdot y_1 + c_2 \cdot y_2 $$
            % Reformulation of the equation above gives the equation
            % $$ y_n = \frac{ x_2 - x_n }{ x_2 - x_1 } \cdot y_1 + 
            % \frac{ x_n - x_1 }{ x_2 - x_1 } \cdot y_2 $$
            
            % find closest x value smaller and greater than x in the vector of
            % original x values. and their corresponding y values.
            smallerIndex = find( x >= originalX, 1, 'last' );
            greaterIndex = find( x <= originalX, 1, 'first' );
            
            % Special case extrapolation
            if ( isempty( smallerIndex ) )
                % x is smaller than all originalX
                %
                % Take the two smallest values
                smallerIndex = 1;
                greaterIndex = 2;
            elseif ( isempty( greaterIndex ) )
                greaterIndex = length( originalX );
                smallerIndex = length( originalX ) - 1;
            end;

            % do some interpolation
            x1 = originalX( smallerIndex );
            x2 = originalX( greaterIndex );
            xn = x;

            c1 = ( x2 - xn ) / ( x2 - x1 );
            c2 = ( xn - x1 ) / ( x2 - x1 );

            % Lets fill the matrix with corresponding entries
            S( x == newX, smallerIndex ) = c1;
            S( x == newX, greaterIndex ) = c2;

        case 'repeat'
            % No interpolation at all, just use the value of the nearest
            % neighbour.
            [ ~, nnIndex ] = min( abs( x - originalX ) );
            S( x == newX, nnIndex ) = 1;
            
        case 'chebyshev'
            % How looks a Chebyshev interpolation?
            % We assume, the given Points result from sampling a function
            % at the roots of the next degree Chebyshev polynomial.
            degree = length( originalX ) - 1;
            
            C = zeros( degree + 1 );  
            for i = 0:1:degree
                if ( i == 0 )
                    C( :, i+1 ) = 1;
                elseif ( i == 1 )
                    C( :, i+1 ) = originalX;
                else
                    C( :, i+1 ) = 2 * originalX .* C( :, i ) - C( :, i-1 );
                end;
            end;
            P = ( 2 / ( degree + 1 )) * C;
            P( :, 1 ) = 0.5 * P( :, 1 );
            
            % Compute the coefficients of the Chebyshev series
%             c = P * originalY;
%             c = c';
            
            % Compute the matrix for the Chebyshev polynomials at the new
            % x values
            D = zeros( 1, degree + 1  );  
            for i = 0:1:degree
                if ( i == 0 )
                    D( i+1 ) = 1;
                elseif ( i == 1 )
                    D( i+1 ) = x;
                else
                    D( i+1 ) = 2 * x .* D( i ) - D( i-1 );
                end;
            end;
            
            D = ( 2 / ( degree + 1 )) * D;
            D( :, 1 ) = 0.5 * D( :, 1 );
            S( x == newX, : ) = D / P;
            
            
        otherwise
            error( [ 'Interpolation mode "' mode '" is not supported. ' ...
                'This part of the code should not reached, what means ' ...
                'something went terribly wrong, or somebody really ' ...
                'does not know, what he is actually doing with this ' ...
                'code.' ] );
    end;
end;

% Just ensure that the vector is a column vector in order to ensure, that
% the latter multiplication is valid.
if( ~iscolumn( originalY ) )
    originalY = originalY';
end;

%% Finally calculate the new values
newY = S * originalY;







end

