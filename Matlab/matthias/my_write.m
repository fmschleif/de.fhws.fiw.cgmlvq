function my_write( file, matrix )

    dlmwrite( file, matrix, 'delimiter', ',', 'precision', 16 );

end