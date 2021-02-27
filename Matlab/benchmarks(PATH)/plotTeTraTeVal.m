function plotTeTraTeVal( benchout, str, epochs )
    %Plots the training and validation error per epoch. Expects a benchout
    %struct which is the output of the benchmark functions.
    figure
    plot(1:epochs,benchout.orig.tetra,'k');
    hold on
    plot(1:epochs,benchout.Fourier.tetra,'b');
    hold on
    plot(1:epochs,benchout.FourierConc.tetra,'g');
    hold on
    xlabel('epoch')
    ylabel('error')
    title('Error/epoch training set')
    savestr = strcat(str, '\tetra');
    savefig(savestr)
    print(savestr, '-dpng')
    
    %test error curves
    figure
    plot(1:epochs,benchout.orig.teval,'k');
    hold on
    plot(1:epochs,benchout.Fourier.teval,'b');
    hold on
    plot(1:epochs,benchout.FourierConc.teval,'g');
    xlabel('epoch')
    ylabel('error')
    title('Error/epoch validation set')
    savestr = strcat(str, '\teval');
    savefig(savestr)
    print(savestr, '-dpng')
end

