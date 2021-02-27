function plotConfmats( benchout, rMax, str, plbl )
%Plot and save confmats, expects a benchout struct which is the output of
%the benchmark functions.
    figure 
    heatmap(benchout.orig.confmat, plbl, plbl, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'FontSize',14);
    title('Orig. space')
    print(strcat(str,'\orig'), '-dpng')
    
    figure
    heatmap(benchout.Fourier.confmat, plbl, plbl, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'FontSize',14);
    title('Full Fourier space')
    print(strcat(str,'\Fourier'), '-dpng')
    
    figure
    heatmap(benchout.FourierConc.confmat, plbl, plbl, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'FontSize',14);
    title('Full Fourier concat')
    print(strcat(str,'\FourierConc'), '-dpng')
    j=1;
    for r=5:5:rMax
        savestrFour = strcat(str,'\truncFour\');
        savestrFourConc = strcat(str,'\truncFourConc\');
        savestrFourBack = strcat(str,'\truncBack\');
        
        figure
        heatmap(benchout.truncFour.confmat(:,:,j), plbl, plbl, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'FontSize',14);
        title(strcat('nrCoefs: ', num2str(r)))
        print(strcat(savestrFour,'nrCoefs',num2str(r)), '-dpng')
        
        figure
        heatmap(benchout.truncFourConc.confmat(:,:,j), plbl, plbl, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'FontSize',14);
        title(strcat('nrCoefs: ', num2str(r)))
        print(strcat(savestrFourConc,'nrCoefs',num2str(r)), '-dpng')
        
        figure
        heatmap(benchout.truncBack.confmat(:,:,j), plbl, plbl, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'FontSize',14);
        title(strcat('nrCoefs: ', num2str(r)))
        print(strcat(savestrFourBack,'nrCoefs',num2str(r)), '-dpng')
        j=j+1;
    end
end


