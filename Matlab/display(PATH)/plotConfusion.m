function plotConfusion( confMats, plbl, normalized, l , h, skip3 )
    nrClasses = size(confMats,1);
    if normalized
        for k=1:size(confMats,3)
            for i=1:nrClasses
                geheel = sum(confMats(i,:,k));
                for j=1:nrClasses
                    confMats(i,j,k) = (confMats(i,j,k)/geheel)*100; 
                end
            end 
            
        end
    end
    
    figure
    heatmap(confMats(:,:,1), plbl, plbl, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'FontSize',14);
    title('Orig space')
    figure
    heatmap(confMats(:,:,2), plbl, plbl, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'FontSize',14);
    title('All coefs')
    numberOfPlots = (h-l)/5 + 1;
    j=0;
    m=l/5+3;
    while j ~= numberOfPlots
        figure
        heatmap(confMats(:,:,m), plbl, plbl, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'FontSize',14);
        title(['nrCoefs:', num2str(l+j*5)]);
        m=m+2;
        if skip3
            m=m+1;
        end
        k=k+1;
        j=j+1;
    end
end

