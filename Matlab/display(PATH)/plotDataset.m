function plotDataset( fvec, lbl, loc, ti, set, classLim, nrExamp )
    L = size(fvec,2);
    classlbl = unique(lbl);
    if classLim ~= 0
       classlbl = classlbl(1:classLim); 
    end
    nrClass = length(classlbl);
    figure
    %first plot the first example of each class so that we can construct a
    %legend
    for c=1:nrClass
       examplesBelongtoC = fvec(lbl==c,:);
       plot(1:L, examplesBelongtoC(1,:), 'Color', colordg(c));
       hold on
    end

    %convert labels to string cells
    classStrings = cell(nrClass,1);
    for i=1:nrClass
       classStrings{i} = sprintf('%i', classlbl(i)); 
    end
    
    %create a legend
    legend(classStrings, 'Location', loc);
    
    for c=1:nrClass
        examplesBelongtoC = fvec(lbl==c,:);
        if size(examplesBelongtoC,1) > nrExamp
            examplesBelongtoC = examplesBelongtoC(1:nrExamp,:);
        end
        plot(1:L, examplesBelongtoC,'Color',colordg(c));
        hold on
    end
    
    title(ti)
    if set
        xlabel('Feature index')
        ylabel('Feature value')
    else
        xlabel('Frequency bin')
        ylabel('Magnitude')
    end
end

