function plotAUCs( benchout, rMax, showMisses ,str )
    %depending or not on whether a plot flag is set, an AUC curve along the
    %lines of Melchert's papers. expects a benchout struct which is the output of
	%the benchmark functions.
    figure
    plot(1:rMax, ones(1, rMax) * benchout.orig.aucval, 'k', 'LineWidth', 3) %plot auc of orig result as black line
    hold on
    plot(1:rMax, ones(1, rMax) * benchout.Fourier.aucval, 'b', 'LineWidth', 3) %plot full Fourier auc as a blue line
    hold on
    plot(1:rMax, ones(1, rMax) * benchout.FourierConc.aucval, 'g', 'LineWidth', 3) %plot concat full Fourier auc as a red line
    hold on
    j=1;
    for r=5:5:rMax
        plot(r, benchout.truncFour.aucval(j), 'o', 'MarkerFaceColor', 'b', 'MarkerSize', 9)
        hold on 
        plot(r, benchout.truncFourConc.aucval(j), 's', 'MarkerFaceColor', 'g', 'MarkerSize', 9)
        hold on
        plot(r, benchout.truncBack.aucval(j),  'd', 'MarkerFaceColor', 'r', 'MarkerSize', 9)
        hold on;
        j=j+1;
    end
    
    xlabel('Number of coefficients')
    ylabel('AUC of validation set(s)')
    title('Performance of the strategies')
    legend('Original','Wirtinger', 'Concat')
    savestr = strcat(str,'\AUCs');
    savefig(savestr)
    print(savestr, '-dpng')
    %display nr of misses
    if showMisses
        figure
        [~,misses]=normalConfmat(benchout.orig.confmat);
        plot(1:rMax, ones(1, rMax) * misses, 'k', 'LineWidth', 3) %plot misses of orig result as black line
        hold on
        
        [~,misses]=normalConfmat(benchout.Fourier.confmat);
        plot(1:rMax, ones(1, rMax) * misses, 'b', 'LineWidth', 3) %plot full Fourier misses as a blue line
        hold on
        
        [~,misses]=normalConfmat(benchout.FourierConc.confmat);
        plot(1:rMax, ones(1, rMax) * misses, 'g', 'LineWidth', 3) %plot concat full Fourier misses as a green line
        hold on
        
        j=1;
        for r=5:5:rMax
            [~,misses]=normalConfmat(benchout.truncFour.confmat(:,:,j));
            plot(r, misses, 'o', 'MarkerFaceColor', 'b', 'MarkerSize', 9)
            hold on 
            [~,misses]=normalConfmat(benchout.truncFourConc.confmat(:,:,j));
            plot(r, misses, 's', 'MarkerFaceColor', 'g', 'MarkerSize', 9)
            hold on
            [~,misses]=normalConfmat(benchout.truncBack.confmat(:,:,j));
            plot(r, misses,  'd', 'MarkerFaceColor', 'r', 'MarkerSize', 9)
            hold on;
            j=j+1;
        end

        xlabel('Number of coefficients')
        ylabel('Number of misclasfficiations on validation set')
        title('Misclassifications')
        legend('Original','Wirtinger', 'Concat')
        savestr = strcat(str, '\misses');
        savefig(savestr)
        print(savestr, '-dpng')
    end
end

