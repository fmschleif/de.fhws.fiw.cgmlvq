function [ aucval, confmat, crisp, tetra, teval ] = evaluateStrat(st, epochs, plbl, show )
    crisp =0;
    [traz,valz]=do_zscore_trainAndVal(st.tra,st.val);
    [~,~,~,tetra,~,~,~,teval,~,aucval,confmat]=do_lcurve(traz,st.lbltra,valz,st.lblval,plbl,epochs);
    aucval = aucval(epochs);
    

    if show
        figure
        heatmap(confmat, plbl, plbl, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'FontSize',14);
    end
end

