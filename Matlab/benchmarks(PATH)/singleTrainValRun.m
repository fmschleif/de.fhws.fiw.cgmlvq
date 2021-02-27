function [ confmats, aucval ] = singleTrainValRun(tra,val,lbltra,lblval, epochs, plbl)
    [traz,valz,mf,st]=do_zscore_trainAndVal(tra,val);
    [~,~,~,~,~,~,~,~,~,aucval,confmats]=do_lcurve(traz,lbltra,valz,lblval,plbl,epochs);
    
    %display aucval of the last epoch (this is aucval on the validation
    %set)
    %disp(aucval(epochs));
    %aucval = aucval(epochs);
    %heatmap(confmats, plbl, plbl, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'FontSize',14);
end

