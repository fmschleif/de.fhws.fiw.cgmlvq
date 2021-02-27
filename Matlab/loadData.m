function [st] = loadData( name )
%LOADDATA Summary of this function goes here
%   Detailed explanation goes here

    trainSet = strcat('UCR_TS_Archive_2015/', name,'/', name,'_TRAIN');
    testSet = strcat('UCR_TS_Archive_2015/', name,'/', name,'_TEST');
    TRAIN = load(trainSet); TEST = load(testSet); 

    st.lbltra = TRAIN(:,1); st.lblval = TEST(:,1);
    plbl = unique(st.lbltra)';
    incr = 1-plbl(1);
    st.lbltra = st.lbltra + incr; st.lblval = st.lblval + incr;
    
    st.tra = TRAIN(:,2:end);
    st.val = TEST(:,2:end);
    st.fvec = [st.tra; st.val];
    st.lbl = [st.lbltra; st.lblval];

%     if showTraining
%         %plotDataset(st.orig.tra,st.lbltra, nrExamp);
%         plotDataset(st.tra,st.lbltra,'northeast','Training set',1,classlim,nrExamp);
%     end
end

