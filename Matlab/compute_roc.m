function [tpr,fpr,auroc,thresh] = compute_roc(binlbl,score,nthresh)

     % Modification April 2016: 
     
     % threshold-based computation of the ROC
     % scores are rescaled to fall into the range 0...1 
     % and then compared with nthresh equi-distant thresholds
     % note that nthresh must be large enough to guarantee correct
     % ROC computation 
     % planned improvement: 
     %  - re-scaling according to observed range of scores
     %  - efficient rank-based computation of ROC, mapped to thresholds 
     %    in order to facilitate threshold-averages
   
     % input
     % binllbl:  binarized labels 0,1 for two classes 
     %           (user-defined selection in multi-class problems)
     % score  :  continuous valued score, e.g. glvq-socre 0....1 
     % nthresh:  number of threshold values, default: 2000 
    
     if(nargin<3 || isempty(nthresh));  nthresh=5000; end;
    
     % heuristic rescaling of the scores to range 0....1 
     % to be done: re-scale according to observed range of values
     score=1./(1+exp(score/2)); 
     
     target =binlbl';   % should be 0,1
     output =score;    
     length(binlbl);
     
    tu=unique(target); % define binary target values
    t1=tu(1);          % target value t1 representing "negative" class
    t2=tu(2);          % target value t2 representing "positive" class

    npos=sum(target==t2);   % number of positive samples
    nneg=sum(target==t1);   % number of negative samples

    % for proper "threshold-averaged" ROC (see paper by Fawcett)
    % we use "nthresh" equi-distant thresholds between 0 and 1
    
    nthresh=5000; % default
    if (length(binlbl)>1250); nthresh=4*length(binlbl); end;
    
    nthresh = 2*floor(nthresh/2); % make sure nthresh is even
    thresh= linspace(0,1,nthresh+1); % odd number is actually used
                                     % so value 0.5 is in the list
    
                                                                    
    fpr=zeros(1,nthresh+1);   % false positive rates
    tpr=fpr;                  % true positive rates
    tpr(1)=1; fpr(1)=1;       % only positives, so tpr=fpr=1
    
    for i=1:nthresh-1;
        % count true positves, false positives etc.
        tp(i+1)= sum( target(score>thresh(i+1))      == t2 ); 
        fp(i+1)= sum( target(score>thresh(i+1))      == t1 );
        fn(i+1)= sum( target(score<=thresh(i+1))     == t2 );
        tn(i+1)= sum( target(score<=thresh(i+1))     == t1 ); 
        % compute corresponding rates
        tpr(i+1)= tp(i+1)/(tp(i+1)+fn(i+1));
        fpr(i+1)= fp(i+1)/(tn(i+1)+fp(i+1)); 
    end;
    
    % simple numerical integration (trapezoidal rule) 
    auroc= -trapz(fpr,tpr);  % minus sign due to order of values
     
    
     % Remark:    
     % an alternative could be the more sophisticated built-in  roc 
     % provided in the  Neural Network and/or Statistics toolboxe
     % which does not work with a fixed list of thresholds
     % but requires interpolation techniques for threshold-average
     % or the determination of the NPC performance. Syntax:      
     % [tpr,fpr,thresholds] = roc(target,output); 
     % only available with appropriate toolbox
     
end

