
% perform repeated training for randomly splits of the data set
% for training and validation

function [gmlvq_mean, roc_validation, lcurves_mean,...
          lcurves_sdt, param_set, permus] = ...
  run_validation(fvec,lbl,totalsteps,nruns,prctg,plbl,permus) 
            
% general algorithm settings and parameters of the Papari procedure
            % general algorithm settings and parameters of the Papari procedure
[showplots,doztr,mode,rndinit, etam, etap, mu, decfac, incfac, ncop] =...
                                        set_parameters(fvec); 
% input and parameters
% fvec      : set of all feature vectors 
% lbl       : class labels of data 
% totalsteps: number of batch gradient steps per training process
% nruns     : number of validation runs (splits of data)
% prctg     : percentage of data reserved for validation in each run
% plbl      : prototype label configuration
% showplots : if ... =1, learning curves etc. are shown
% doztr     : if ... =1, z-score transform. is performed in each training
% mode 
  % 0 for matrix without regularization
  % 1 for matrix with null-space correction
  % 2 for diagonal updates only  
% rndinit
  % 0 for initialization of relevances as identity matrix 
  % 1 for randomized initialization of relevance matrix
 
% output
% structures defined below
  
%close all;
       
% reproducible random numbers
 rng('default'); 
 rngseed=4713;
 rng(rngseed); 

% set defaults if necessary

if (nargin<6||isempty(plbl));       % one prototype per class?
    plbl=[1:length(unique(lbl))];  
    display('default: one prototype per class'); 
end;
display('prototype configuration'); plbl

if (nargin<5||isempty(prctg)); prctg=10; end; % 10% of data for testing?
 
if (nargin<4||isempty(nruns)); nruns=5; end;  % 5 validation runs? 

usePerms = 1;
if (nargin < 7 || isempty(permus))
    usePerms = 0;
end
display(['learning curves, averages over ',num2str(nruns),...
                                                   ' validation runs']); 
display(['with ',num2str(prctg),' % of examples left out for testing']); 

nprots=length(plbl);                   % total number of prototypes
nclasses = length(unique(plbl));       % number of classes 
nfv=size(fvec,1);                      % number of feature vectors
ndim=size(fvec,2);                     % dimension of feature vectors
numtrain = floor(nfv*(100-prctg)/100); % size of individual training sets
numtest  = nfv-numtrain;               % size of individual test sets

% check for consistency and output error messages
% transpose lbl if necessary
[lbl]=check_arguments(plbl,lbl,fvec,ncop,totalsteps); 

% initialize all observed quantities
mcftra=zeros(totalsteps,1); mcfval=mcftra; 
mtetra=mcftra; mteval=mcftra; mauctra=mcftra; maucval=mcftra;
mcwtra= zeros(totalsteps,nclasses); mcwval=mcwtra; 
scftra=zeros(totalsteps,1); scfval=scftra;
stetra=scftra; steval=scftra; sauctra=scftra; saucval=scftra;
scwtra= zeros(totalsteps,nclasses); scwval=scwtra; 
confmat=zeros(nclasses,nclasses); 

lambda=zeros(nruns,ndim,ndim);             % relevance matrix
protos=zeros(nruns,length(plbl),ndim);     % prototypes

for krun=1:nruns;  % loop for validation runs
    confact=zeros(nclasses,nclasses); % actual confusion matrix in the run
    display(['validation run ',num2str(krun),' of ',num2str(nruns)])

    % shuffle data randomly and repeat if test or training set 
    % does not contain all classes
    % error message terminates program when 10 attempts exceeded 
    % to be improved in future versions 
    mix=1; attempts=1; 
    
    while(mix<nclasses && attempts<=10); 
       if usePerms ~= 1
        permus(krun,:) = randperm(nfv);    % random order of samples
       end
       permulbl = lbl(permus(krun,:));    % re-ordered labels
       permufvc = fvec(permus(krun,:),:); % re-order feature vectors
       
       fvectrain= permufvc(1:numtrain,:);   % training set feature vectors
       lbltrain = permulbl(1:numtrain);     % training set labels
       fvecout  = permufvc((numtrain+1):end,:);  % left out feature vectors
       lblout   = permulbl((numtrain+1):end);    % corresponding labels

       mix=min(length(unique(lblout)),length(unique(lbltrain))); 
       % test set and training set should contain all classes
       attempts=attempts+1; 
    end; 
    if (attempts > 10);
       msg =['test or training set did not contain all classes ', ...
             'after 10 attempts ', ...
              ' - increase or decrease test set size (prctg) accordingly']; 
       error(msg)
    end; 
    
% if doztr==1 do z-transformation of training set
% and rescale test set accordingly 
  mf=zeros(1,ndim); st=ones(1,ndim); % initialize feature means and st.dev. 
  if doztr==1;
      [fvectrain,mf,st] = do_zscore(fvectrain);  % transform training data
      for i =1:numtest
         fvecout(i,:)= (fvecout(i,:)-mf)./st; % transfrom validation set
                                              % accordingly 
      end; 
  end;
  
  % perform one run and get learning curve variables
   [w,omega,cftra,tetra,cwtra,auctra,cfval,teval,cwval,aucval]= ...
    do_lcurve(fvectrain,lbltrain,fvecout,lblout,...
                                          plbl,totalsteps);                                      
                                      
   protos(krun,:,:)=w; 
   lambda(krun,:,:) = omega'*omega; 
   
   % get final classification labels and score
   [~,crout,~,score]    = compute_costs(fvecout,lblout,w,plbl,omega,mu); 
   % compute ROC
   [tpr,fpr,~,thresh] = compute_roc(lblout>1,score);
   % [thresh,tpr,fpr,auroc]=eval_roc(score,lblout>1,nthresh);
   tprs(krun,:) =tpr; % true positive rates
   fprs(krun,:) =fpr; % false positive rates
   
   
   for ios = 1:length(lblout);  % loop through all classes
                                % computation of actual confusion matrix
       confact(lblout(ios),crout(ios))=confact(lblout(ios),crout(ios))+1;
   end; 
   % calculate averaged confusion matrix (percentages) over runs 
%    for icr = 1:size(confmat,1);
%        confmat(icr,:)=confmat(icr,:)+confact(icr,:)/sum(confact(icr,:))*100/nruns;
%    end;
   
   for icr = 1:size(confmat,1);
       confmat(icr,:)=confmat(icr,:)+confact(icr,:)/nruns;
   end;
  
   
   % compute averages and standard deviations of relevant quantities
   mcftra= mcftra + cftra/nruns;  mcfval= mcfval + cfval/nruns;
   mtetra= mtetra + tetra/nruns;  mteval= mteval + teval/nruns; 
   mauctra = mauctra + auctra/nruns; maucval= maucval + aucval/nruns;
   mcwtra = mcwtra + cwtra/nruns; mcwval= mcwval + cwval/nruns; 
   scftra= scftra + cftra.^2/nruns;  scfval= scfval + cfval.^2/nruns;
   stetra= stetra + tetra.^2/nruns;  steval= steval + teval.^2/nruns; 
   sauctra = sauctra + auctra.^2/nruns; saucval= saucval + aucval.^2/nruns;
   scwtra = scwtra + cwtra.^2/nruns; scwval= scwval + cwval.^2/nruns;
  
end; 

%  sort and average prototypes properly
wm=squeeze(protos(1,:,:)); wm2=wm.^2;
for ipt=2:nruns;
   protlist=1:nprots;  % list of prototypes not yet used for update of mean
  for i=1:nprots;         % loop through the mean prototypes  
      dij=NaN(1,nprots);  % preset distances to NaN
      for j=protlist;     % loop only through unused actual prototypes
          if (plbl(j)==plbl(i))  % only consider same class prototypes
                                 % all other distances remain NaN
               dij(j) = norm((wm(i,:)'-squeeze(protos(ipt,j,:))),2); 
          end
      end;
      [~,jmin] =  min(dij); % determine smallest distance prototype 
                            % in the list 
      % update mean and mean square
      wm (i,:) =   wm(i,:) +  (squeeze(protos(ipt,jmin,:))')   /nruns;  
      wm2(i,:) =  wm2(i,:) +  (squeeze(protos(ipt,jmin,:))').^2/nruns; 
      protlist= setdiff(protlist,jmin); % remove jmin from list of unused
  end;
end;

 % updated mean prototypes, matrix, cost function values, training
 % errors, auc and class-wise errors and corresponding standard deviations
   protos_mean = wm; 
   protos_std  = sqrt(wm2 - wm.^2);
   lambda_mean = squeeze(mean(lambda,1)); 
   lambda_std  = sqrt(squeeze(mean(lambda.^2,1))-lambda_mean.^2); 
   scftra= sqrt(scftra-mcftra.^2);   scfval= sqrt(scfval-mcfval.^2);
   stetra= sqrt(stetra-mtetra.^2);   steval= sqrt(steval-mteval.^2); 
   sauctra=sqrt(sauctra-mauctra.^2); saucval=sqrt(saucval-maucval.^2);
   scwtra =sqrt(scwtra- mcwtra.^2);  scwval= sqrt(scwval - mcwval.^2);

 % define structures for output
   % mean learning curves
   lcurves_mean = ... 
               struct('mcftra',mcftra,'mcfval',mcfval,...
                      'mtetra',mtetra,'mteval',mteval,... 
                      'mauctra',mauctra,'maucval',maucval,...
                      'mcwtra',mcwtra,'mcwval',mcwval); 
                  
   % standard deviations of learning curves                
   lcurves_sdt = ... 
               struct('scftra',scftra,'scfval',scfval,...
                      'stetra',mtetra,'steval',steval,... 
                      'sauctra',sauctra,'saucval',saucval,...
                      'scwtra',scwtra,'scwval',scwval);                  
            
   roc_validation = ...
               struct('tprs', tprs, 'fprs', fprs, 'thresh', thresh, ...
                      'meantpr', mean(tprs,1), 'meanfpr', mean(fprs,1),...
                      'confmat',confmat); 
      
   gmlvq_mean = ... 
           struct('protos_mean', protos_mean, 'protos_std', protos_std,...
                  'lambda_mean', lambda_mean, 'lambda_std', lambda_std,...
                  'plbl',plbl); 
  
   param_set = struct('totalsteps',totalsteps,'doztr',...
           doztr,'mode',mode,'rndinit',rndinit,'etam',etam,'etap',etap,...
          'mu',mu,'decfac',decfac,'infac',incfac,'ncop',ncop,...
          'rngseed',rngseed);                 
  
              
if(showplots==1)   % display results
     

  figure(1); 
  onlyat= floor(linspace(1,totalsteps,10));  % set "time" points for output
  onlyatval= onlyat(1:end-1)+1;
     
  figure(1)  
  msize=15; % size of symbols
  if (totalsteps< 50); msize=20; end;
  
  subplot(3,2,1);                 % total training and validation errors
  % axis([0 totalsteps 0 1.2*max(mteval)]); 
   plot(1:totalsteps,mtetra,':.',1:totalsteps,mteval,':.',...
                                              'MarkerSize',msize); 
   axis tight; axis 'auto y'; 
   hold on; box on;
   title('total error rates',...
       'FontName','LucidaSans', 'FontWeight','bold'); 
   legend('training','test','Location','Best'); 
   xlabel('gradient steps');
   errorbar(onlyat,mtetra(onlyat),stetra(onlyat)/sqrt(nruns),...
           'co','MarkerSize',1); 
   errorbar(onlyatval,mteval(onlyatval),steval(onlyatval)/sqrt(nruns),...
           'go','MarkerSize',1);  
   hold off;
  
   
  subplot(3,2,2);                 % AUC(ROC) for training and validation 
   plot(1:totalsteps,mauctra,':.',1:totalsteps,maucval,':.',...
                                                 'MarkerSize',msize); 
   axis([1 totalsteps min(0.7,min(maucval)) 1.05]); % axis 'auto y';
   hold on; box on;
   % plot(1:totalsteps,maucval,'g.'); 
   legend('training','test','Location','Best'); 
   title('AUC(ROC) w.r.t. to class 1 vs. all others',...
         'FontName','LucidaSans', 'FontWeight','bold'); 
   errorbar(onlyat,mauctra(onlyat),sauctra(onlyat)/sqrt(nruns),'b.'); 
   errorbar(onlyatval,maucval(onlyatval),saucval(onlyatval)/sqrt(nruns),'g.'); 
   hold off;
   
   subplot(3,2,3);                % class-wise training errors 
   plot(1:totalsteps,mcwtra,':.','MarkerSize',msize); 
   title('class-wise training errors',...
   'FontName','LucidaSans', 'FontWeight','bold');
   xlabel('gradient steps');
   legend(num2str([1:nclasses]'),'Location','Best');
   axis tight; axis 'auto y'; 
   hold on; box on;
   hold off;
   
   subplot(3,2,4);                % class-wise test errors 
   plot(1:totalsteps,mcwval,':.','MarkerSize',msize); 
   title('class-wise test errors',...
       'FontName','LucidaSans', 'FontWeight','bold'); 
   xlabel('gradient steps')
   legend(num2str([1:nclasses]'),'Location','Best');
   axis tight; axis 'auto y'; 
   hold on; box on;
   % plot(1:totalsteps,mcwval,'.'); 
   hold off;
   
   subplot(3,2,5);                % cost-function (training set)
   plot(1:totalsteps,mcftra,':.','MarkerSize',msize); 
   title('cost fct. per example w/o penalty',...
       'FontName','LucidaSans', 'FontWeight','bold'); 
   xlabel('gradient steps')
   axis tight; axis 'auto y'; 
   hold on; box on;
   % plot(1:totalsteps,mcftra,'.'); 
   hold off;
   
   subplot(3,2,6);                % cost-function (test)
   plot(1:totalsteps,mcfval,':.','MarkerSize',msize); 
   title('analagous for validation set',...
       'FontName','LucidaSans', 'FontWeight','bold'); 
   xlabel('gradient steps') 
   axis tight; axis 'auto y'; 
   hold on; box on;
   % plot(1:totalsteps,mcfval,'.'); 
   hold off;
   
   
%  threshold-averaged validation roc of final gmlvq systems after 
%  totalsteps gradient steps
   figure(2); 
   plot(mean(fprs,1),mean(tprs,1),'b-','LineWidth',2);
   hold on;
   plot(mean(fprs(:,thresh==0.5)),...
        mean(tprs(:,thresh==0.5)),'ko',...
        'MarkerSize',10,'MarkerFaceColor','g');
   legend(['AUC= ',num2str(-trapz(mean(fprs,1),mean(tprs,1)))],...
           'NPC performance','Location','SouthEast');
   plot([0 1],[0 1],'k:'); 
   xlabel('false positive rate');
   ylabel('true positive rate'); 
   axis square; 
   title('threshold-avg. test set ROC (class 1 vs. all others)',...
         'FontName','LucidaSans', 'FontWeight','bold'); 
   hold off;

   figure(3);    % visualize the GMLVQ system 
   display_gmlvq(protos_mean,lambda_mean,plbl,ndim);  
   
end; 
 
   

