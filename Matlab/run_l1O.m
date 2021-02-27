
% perform repeated training according to a leave-one-out validation scheme

function [gmlvq_mean, roc_l1O, lcurves_mean,lcurves_sdt, param_set ] = ...
  run_l1O(fvec,lbl,totalsteps,plbl) 
              
close all;

% reproducible random numbers
 rng('default'); 
 rngseed=291065;
 rng(rngseed); 

% set defaults if necessary

if (nargin<4||isempty(plbl)); % one prototype per class
    plbl=[1:length(unique(lbl))];  
    display('default: 1 prototype per class'); 
end;
display('prototype configuration'); 
plbl

% GMLVQ system and training parameters

[showplots,doztr,mode,rndinit,etam,etap,mu,decfac,incfac,ncop] =...
                                        set_parameters(fvec);
% input arguments and parameters   
% fvec, lbl              training data, feature vectors and labels
% totalsteps: number of batch gradient steps to be performed
% plbl: labels assigned to prototypes, also specifies number per class
% e.g. plbl=[1,1,2,2,3] for 5 prototypes with labels 1,2,3  
% showplots: display learning curves? 
% doztr: do z-transformation based on training set? 
% if doztr=0, data is only centered
% mode 
  % 0 for matrix without regularization
  % 1 for matrix with null-space correction
  % 2 for diagonal matrix (GRLVQ)
  % 3 for GLVQ with Euclidean distance (equivalent)
% rndinit
  % 0 for initialization of relevances as identity matrix 
  % 1 for randomized initialization of relevance matrix

display(['learning curves, averages over ',num2str(size(fvec,1)),...
                                                   ' l1O runs']); 

nprots=length(plbl);                   % total number of prototypes
nclasses = length(unique(plbl));       % number of classes 
nfv=size(fvec,1);                      % number of feature vectors
ndim=size(fvec,2);                     % dimension of feature vectors
numtrain = nfv-1;      % size of individual training sets
numtest  = 1;          % size of individual test sets

% initialize random number generator 
rng(79061);   

% check for consistency and output error messages
% transpose lbl if necessary
[lbl]=check_arguments(plbl,lbl,fvec,ncop,totalsteps); 

% initialize quantities of interest
mcftra=zeros(totalsteps,1); 
mtetra=mcftra; mauctra=mcftra; 
mcwtra= zeros(totalsteps,nclasses); 
scftra=zeros(totalsteps,1); 
stetra=scftra;  sauctra=scftra; 
scwtra= zeros(totalsteps,nclasses); 
scoreval = zeros(1,nfv); % score of leave-one-out example
nruns = nfv;  % number of l1O runs equal number of samples
nthresh=2000; % number of threshold values for roc evaluation
lambda=zeros(nruns,ndim,ndim); 
protos=zeros(nruns,length(plbl),ndim); 
confmat =zeros(nclasses,nclasses);

for krun=1:nruns;  % leave one out runs
    display(['leave one out: ',num2str(krun),' of ',num2str(nruns)])

    trainsetind = setdiff(1:nfv,krun); % leave only sample krun out
    fvectrain   = fvec(trainsetind,:);
    lbltrain    = lbl(trainsetind); 
    fvecout     = fvec(krun,:);
    lblout      = lbl(krun); 
    
% if doztr==1 do z-transformation of training set
% and rescale test set accordingly 
  mf=zeros(1,ndim); st=ones(1,ndim); % initialize feature means and st.dev.
  if doztr==1;
      [fvectrain,mf,st] = do_zscore(fvectrain);
      for i =1:numtest
           fvecout(i,:)= (fvecout(i,:)-mf)./st; 
      end; 
  end; 
 
 % perform leave-one-out run and compute learning curve quantities 
 [w,omega,cftra,tetra,cwtra,auctra]= ...
      do_lcurvel1O(fvectrain,lbltrain,fvecout,lblout,plbl,...
                                            mode,rndinit,totalsteps);      
 % store prototypes and matrix of run  
   protos(krun,:,:)=w; 
   lambda(krun,:,:) = omega'*omega; 
   
 % compute costs, scores and confusion matrix in the run  
   [~,crout,~,score]    =  compute_costs(fvecout,lblout,w,plbl,omega,mu); 
   scoreval(krun)=score;           % score of validation example
   confmat(lblout,crout) = confmat(lblout,crout)+1; 

   % average of monitored quantities
   mcftra   = mcftra + cftra/nruns;  
   mtetra   = mtetra + tetra/nruns;   
   mauctra  = mauctra + auctra/nruns; 
   mcwtra   = mcwtra + cwtra/nruns; 
   scftra   = scftra + cftra.^2/nruns;  
   stetra   = stetra + tetra.^2/nruns;  
   sauctra  = sauctra + auctra.^2/nruns; 
   scwtra   = scwtra + cwtra.^2/nruns; 
  
end; % leave one out runs performed

for icr = 1:size(confmat,1);
       confmat(icr,:)=confmat(icr,:)/sum(confmat(icr,:))*100;
end;



% sort and average prototypes properly
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

   protos_mean = wm; 
   protos_std  = sqrt(wm2 - wm.^2);
   lambda_mean = squeeze(mean(lambda,1)); 
   lambda_std  = sqrt(squeeze(mean(lambda.^2,1))-lambda_mean.^2); 
    
   scftra= sqrt(scftra-mcftra.^2);   
   stetra= sqrt(stetra-mtetra.^2);    
   sauctra=sqrt(sauctra-mauctra.^2); 
   scwtra =sqrt(scwtra- mcwtra.^2);  

   
[tprs,fprs,auroc,thresh] = compute_roc(lbl>1,scoreval);    

% nthresh=2000;
% [thresh,tprs,fprs,auroc]=eval_roc(scoreval,lbl>1,nthresh);
   
   % mean learning curves
   lcurves_mean = ... 
               struct('mcftra',mcftra,...
                      'mtetra',mtetra,...
                      'mauctra',mauctra,...
                      'mcwtra',mcwtra); 
                  
   % standard deviations of learning curves                
   lcurves_sdt = ... 
               struct('scftra',scftra,...
                      'stetra',mtetra,... 
                      'sauctra',sauctra,...
                      'scwtra',scwtra);                  
            
   roc_l1O = ...
                struct('tprs', tprs, 'fprs', fprs, 'thresh', thresh, ...
                       'auroc',auroc,'confmat',confmat); 
        
   gmlvq_mean = ... 
           struct('protos_mean', protos_mean, 'protos_std', protos_std,...
                  'lambda_mean', lambda_mean, 'lambda_std', lambda_std,...
                  'plbl',plbl); 
              
   param_set = struct('totalsteps',totalsteps,'doztr',...
           doztr,'mode',mode,'rndinit',rndinit,'etam',etam,'etap',etap,...
          'mu',mu,'decfac',decfac,'infac',incfac,'ncop',ncop,...
          'rngseed',rngseed);       
              
              
    
if(showplots==1);    % display results
    
  onlyat= floor(linspace(1,totalsteps,10));  % "time" points for plotting
  figure(1)
  msize=15;   % symbol size
  if (totalsteps< 50); msize=20; end;
   
  subplot(2,2,1);   % total training error rate 
  % axis([0 totalsteps 0 1.2*max(mteval)]); 
   plot(1:totalsteps,mtetra,':.','MarkerSize',msize); 
   axis tight; axis 'auto y'; 
   hold on; box on;
   title('total training error rate',...
       'FontName','LucidaSans', 'FontWeight','bold'); 
   legend('training','Location','Best'); 
   xlabel('gradient steps');
   errorbar(onlyat,mtetra(onlyat),stetra(onlyat)/sqrt(nruns),...
           'co','MarkerSize',1); 
   hold off;
   
  subplot(2,2,2);  % total training AUC(ROC)
   plot(1:totalsteps,mauctra,':.','MarkerSize',msize); 
   axis([1 totalsteps 0.8 1.05]); % axis 'auto y';
   hold on; box on;
   % plot(1:totalsteps,maucval,'g.'); 
   legend('training','Location','Best'); 
   title('AUC(ROC) w.r.t. to class 1 vs. all others',...
         'FontName','LucidaSans', 'FontWeight','bold'); 
   errorbar(onlyat,mauctra(onlyat),sauctra(onlyat)/sqrt(nruns),'b.'); 
   hold off;
   
   subplot(2,2,3);   % class-wise training errors
   plot(1:totalsteps,mcwtra,':.','MarkerSize',msize); 
   title('class-wise training errors',...
   'FontName','LucidaSans', 'FontWeight','bold');
   xlabel('gradient steps');
   legend(num2str([1:nclasses]'),'Location','Best');
   axis tight; axis 'auto y'; 
   hold on; box on;
   hold off;
   
   subplot(2,2,4);   % cost function (training)
   plot(1:totalsteps,mcftra,':.','MarkerSize',msize); 
   title('cost fct. w/o penalty term (training)',...
   'FontName','LucidaSans', 'FontWeight','bold');
   xlabel('gradient steps');
   axis tight; axis 'auto y'; 
   hold on; box on;
   hold off;
   
%  single l1O roc of final gmlvq systems after 
%  totalsteps gradient steps

   figure(2); 
   plot(mean(fprs,1),mean(tprs,1),'b-','LineWidth',2);
   hold on;
   plot((fprs(:,thresh==0.5)),...
        (tprs(:,thresh==0.5)),'ko',...
        'MarkerSize',10,'MarkerFaceColor','g');
   legend(['AUC= ',num2str(-trapz((fprs),(tprs)))],...
           'NPC performance','Location','SouthEast');
   plot([0 1],[0 1],'k:'); 
   xlabel('false positive rate');
   ylabel('true positive rate'); 
   axis square; 
   title('Leave-One-Out ROC (class 1 vs. all others)',...
         'FontName','LucidaSans', 'FontWeight','bold'); 
   hold off;

   figure(3); 
   display_gmlvq(protos_mean,lambda_mean,plbl,ndim);  
   
end; 
 
   

