
% perform a single GMLVQ training process
% optional visualization of training curves and gmlvq system

function [gmlvq_system, training_curves, param_set]= ...
   run_single(fvec,lbl,totalsteps,plbl)    
 
% gmlvq with global matrix only, square Omega, potentially diagonal
% batch gradient descent with step size control
% following a modified Papari procedure (unpublished) 
% perform training based using fvec, lbl
% evaluate performance measures after each step
% with respect to training set and validation set

% input arguments    
% fvec, lbl              training data, feature vectors and labels
% plbl: labels assigned to prototypes, also specifies number per class
% e.g. plbl=[1,1,2,2,3] for 5 prototypes with labels 1,2,3  
% totalsteps: number of batch gradient steps to be performed


if (nargin<4||isempty(plbl)); plbl=[1:length(unique(lbl))]; 
display('default: one prototype per class'); 
end;
display('prototype configuration'); plbl

if (nargin<3||isempty(totalsteps)); totalsteps=10; 
display('default number of training steps'); end;
% default: 10 gradient steps only 
display('number of training steps'); totalsteps

% general algorithm settings and parameters of the Papari procedure
[showplots,doztr,mode,rndinit, etam0, etap0, mu, decfac, incfac, ncop] =...
                                        set_parameters(fvec); 
etam=etam0;  % initial step size matrix
etap=etap0;  % intitial step size prototypes
               
                                    
% showplots (0 or 1): plot learning curves etc? 
% doztr (0 or 1): perform z-score transformation based on training set
% mode 
  % 0 for matrix without null-space correction
  % 1 for matrix with null-space correction
  % 2 for diagonal matrix (GRLVQ)                    DISCOURAGED
  % 3 for GLVQ with Euclidean distance (equivalent)
% rndinit
  % 0 for initialization of relevances as identity matrix 
  % 1 for randomized initialization of relevance matrix 
% etam:    step size of matrix updates
% etap:    step size of prototype updates
% mu  :    control parameter of penalty term for singular Lambda
% decfac:  factor for decrease of step sizes for oscillatory behavior
% incfac:  factor for increase of step sizes for smooth behavior
% ncop:    number of copies in Papari procedure

% check for consistency and output error messages
% transpose lbl if necessary

[lbl]=check_arguments(plbl,lbl,fvec,ncop,totalsteps); 

close all;   % close all figures

% reproducible random numbers
 rng('default'); 
 rngseed=291024;
 rng(rngseed); 


nfv = size(fvec,1);          % number of feature vectors in training set
ndim = size(fvec,2);         % dimension of feature vectors
ncls = length(unique(lbl));  % number of classes 
nprots = length(plbl);       % total number of prototypes

te=zeros(totalsteps+1,1);      % define total error
cf=te; auc=te;               % define cost function and AUC(ROC)
cw=zeros(totalsteps+1,ncls);   % define class-wise errors
stepsizem= te;               % stepsize matrix in the course of training
stepsizep= te;               % stepsize prototypes in the course ...

      
      mf=zeros(1,ndim);      % initialize feature means
      st=ones(1,ndim);       % and standard deviations
if doztr==1;
      [fvec,mf,st] = do_zscore(fvec);  % perform z-score transformation
else  [~,mf,st]= do_zscore(fvec);      % evaluate but don't apply 
end; 

% initialize prototypes and omega 
  [proti,omi] =  set_initial(fvec,lbl,plbl,mode,rndinit);
  prot=proti;  om =omi;   % initial values   
 
  
% copies of prototypes and omegas stored in protcop and omcop
% for the adaptive step size procedure 
  protcop = zeros(ncop,size(prot,1),size(prot,2));
  omcop   = zeros(ncop,size(om,1) , size(om,2) );

  % calculate initial values for learning curves
  [costf,~,marg,score] = compute_costs(fvec,lbl,prot,plbl,om,mu);  
       te(1) = sum(marg>0)/nfv;
       cf(1) = costf; 
       stepsizem(1)=etam;
       stepsizep(1)=etap; 
       
  [tpr,fpr,auroc,thresholds] = compute_roc(lbl>1,score);
       auc(1)=auroc; 
 
       
  % perform the first ncop steps of gradient descent
  for inistep=1: ncop;
      % actual batch gradient step
      
      [prot,om]= do_batchstep(fvec,lbl,prot,plbl,om,etap,etam,mu,mode); 
      protcop(inistep,:,:)= prot; 
      omcop  (inistep,:,:)= om;
     
      % determine and save training set performances 
       [costf,~,marg,score] = compute_costs(fvec,lbl,prot,plbl,om,mu);  
       te(inistep+1) = sum(marg>0)/nfv;
       cf(inistep+1) = costf; 
       stepsizem(inistep+1)=etam; 
       stepsizep(inistep+1)=etap;
       % compute training set errors and cost function values
       for icls=1:ncls;
          % compute class-wise errors (positive margin = error) 
          cw(inistep+1,icls) = sum(marg(lbl==icls)>0)/sum(lbl==icls); 
       end;
       
       % training set roc with respect to class 1 versus all others only
      
       [tpr,fpr,auroc,thresholds] = compute_roc(lbl>1,score);
       auc(inistep+1)=auroc;
  end; 
  
  % compute cost functions, crisp labels, margins and scores
  % scores with respect to class 1 (negative) or all others (positive)
  [~,~,~,score]    = compute_costs(fvec,lbl,prot,plbl,om,mu); 
   
for jstep=(ncop+1):totalsteps;    
 % calculate mean positions over latest steps
 protmean = squeeze(mean(protcop,1)); 
 ommean = squeeze(mean(omcop,1));
 ommean=ommean/sqrt(sum(sum(abs(ommean).^2))); 
 % note: normalization does not change cost function value
 %       but is done here for consistency

% compute cost functions for mean prototypes, mean matrix and both 
[costmp,~,~,score] = compute_costs(fvec,lbl,protmean,plbl,om, 0);
[costmm,~,~,score] = compute_costs(fvec,lbl,prot,    plbl,ommean,mu); 
% [costm, ~,~,score ] = compute_costs(fvec,lbl,protmean,plbl,ommean,mu); 

% remember old positions for Papari procedure
ombefore=om; 
protbefore=prot;
 
 % perform next step and compute costs etc.
[prot,om]= do_batchstep (fvec,lbl,prot,plbl,om,etap,etam,mu,mode);  
    
[costf,~,~,score] = compute_costs(fvec,lbl,prot,plbl,om,mu); 
 

% by default, step sizes are increased in every step
 etam=etam*incfac; % (small) increase of step sizes
 etap=etap*incfac; % at each learning step to enforce oscillatory behavior 

% costfunction values to compare with for Papari procedure
% evaluated w.r.t. changing only matrix or prototype
[costfp,~,~,score] = compute_costs(fvec,lbl,prot,plbl,ombefore,0);
[costfm,~,~,score] = compute_costs(fvec,lbl,protbefore,plbl,om,mu); 
   
% heuristic extension of Papari procedure
% treats matrix and prototype step sizes separately
 if (costmp <= costfp ); % decrease prototype step size and jump
                         % to mean prototypes
     etap = etap/decfac;
     prot = protmean;
 end; 
 if (costmm <= costfm ); % decrease matrix step size and jump
                         % to mean matrix
     etam = etam/decfac;
     om = ommean;   
 end
 
 % update the copies of the latest steps, shift stack of stored configs. 
 % plenty of room for improvement, I guess ...
 for iicop = 1:ncop-1;
   protcop(iicop,:,:)=protcop(iicop+1,:,:);
   omcop(iicop,:,:)  =omcop  (iicop+1,:,:);
 end;
 protcop(ncop,:,:)=prot;  omcop(ncop,:,:)=om;
 
 % determine training and test set performances 
 % here: costfunction without penalty term! 
[costf0,~,marg,score] = compute_costs(fvec,lbl,prot,plbl,om,0); 

 
 % compute total and class-wise training set errors
 te(jstep+1) = sum(marg>0)/nfv;
 cf(jstep+1) = costf0; 
 for icls=1:ncls;
     cw(jstep+1,icls) = sum(marg(lbl==icls)>0)/sum(lbl==icls); 
 end;
 stepsizem(jstep+1)=etam;
 stepsizep(jstep+1)=etap;
 
 % ROC with respect to class 1 (negative) vs. all others (positive)
 binlbl= lbl>1; 
 [tpr,fpr,auroc,thresholds] = compute_roc(binlbl,score);
 auc(jstep+1)=auroc; 
 
%  figure(10);plot(fpr,tpr);pause(1)
 
 
end;   % totalsteps training steps performed

%if the data was z transformed then also save the inverse prototypes,
%actually it is not necessary since the mf and st are returned.
if doztr == 1
    protsInv = do_inversezscore(prot, mf, st);
else
    protsInv = prot;
end

lambda=om'*om;   % actual relevance matrix
% define structures corresponding to the trained system and training curves
gmlvq_system =    struct('protos',prot, 'protosInv',protsInv,'lambda',lambda,'plbl',plbl,...
                         'mean_features',mf,'std_features',st); 
training_curves = struct('costs',cf,'train_error',te,...
                         'class_wise',cw,'auroc',auc); 
param_set = struct('totalsteps',totalsteps,'doztr',...
          doztr,'mode',mode,'rndinit',rndinit,...
          'etam0',etam0,'etap0',etap0,'etamfin',etam,'etapfin',etap,...
          'mu',mu,'decfac',decfac,'infac',incfac,'ncop',ncop,...
          'rngseed',rngseed);
                                        

% learning curves and system visualization, only if showplot = 1                      
if(showplots==1)


scrsz = get(0,'ScreenSize');
figure('Position',[1 scrsz(4)*8/10 scrsz(3)*4/10 scrsz(4)*8/10])   
    
 % figure(1);                                       % learning curves 
 msize=15;   % size of symbols
 
 totl=totalsteps+1;
 subplot(3,2,1);  % plot glvq cost fucntion vs. steps
    plot([1:totl],cf,'b.','MarkerSize',msize); hold on;
    
   title('glvq costs per example w/o penalty term ',...
       'FontName','LucidaSans', 'FontWeight','bold'); 
   xlabel('gradient steps');
   axis([0 totl -1 1]); axis 'auto y';
 
 subplot(3,2,2);  % plot total training error vs. steps
   plot(1:totl,te,'r.','Markersize',msize); hold on;
   
   title('total training error',...
       'FontName','LucidaSans', 'FontWeight','bold'); 
   xlabel('gradient steps'); 
   axis([0 totl 0 1]); axis 'auto y';
   
   subplot(3,2,3); % plot the class-wise errors vs. steps
       plot(1,cw(1,:,:),'.','MarkerSize',msize+10); hold on;
       plot(1,cw(1,:,:),'w.','MarkerSize',msize+10);
       legend(num2str([1:ncls]'),'Location','NorthEast');
       plot(1:totl,cw,'.','MarkerSize',msize);    
  
   title('class-wise training errors',...
       'FontName','LucidaSans', 'FontWeight','bold'); 
   xlabel('gradient steps'); 
   axis([0 totl 0 1]);  axis 'auto y';
   
   
 subplot(3,2,4);   % plot AUC (ROC) vs. steps
   plot(1:totl,auc,'k.','MarkerSize',msize); 
   axis([ 0 totl min(0.9,min(auc)) 1.05 ]); 
   title('AUC(ROC), class 1 vs. all others',...
       'FontName','LucidaSans', 'FontWeight','bold');
 
   
   subplot(3,2,5);  % plot glvq cost fucntion vs. steps
    plot([1:totl],stepsizep,'b.','MarkerSize',msize); hold on;
    plot([1:totl],stepsizem,'r.','Markersize',msize);
   title('stepsizes',...
       'FontName','LucidaSans', 'FontWeight','bold'); 
   xlabel('gradient steps');
   legend('prototype','relevances','Location','NorthEast');
   axis([0 totl -1 1]); axis 'auto y';
   
   
 figure(2);             % display the ROC curve of the final classifier 
   fprnpc = fpr(thresholds==0.5); % false positive of NPC
   tprnpc = tpr(thresholds==0.5); % true  positive of NPC
   
   plot(fpr,tpr,'-'); hold on;
   plot(fprnpc,tprnpc,'ko','MarkerSize',10,'MarkerFaceColor','b');
   axis square;
   xlabel('false positive rate');
   ylabel('true positive rate');
   title('Training ROC, class 1 (neg.) vs. all others (pos.)',...
       'FontName','LucidaSans', 'FontWeight','bold'); 
   legend(['AUC = ',num2str(auc(jstep))],'NPC', 'Location','SouthEast');
   plot([0 1],[0 1],'k:'); 
   hold off;
 

  figure(3);                   % visualize prototyeps and lambda matrix 
  
  display_gmlvq(prot,lambda,plbl,ndim); 
    
%   figure(4);         % visualize data set in terms of projections on the
%                      % leading eigenvectors of Lambda
%   visu_2d(lambda,prot,fvec,lbl,plbl,1); axis tight; axis square;
%   title('2d-visualization of the data set',...
%         'FontName','LucidaSans', 'FontWeight','bold');
  end;


                     
                                 
                     
                     
                    
                     
                          
                         
                     
                     
   
                     
                     
   

