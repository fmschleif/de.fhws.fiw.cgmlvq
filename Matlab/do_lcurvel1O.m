
function [w,omega,cftra,tetra,cwtra,auctra]= ...
   do_lcurvel1O(fvec,lbl,fvecval,lblval,plbl,mode,rndinit,totalsteps);    
 
% gmlvq with global matrix only, square Omega, potentially diagonal
% batch gradient descent with step size control
% following a modified Papari procedure (unpublished) 
% perform training based using fvec, lbl
% evaluate performance measures after each step
% with respect to training set and a single validation example

% input arguments    
% fvec, lbl              training data, feature vectors and labels
% fvecval, lblval        single validation sample, feature vector and label
% plbl: labels assigned to prototypes, also specifies number per class
% e.g. plbl=[1,1,2,2,3] for 5 prototypes with labels 1,2,3  

% set GMLVQ system and training parameters including Papari procedure
[~,~,mode,rndinit,etam,etap,mu,decfac,incfac,ncop] =...
                                        set_parameters(fvec);
                                    
% mode 
  % 0 for matrix without regularization
  % 1 for matrix with null-space correction
  % 2 for diagonal updates only  
% rndinit
  % 0 for initialization of relevances as identity matrix 
  % 1 for randomized initialization of relevance matrix
% totalsteps: number of batch gradient steps to be performed
% etam:    step size of matrix updates
% etap:    step size of prototype updates
% mu  :    controlparameter for
% decfac:  factor for decrease of step sizes for oscillatory behavior
% incfac:  factor for increase of step sizes for smooth behavior
% ncop:    number of copies in Papari procedure

nfv = size(fvec,1);          % number of feature vectors in training set
nfvval = size(fvecval,1);    % number of feature vectors for validation
ndim = size(fvec,2);         % dimension of feature vectors
ncls = length(unique(lbl));  % number of classes 
nprots = length(plbl);       % total number of prototypes
          
% initialize prototypes and omega 
  [proti,omi] =  set_initial(fvec,lbl,plbl,mode,rndinit);
  prot=proti;  om =omi;   % initial values   
% copies of prototypes and omegas stored in protcop and omcop
  protcop = zeros(ncop,size(prot,1),size(prot,2));
  omcop   = zeros(ncop,size(om,1) , size(om,2) );

% learning curves, perfomance w.r.t. training set
 cftra =  NaN(totalsteps,1); 
 tetra =  NaN(totalsteps,1); 
 cwtra =  NaN(totalsteps,ncls); 
 auctra = NaN(totalsteps,1); 
  
  
  % perform the first ncop steps of gradient descent
  for inistep=1: ncop;
      [prot,om]= do_batchstep(fvec,lbl,prot,plbl,om,etap,etam,mu,mode); 
      protcop(inistep,:,:)= prot; 
      omcop  (inistep,:,:)= om;
      
      om=om/sqrt(sum(sum(om.*om))); 
      % compute initial costs without penalty term
      [costf,~,marg,score]    = compute_costs(fvec,lbl,prot,plbl,om,0);
      [costval,~,margval,scoreval]= ...
                         compute_costs(fvecval,lblval,prot,plbl,om,0);    
      cftra(inistep)= costf; cfval(inistep)= costval;
      tetra(inistep)= sum(marg>0)/nfv; 
      teval(inistep) = sum(margval>0)/nfvval; 
      for icls=1:ncls;
          cwtra(inistep,icls) = sum(marg(lbl==icls)>0)/sum(lbl==icls); 
          cwval(inistep,icls) = sum(margval(lblval==icls)>0)/sum(lblval==icls);
      end;    
      
      % roc with respect to class 1 versus all others only
     [~,~,auroc,thresholds] = compute_roc(lbl>1,score);
     auctra(inistep)=auroc; 
      
      
  end;
 
  % scores are independent of mu, use mu=0 here
  [~,~,~,score]    = compute_costs(fvec,lbl,prot,plbl,om,0); 

 
for jstep=(ncop+1):totalsteps;  
 % calculate mean positions over latest steps
 protmean = squeeze(mean(protcop,1)); 
 ommean = squeeze(mean(omcop,1));
 ommean=ommean/sqrt(sum(sum(ommean.^2))); 
 % note: normalization does not change cost function value
 %       but is done for consistency
 
 % compute cost functions for mean prototypes, mean matrix and both 
[costmp,~,~,score ] = compute_costs(fvec,lbl,protmean,plbl,om, 0);
[costmm,~,~,score ] = compute_costs(fvec,lbl,prot,    plbl,ommean,mu); 
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
                          
 % update the copies of the latest steps, shift stack 
 for iicop = 1:ncop-1;
   protcop(iicop,:,:)=protcop(iicop+1,:,:);
   omcop(iicop,:,:)  =omcop  (iicop+1,:,:);
 end;
 protcop(ncop,:,:)=prot;  omcop(ncop,:,:)=om;
 
 % determine training and test set performances (costs for mu=0)
[costf,~,marg,score] = compute_costs(fvec,lbl,prot,plbl,om,0); 

 [~,~,margval,scoreval]= ...
                      compute_costs(fvecval,lblval,prot,plbl,om,0);

       
 cftra(jstep)= costf;
 tetra(jstep)= sum(marg>0)/nfv; 
 teval(jstep) = sum(margval>0)/nfvval; % this is either 0 or 1 for
                                       % the single example (nfvval=1)
 
 for icls=1:ncls;
     % class-wise training errors
     cwtra(jstep,icls) = sum(marg(lbl==icls)>0)/sum(lbl==icls); 
 end; 
 
 % training set roc with respect to class 1 versus all others only
 [~,~,auroc,thresholds] = compute_roc(lbl>1,score);
 auctra(jstep)=auroc; 
 
end; % end of totalsteps gradient steps (jrun)

 % output final lvq configuration after totalsteps steps: 
 w  =prot;
 omega =om;

 % end of batch gradient descent

