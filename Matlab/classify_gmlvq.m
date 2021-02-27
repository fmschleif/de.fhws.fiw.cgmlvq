
% apply a gmlvq classifier to a given data set
% with unknown class labels for predication
% or known class labels for testing/validation 

function [crisp,score,margin,costf]= ...
                      classify_gmlvq(gmlvq_system,fvec,ztr,lbl)

% for classification with unknown ground truth labels, use: 
%                 [crisp,score] =classify_gmlvq(gmlvq_system,fvec,ztr) 

% if used for testing with known test labels (lbl) you can use:
%                 [crisp,score,margin,costf]...
%                           =classify_gmlvq(gmlvq_system,fvec,ztr,lbl) 

% input
prot=gmlvq_system.protos;         % prototypes 
lambda=gmlvq_system.lambda;       % relevance matrix lambda
plbl=gmlvq_system.plbl;           % prototype labels 
mf=gmlvq_system.mean_features;    % mean features from potential z-score
st=gmlvq_system.std_features;     % st.dev. from potential z-score transf.
% fvec  : set of feature vectors to be classified 
% ztr = 1 if z-score transformation was done in the training

% output
% crisp:  crisp labels of Nearest-Prototype-Classifier
% score:  distance based score with respect to class 1
% margin: GLVQ-type margin with respect to class 1 evaluated in glvqcosts
% costf:  GLVQ costfunction (if ground truth is known)

omat=sqrtm(lambda); % symmetric matrix square root as one representation
                    % of the distance measure 

nfv = size(fvec,1);          % number of feature vectors in training set
ndim = size(fvec,2);         % dimension of feature vectors
ncls = length(unique(lbl));  % number of classes 
nprots = length(plbl);       % total number of prototypes

if (nargin<4 || isempty(lbl)); % ground truth unknown 
   lbl=ones(1,ndim); lbl(ceil(ndim/2),end)=2; % fake labels, meaningless
                                              % if ground truth unknown
end; 

% if z-transformation was applied in training, apply the same here:
if (ztr==1); 
    for i=1:nfv;
        fvec(i,:)= (fvec(i,:)-mf)./st; 
    end;
end;

% call glvqcosts, crout=crisp labels
% score between 0= "class 1 certainly" and 1= "any other class"                                         
% margin and costf are meaningful only if lbl is ground truth

mu=0; % cost function can be computed without penalty term for margins/score
[costf,crisp,margin,score] = compute_costs(fvec,lbl,prot,plbl,omat,mu);



 

                     
                     

                         
                     
                     
   
                     
                     
   

