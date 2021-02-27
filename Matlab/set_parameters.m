
function [showplots,doztr,mode,rndinit,etam,etap,mu,decfac,incfac,ncop] =...
                                        set_parameters(fvec)
  
% set general parameters 
% set initial step sizes and control parameters of
% modified procedure based on [Papari, Bunte, Biehl] 

nfv=size(fvec,1);
ndim = size(fvec,2);

% GMLVQ parameters, explained below
showplots  = 1;  
doztr      = 1; 
mode       = 1; 
rndinit    = 0; 
mu         = 0;

% showplots (0 or 1): plot learning curves etc? recommended: 1
% doztr (0 or 1): perform z-score transformation based on training set
% mode 
  % 0 for matrix without null-space correction            
  % 1 for matrix with null-space correction
  % 2 for diagonal matrix (GRLVQ)                         discouraged 
  % 3 for GLVQ with Euclidean distance (equivalent)
% rndinit
  % 0 for initialization of relevances as identity matrix 
  % 1 for randomized initialization of relevance matrix
% mu
  % control parameter of penalty term for singularity of Lambda
  % mu=0: unmodified GMLVQ
  % mu>0: prevents singular Lambda
  % mu very large: Lambda proportional to Identity (Euclidean) 
 
  
% parameters of stepsize adaptation 
if (mode<2); % full matrix updates with (0) or w/o (1) null space correction
  etam   = 2; % suggestion: 2
  etap   = 1; % suggestion: 1
  if (mode==0); 
      display('matrix relevances without null-space correction'); 
  end;
  if (mode==1);
      display('matrix relevances with null-space correction'); 
  end;

elseif (mode==2) % diagonal relevances only, DISCOURAGED
  display('diagonal relevances, not encouraged, sensitive to step sizes');
  etam   = 0.2;  % initital step size for diagonal matrix updates
  etap   = 0.1;  % initital step size for prototype update

elseif (mode==3) % GLVQ, equivalent to Euclidean distance
    display('GLVQ without relevances')
    etam=0; 
    etap = 1; 
end;   
  decfac = 1.5;       % step size factor (decrease) for Papari steps
  incfac = 1.1;       % step size factor (increase) for all steps
  ncop = 5;           % number of waypoints stored and averaged      

if (nfv <= ndim & mode==0); 
   display('dim. > # of examples, null-space correction recommended');
end; 
if (doztr==0); 
    display('no z-score transformation, you may have to adjust step sizes'); 
    if (mode<3);
    display('rescale relevances for proper interpretation'); 
    end;
end;

end

