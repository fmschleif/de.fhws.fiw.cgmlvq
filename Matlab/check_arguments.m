
% check consistency of some arguments and input parameters

function [lbl] = check_arguments(plbl,lbl,fvec,ncop,totalsteps)

% plbl:  prototype labels
% lbl :  data set labels
% fvec:  feature vectors in data set
% ncop:  number of copies in step size control procedure
% totalsteps: total number of batch gradient update steps

% output:
% lbl :  data set labels, protentially transposed for consistency

if(size(lbl,2)>1);   % lbl may be column or row vector
    lbl=lbl';
    warning('vector lbl has been transposed');
end;

if size(fvec,1) ~= length(lbl) 
    error('number of training labels differs from number of samples');
end; 

if((min(lbl)~=1)|max(lbl)~=length(unique(lbl)))
   warning(['unique(lbl)=  ',num2str(unique(lbl))]);
   error('data labels should be: 1,2,3,...,nclasses');
end;

if(length(unique(plbl))>2)
    warning(['multi-class problem, ROC analysis is for class 1 (neg.)',...
        ' vs. all others (pos.)']);
end;

if (length(unique(plbl)) ~= length(unique(lbl)))
   warning(['unique(plbl)=   ',num2str(unique(plbl))]);
   error('number of prototype labels must equal number of classes');
end; 

if (sum( unique(plbl')~=unique(lbl)) >0 ) 
   warning(['unique(plbl)=   ',num2str(unique(plbl))]);
   error('prototype labels inconsistent with data, please rename/reorder'); 
end;

for i=1:size(fvec,2);
    st(i)=std(fvec(:,i));  % standard deviation of feature i
end;
display(' ');
display(['minimum standard deviation of features: ',num2str(min(st))]); 
display(' ');
if (min(st)<1.e-10); 
    error('at least one feature displays (close to) zero variance');
end; 

if (ncop >= totalsteps); 
    error('number of gradient steps must be larger than ncop');
end; 

end

