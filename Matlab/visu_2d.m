function  [] = visu_2d(lam,w,fvec,lbl,plbl,show_legend);

% input
% lam  : global relevance matrix
% w    : set of prototype vectors
% fvec : feature vector, must be scaled and transformed as in training
%        in order to be consistent with the glmvq system
% lbl  : labels assigned to feature vectors
% plbl : prototype labels
% show_legend = 1 : display legends in plots 

ndim= size(fvec,2);     % dimension of feature vectors
nfv= size(fvec,1);      % number of feature vectors
np = size(w,1);         % number of prototypes

[ev,ew]=eig(lam);       % determine eigenvectors and -values of lambda

omat(1,:)= ev(:,end);   % leading eigenvector of lambda
omat(2,:)= ev(:,end-1); % second eigenvector of lambda 

  scale1=sqrt(ew(end,end));
  scale2=sqrt(ew(end-1,end-1));  % scale projections with sqrt(eigenvalue) 
 % scale1=1; scale2=2;  % plot projections on orthonormal eigenvectors

proj= zeros(nfv,2);     % projections of all feature vectors
for i=1:nfv;
    proj(i,1) = scale1*dot(omat(1,:),fvec(i,:));  % proj. on eigenvector 1
    proj(i,2) = scale2*dot(omat(2,:),fvec(i,:));  % proj. on eigenvector 2 
end 

   projw= zeros(np,2);  % projections of prototypes
for i=1:np;
    projw(i,1) = scale1*dot(omat(1,:),w(i,:));  % proj. on eigenvector 1
    projw(i,2) = scale2*dot(omat(2,:),w(i,:));  % proj. on eigenvector 2
end 

% define symbols and colors for up to 7 classes  (prototypes)
  symbstrings = ['b';'r';'g';'c';'m';'y';'k'];   
  lgstr = [];   % initialize legend as empty text
for ip=1:np;
      plot(projw(ip,1), projw(ip,2),'ko',...            % plot prototype
          'MarkerSize',10,...                           % large symbol in
          'MarkerFaceColor',symbstrings(plbl(ip),1));   % color acc. to 
      hold on;                                          % prototype label
      lgstr = [lgstr;num2str(plbl(ip))]; 
end;      
     
if (show_legend==1); 
legend('Location','NorthEastOutside',lgstr); % legend of prototype symbols
end; 

% define symbols and colors for up to 7 classes  (data points)
    symbstrings = ['bo';'ro';'go';'co';'mo';'yo';'ko']; 
for i=1:nfv; 
     plot(proj(i,1),proj(i,2),'ko',... % symbstrings(lbl(i),:),...
         'MarkerFaceColor',symbstrings(lbl(i))); % plot data points 
   % optional: mark samples by their number 
   %  text(proj(i,1)+0.1,proj(i,2),num2str(i));
end; 
   

% plot prototypes again to make them visible on top of data, larger now
symbstrings = ['b';'r';'g';'c';'m';'y';'k']; % up to 7 classes
for ip=1:np;
      plot(projw(ip,1), projw(ip,2),'ko',...
          'MarkerSize',20,'MarkerFaceColor',symbstrings(plbl(ip)));
    %  plot(projw(ip,1), projw(ip,2),'wp',...
    %      'MarkerSize',8); % 'MarkerFaceColor','w');     
end;      

if (scale1==1 && scale2==1) 
   xlabel('projection on first eigenvector of \Lambda');
   ylabel('projection on second eigenvector of \Lambda');
else
 %   xlabel('projection on first eigenvector of \Lambda');
 %  ylabel('projection on second eigenvector of \Lambda');
    xlabel('proj. on first eigenvector of \Lambda, scaled by sqrt of eigenvalue');
    ylabel('proj. on second eigenvector of \Lambda, scaled by sqrt of eigenvalue');   
end
axis square;  axis equal;
title ('visualization of all data',...
       'FontName','LucidaSans','FontWeight','bold');
hold off; 
