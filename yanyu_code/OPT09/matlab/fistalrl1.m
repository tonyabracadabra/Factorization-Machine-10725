% fistalrl1 - L1-regularized logistic regression by FISTA algorithm
%
% Reference:
% A. Beck and M. Teboulle. A fast iterative shrinkage-thresholding
% algorithm for linear inverse problems. 
% SIAM J. Imaging Sciences, 2(1):183202, 2009.
%
% Copyright (C) 2010  Ryota Tomioka

function [xx,stat]=fistalrl1(A,ytr,lambda,varargin)

opt=propertylist2struct(varargin{:});
opt=set_defaults(opt, 'eta',2,'tol',0,'w0',[],'maxiter',1000);


[m,n]=size(A);
tt=1;
L =ones(1,opt.maxiter);
xx=zeros(n,1);
yy=xx;

fval=nan*ones(1,opt.maxiter);
dist=nan*ones(1,opt.maxiter);
time=nan*ones(1,opt.maxiter);

info.ginfo=0;

time0=cputime;

[floss,gloss]=loss_lrp(A*xx,ytr);
fval(1)=floss+lambda*sum(abs(xx));
gg=fix_grad(xx, A'*gloss+lambda*sign(xx), lambda);

if ~isempty(opt.w0)
  dist(1)=norm(xx-opt.w0);
end
time(1)=cputime-time0;

kk=1;
while 1
  fprintf('[%d] fval=%g normg=%g dist=%g L=%g\n', kk, fval(kk), norm(gg), ...
          dist(kk), L(kk));

  fval(kk+1)=inf; qval=0;
  if kk>1
    L(kk)=L(kk-1);
  end
  while 1
    Ayy=A*yy;
    [floss,gloss]=loss_lrp(Ayy,ytr);
    Atgg = A'*gloss;
    xx1=softth(yy-Atgg/L(kk), lambda/L(kk));
    gg = fix_grad(xx, Atgg+lambda*sign(xx), lambda);
    
    Axx1=A*xx1;
    fval(kk+1)=loss_lrp(Axx1,ytr)+lambda*sum(abs(xx1));
    qval=floss+gloss'*(Axx1-Ayy)+0.5*L(kk)*norm(xx1-yy)^2+lambda*sum(abs(xx1));
    if fval(kk+1)<qval
      break;
    end
    fprintf('fval=%g qval=%g\n',fval(kk+1),qval);
    L(kk)=L(kk)*opt.eta;
  end
  
  tt1 = (1+sqrt(1+4*tt^2))/2;
  
  yy = xx1 + (tt-1)/tt1*(xx1-xx);

  xx=xx1;
  tt=tt1;

  kk=kk+1;

  if ~isempty(opt.w0)
    dist(kk)=norm(xx-opt.w0);
  end
  time(kk)=cputime-time0;
  
  if fval(kk)<opt.tol
    break;
  end
  
  if kk==opt.maxiter
    break;
  end
end

stat=struct('fval',fval,...
            'dist',dist,...
            'L',L,...
            'opt',opt,...
            'time',time);




function vv=softth(vv,lambda)
n = size(vv,1);

Ip=find(vv>lambda);
In=find(vv<-lambda);

vv=sparse([Ip;In],1,[vv(Ip)-lambda;vv(In)+lambda],n,1);



function gg=fix_grad(xx, gg, lambda)

% gg  =A'*gg+lambda*sign(xx);

I1=find(xx==0 & gg>0);
I2=find(xx==0 & gg<0);

gg(I1)=gg(I1)+max(-lambda, -gg(I1));
gg(I2)=gg(I2)+min(lambda, -gg(I2));


function [floss, gloss]=loss_lrp(zz, yy)

zy      = zz.*yy;
z2      = 0.5*[zy, -zy];
outmax  = max(z2,[],2);
sumexp  = sum(exp(z2-outmax(:,[1,1])),2);
logpout = z2-(outmax+log(sumexp))*ones(1,2);
pout    = exp(logpout);

floss   = -sum(logpout(:,1));
gloss   = -yy.*pout(:,2);
