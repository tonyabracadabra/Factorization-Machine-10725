% owlbfgs - Orthant-wise L-BFGS algorithm
%
% Syntax:
%  [xx, status] = lbfgs(fun, xx, ll, uu, <opt>)
%
% Input:
%  fun     - objective function
%  xx      - Initial point for optimization
%  ll      - lower bound on xx
%  uu      - upper bound on xx
%  Ac      - inequality constraint:
%  bc      -     Ac*xx<=bc
%  opt     - Struct or property/value list of optional properties:
%   .m          - size of limited memory
%   .epsg       - gradient tolerance
%   .maxiter    - maximum number of iterations
%   .display    - display level
% 
% Output:
%  xx      - Final point of optimization
%  status  - Various numbers
%
% Reference:
%  Jorge Nocedal. Updating Quasi-Newton Matrices with Limited
%  Storage. Mathematics of Computation, Vol. 35, No. 151,
%  pp. 773--782, 1980.
% 
%  Galen Andrew and Jianfeng Gao. Scalable training of
%  L1-regularized log-linear models. In Proceedings of the 24th
%  International Conference on Machine Learning (ICML 2007),
%  pp. 33-40, 2007.
%
% See also
%  libLBFGS written by Naoaki Okazaki
%  http://www.chokkan.org/software/liblbfgs/
%
% Copyright(c) 2009 Ryota Tomioka
% This software is distributed under the MIT license. See license.txt

function [W, counter, objs, status] = owlbfgs_for_argmin_w0wW(fun, W, w0, w, U, u, x, y, yhat, lambda2, lambda3, tempUu, rho, counter, objs, varargin)
objs = [objs; [compute_obj1(lambda2, lambda3, x, y, W, w, w0, U, u, rho), 1]];
opt=propertylist2struct(varargin{:});
opt=set_defaults(opt, 'm', 6,...
                      'ftol', 1e-5, ...
                      'maxiter', 0,...
                      'max_linesearch', 50,...
                      'display', 0,...
                      'epsg', 1e-5,...
                      'tol',[],...
                      'w0', []);

nn = size(W);
% objs = [];
fval_sv=nan*ones(1,opt.maxiter);
time   =nan*ones(1,opt.maxiter);
dist   =nan*ones(1,opt.maxiter);

% Limited memory
lm = repmat(struct('s',zeros(nn),'y',zeros(nn),'ys',0,'alpha',0),[1, opt.m]);

[fval,gg]=eval_obj(fun, W, x, yhat, lambda2, tempUu, rho);

pre_obj = fval;
% The initial step is gradient
dd = -gg;

kk = 1;
stp = 1/norm(dd);

ixend = 1;
bound = 0;
time0=cputime;
while 1
  if ~isempty(opt.w0)
    dist(kk)=norm(W-opt.w0);
  end
  time(kk)=cputime-time0;
  fval_sv(kk)=fval;

  % Progress report
  gnorm = norm(gg);
  if opt.display>1
    if ~isempty(opt.w0)
      fprintf('[%d] fval=%g gnorm=%g dist=%g step=%g\n',kk,fval,gnorm,dist(kk),stp);
    else
      fprintf('[%d] fval=%g gnorm=%g step=%g\n',kk,fval,gnorm,stp);
    end
  end
  
  % Keep previous values
  fp = fval;
  xxp = W;
  ggp = gg;

  % Perform line search
  stp = 1.0;
  [ret, W,fval,gg,stp]=...
      linesearch_backtracking(fun, W, x, yhat, lambda2, tempUu, rho, fval, gg, dd, stp,opt);
  

  
  if (ret<0 || kk==opt.maxiter-1 ...
        || (~isempty(opt.tol) && fval<opt.tol) ...
        || (isempty(opt.tol) && gnorm<opt.epsg))
    time(kk+1)   =cputime-time0;
    fval_sv(kk+1)=fval;
    if ~isempty(opt.w0)
      dist(kk+1)   =norm(W-opt.w0);
    end
  
    if ((~isempty(opt.tol) && fval<opt.tol) ...
        || (isempty(opt.tol) && gnorm<opt.epsg))
      ret=0;
      if opt.display>1
        fprintf('Optimization success! gnorm=%g\n',gnorm);
      end
      ret=0;
    elseif kk==opt.maxiter-1
      if opt.display>0
        fprintf('Maximum #iterations=%d reached.\n', kk);
      end
      ret = -3;
    end
    break;
  end


  % L-BFGS update
  if opt.m>0
    lm(ixend).s = W-xxp;
    lm(ixend).y = gg-ggp;
    ys = sum(sum(lm(ixend).y.*lm(ixend).s)); yy = sum(sum(lm(ixend).y.^2));
    lm(ixend).ys  = ys;
  else
    ys = 1; yy = 1;
  end
  
  bound = min(bound+1, opt.m);
  ixend = (opt.m>0)*(mod(ixend, opt.m)+1);

  % Initially set the negative gradient as descent direction
  dd = -gg;
  
  jj = ixend;
  for ii=1:bound
    jj = mod(jj + opt.m -2, opt.m)+1;
    lm(jj).alpha = sum(sum(lm(jj).s.*dd))/lm(jj).ys;
    dd = dd -lm(jj).alpha*lm(jj).y;
  end

  dd = dd *(ys/yy);
  
  for ii=1:bound
    beta = sum(sum(lm(jj).y.*dd))/lm(jj).ys;
    dd = dd + (lm(jj).alpha-beta)*lm(jj).s;
    jj = mod(jj,opt.m)+1;
  end

  % Fix direction in the orthant-wise manner
  dd(dd.*gg>=0)=0;

  
  kk = kk + 1;
  counter(1) = counter(1) + 1;
  
  objs = [objs; [compute_obj1(lambda2, lambda3, x, y, W, w, w0, U, u, rho), 1]];
  if fp - fval  < opt.tol
      break
%   else
%       fp = fval;
  end
end

status=struct('ret', ret,...
              'kk', kk,...
              'fval', fval_sv,...
              'gg', gg,...
              'time', time,...
              'dist', dist,...
              'opt', opt);


function [fval,gg]=eval_obj(fun, W, x, yhat, lambda2, tempUu, rho)

[fval,gg]=feval(fun, W, x, yhat, lambda2, tempUu, rho);

% fval=fval+lambda*sum(abs(xx));
% 
% gg  =A'*gg+lambda*sign(xx);
% 
% I1=find(xx==0 & gg>0);
% I2=find(xx==0 & gg<0);
% 
% gg(I1)=gg(I1)+max(-lambda, -gg(I1));
% gg(I2)=gg(I2)+min(lambda, -gg(I2));
function L = compute_obj1(lambda2, lambda3, x, y, W, w, w0, U, u, rho)
    y_hat = w0 + x * w + diag(x * W * x');
    L = sum((y - y_hat) .^ 2) + lambda2 * sum(sum(abs(W))) + lambda3 * sum(w .^ 2) + rho / 2 * sum(sum((W - U + u) .^ 2));


function [ret, W, fval, gg, step]...
    =linesearch_backtracking(fun, W, x, yhat, lambda2, tempUu, rho, fval, gg, dd, step, opt)

floss=0;
gloss=zeros(size(gg));

dginit=sum(sum(gg.*dd));

if dginit>=0
  if opt.display>0
    fprintf('dg=%g is not a descending direction!\n', dginit);
  end
  step = 0;
  ret = -1;
  return;
end


xx0 = W;
f0  = fval;
gg0 = gg;
cc = 0;

if opt.display>2
  fprintf('finit=%.20f\n',f0);
end

while cc<opt.max_linesearch
  ftest = f0  + opt.ftol*step*dginit;
  W    = xx0 + step*dd;
  
  % Make sure that the new point belongs to the same orthant
  W(W.*xx0<0)=0;
  
  [fval, gg]=eval_obj(fun, W, x, yhat, lambda2, tempUu, rho);
    
  if fval<=ftest
      break;
  end

  if opt.display>2
    fprintf('[%d] step=%g fval=%.20f > ftest=%.20f\n', cc, step, fval, ftest);
  end
  
  step = step/2;
  cc = cc+1;
end

if cc==opt.max_linesearch
  if opt.display>0
    fprintf('Maximum linesearch=%d reached\n', cc);
  end
  W   = xx0;
  [fval, gg]=eval_obj(fun, W, x, yhat, lambda2, tempUu, rho);
  step = 0;
  ret = -2;
  return;
end

ret = 0;
