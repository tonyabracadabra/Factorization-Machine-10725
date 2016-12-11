function [niters, times, fvals, dists, logs]=run_exp(m,n,lmd,maxiter)

niters=zeros(1,length(maxiter));
times =cell(1,length(maxiter));
fvals =cell(1,length(maxiter));
dists =cell(1,length(maxiter)+2);

fprintf('Generating a random test problem.\n');
fprintf('n=%d...\n',n);
A=randn(m,n);
k = round(0.04*n);
I=randperm(n);w0=zeros(n,1);w0(I(1:k))=sign(randn(k,1));
yy=sign(A*w0+0.01*randn(m,1));


lmd=1;

fprintf('First run DAL to high precision (RDG<1e-9).\n');
[ww,bias,stat]=dallrl1(zeros(n,1),[],A,yy,lmd,'tol',1e-9,'solver','cg','iter',1,'maxiter',maxiter(1),'eta',1);

info.ginfo=nan;
[fval,gg,info]=obj_lrl1(ww(:,end),info,A,yy,lmd);

fprintf('norm(grad)=%g\n',norm(gg));

fprintf(['Assuming that the support is correct, fit logistic model ' ...
         'within the active set.\n']);

w0=fitlrl1(ww(:,end),A,yy,lmd);
[fval0,gg0,info]=obj_lrl1(w0,info,A,yy,lmd);
fprintf('norm(grad)=%g\n',norm(gg0));

fprintf('Empirically computing sigma...\n');
dtmp=sum((ww-w0*ones(1,size(ww,2))).^2);
sigma = 0.7*min((stat.fval-fval0)./dtmp);
fprintf('sigma=%g\n',sigma);

fprintf('Executing DAL to moderate precision (RDG<1e-7)\n');
[ww,bias,stat]=dallrl1(zeros(n,1),[],A,yy,lmd,'tol',1e-7,'solver','cg', ...
                     'iter',1,'eta',1);

niters(1)=length(stat.fval);
fvals{1} =fillnan(stat.fval-fval0,maxiter(1));

if any(fvals{1})<0
  keyboard;
end

times{1} =fillnan(stat.time,maxiter(1));

dists{1}=fillnan(sqrt(sum((ww-w0*ones(1,size(ww,2))).^2)),maxiter(1));
dist2=dists{1}(1)*exp(cumsum(log([1,1./sqrt(1+2*sigma*stat.eta)]))); dist2(end)=[];
dists{2}=fillnan(dist2,maxiter(1));
dist3=dists{1}(1)*exp(cumsum(log([1,1./(1+sigma*stat.eta)]))); dist3(end)=[];
dists{3}=fillnan(dist3,maxiter(1));


%%% fista
fprintf('Executing FISTA to the precision obtained by DAL\n');
[xx,result_fista]=fistalrl1(A,yy,lmd,'tol',stat.fval(end),'w0',w0,'maxiter',maxiter(2));

niters(2)=length(result_fista.fval);
times{2} =result_fista.time;
fvals{2} =result_fista.fval-fval0;
dists{4} =result_fista.dist;

%%% OWLQN
fprintf('Executing OWLQN to the precision obtained by DAL\n');
[ww_owlqn,result_owlqn]=owlbfgs(@loss_lrp, zeros(n,1), A, yy, lmd,'display',2,'maxiter',maxiter(3),'w0',w0,'tol',stat.fval(end));


niters(3)=length(result_owlqn.fval);
times{3} =result_owlqn.time;
fvals{3} =result_owlqn.fval-fval0;
dists{5} =result_owlqn.dist;


%%% SpaRSA
fprintf('Executing SpaRSA to RDG<1e-7\n');
[xx,xx_deb,objs,time,deb_s,mses]=sparsalrl1(yy,A,lmd,'True_x',w0,'ToleranceA',1e-7,'MaxiterA',maxiter(4)-1,'BB_variant',1);

niters(4)=length(objs);
times{4} =fillnan(time,maxiter(4));
fvals{4} =fillnan(objs-fval0,maxiter(4));
dists{6} =fillnan(sqrt(mses*n),maxiter(4));

logs=struct('fval0',fval0,'gnorm',norm(gg0),'dal',stat,'fista',result_fista,'owlqn',result_owlqn);
