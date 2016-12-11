addpath ../dal/

m=1024;
n=16384;
lmd=1;
nrep=1;

maxiter=[20, 1000, 500, 500];

niters=zeros(nrep,length(maxiter));
times =cell(nrep,length(maxiter));
fvals =cell(nrep,length(maxiter));
dists =cell(nrep,length(maxiter)+2);
logs  =repmat(struct('fval0',[],'gnorm',[],'dal',[],'fista',[],'owlqn',[]),[nrep,1]);

for ii=1:nrep
  [niters(ii,:), times(ii,:), fvals(ii,:), dists(ii,:), logs(ii)]=run_exp(m,n,lmd,maxiter);
end

mtime=meancell(times);
mfval=meancell(fvals);
mdist=meancell(dists);

plotresult(maxiter, mtime, mfval, mdist);
