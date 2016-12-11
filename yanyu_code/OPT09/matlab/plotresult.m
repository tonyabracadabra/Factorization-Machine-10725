function plotresult(niters, times, fvals, dists)

subplot(2,2,1);
plot(1:niters(1), cell2mat(dists(1:3)')', '-o',...
     1:niters(2), dists{4}, 'c--^',...
     1:niters(3), dists{5}, 'm-.x',...
     1:niters(4), dists{6}, 'y-+', 'linewidth',2) 

set(gca,'yscale','log','xscale','log','fontsize',12);
grid on;
xlim([1 10000]); ylim([1e-5 10]);
set(gca,'ytick',logspace(-4,0,3));
legend('DAL','DAL (thm2)','DAL (thm1)','FISTA','OWLQN','SpaRSA');
ylabel('||w^t - w*||');
flipplotorder(gca);
h=get(gca,'children');
set(h(end), 'color',[.75 .75 0]);

subplot(2,2,2);
plot(times{1}, dists{1}, '-o',...
     times{2}, dists{4}, 'c--^',...
     times{3}, dists{5}, 'm-.x',...
     times{4}, dists{6}, 'y-+', 'linewidth', 2);

set(gca,'yscale','log','xscale','log','fontsize',12);
grid on;
xlim([1 200]); ylim([1e-5 10]);
set(gca,'ytick',logspace(-4,0,3));
flipplotorder(gca);
h=get(gca,'children');
set(h(end), 'color',[.75 .75 0]);


subplot(2,2,3);
plot(1:niters(1), fvals{1}, '-o',...
     1:niters(2), fvals{2}, 'c--^',...
     1:niters(3), fvals{3}, 'm-.x',...
     1:niters(4), fvals{4}, 'y-+', 'linewidth',2) 

set(gca,'yscale','log','xscale','log','fontsize',12);
grid on;
xlim([1 10000]); ylim([1e-9 2e+2]);
set(gca,'ytick',logspace(-8,2,6));
xlabel('#iteretions');
ylabel('f(w^t) - f(w*)');
flipplotorder(gca);
h=get(gca,'children');
set(h(end), 'color',[.75 .75 0]);

subplot(2,2,4);
plot(times{1}, fvals{1}, '-o',...
     times{2}, fvals{2}, 'c--^',...
     times{3}, fvals{3},'m-.x',...
     times{4}, fvals{4},'y-+', 'linewidth', 2);
set(gca,'yscale','log','xscale','log','fontsize',12);
grid on;
xlim([1 200]);  ylim([1e-9 2e+2]);
set(gca,'ytick',logspace(-8,2,6));
xlabel('CPU time (sec)');
legend('DAL','FISTA','OWLQN','SpaRSA');
flipplotorder(gca);
h=get(gca,'children');
set(h(end), 'color',[.75 .75 0]);
