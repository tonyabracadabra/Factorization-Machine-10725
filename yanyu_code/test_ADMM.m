%% ADMM
% compare convergence rate of proximal gradient and coordinate descent
function [objs_coors,objs_proxs, objs_quasis] = test_ADMM(datatype, p)
opt_model2 = optimize_model2_v2_for_argmin_w0wW;
objs_coors = {};
objs_proxs = {};
objs_quasis = {};
for i = 1 : 3
    mymatname = ['../simulated_data/', datatype, '_p', num2str(p), '_', num2str(i)];
    load(mymatname);
    x = data{1};
    y = data{2};
    w0 = data{3};
    w = data{4};
    W = data{5};
    n = size(x, 1);
    fraction_train = 0.5;
    ntrain = ceil(n * fraction_train);
    xtrain = x(1 : ntrain, :);
    ytrain = y(1 : ntrain, :);
    xtest = x(ntrain + 1 : n,:);
    ytest = y(ntrain + 1 : n, :);
    lambda1 = 5;
    lambda2 = 5;
    lambda3 = 1;
    lambda4 = 1;
    w0_init = 0;
    w_init = zeros(p, 1);
    W_init = zeros(p, p); 
    U_init = zeros(p, p);
    u_init = W_init;
    tol = 1e-2;
    rho = 5;
    max_step = 200;
    addpath('OPT09/matlab');
    addpath('dal/matlab/');
    counter = [0, 0, 0];
    debug = 3;
    objs_coor = 0;
    objs_prox = 0;
    objs_quasi = 0;
    is_opjective = opt_model2.compute_obj(w0, w, W, W, u_init, rho, x, y, lambda1, lambda2, lambda3);
    [w0_prox, w_prox, W_prox, U_prox, u_prox, objs_prox, counter_prox] ...
        = opt_model2.admm(xtrain, ytrain, lambda1, lambda2, lambda3, w0_init, w_init, W_init, U_init, ...
            u_init, rho, max_step, tol, debug, 1); % for proximal
    [w0_coor, w_coor, W_coor, U_coor, u_coor, objs_coor, counter_coor] ...
        = opt_model2.admm(xtrain, ytrain, lambda1, lambda2, lambda3, w0_init, w_init, W_init, U_init, ...
            u_init, rho, max_step, tol, debug, 2); % for coord-proximal
    [w0_quasi, w_quasi, W_quasi, U_quasi, u_quasi, objs_quasi, counter_quasi,] ...
        = opt_model2.admm(xtrain, ytrain, lambda1, lambda2, lambda3, w0_init, w_init, W_init, U_init, ...
            u_init, rho, max_step, tol, debug, 3); % for coord-newton

% 
    objs_coors{i} = objs_coor;
    objs_proxs{i} = objs_prox;
    objs_quasis{i} = objs_quasi;
%     figure;
%     plot(objs_coor(:,2), log(objs_coor(:,1)), 'r');
%     hold on;
%     plot(objs_prox(:,2), log(objs_prox(:, 1)),'b');
%     hold on;
%     plot(objs_quasi(:,2), log(objs_quasi(:,1)),'g');
%     hold on;
%     h = legend('coordinate', 'proximal', 'quasi-newton');
%     set(h,'FontSize',20);
%     figure;
%     plot(objs_coor(:,2), objs_coor(:,3), 'r');
%     hold on;
%     plot(objs_prox(:,2), objs_prox(:, 3),'b');
%     hold on;
%     plot(objs_quasi(:,2), objs_quasi(:,3),'g');
%     hold on;
%     h = legend('coordinate', 'proximal', 'quasi-newton');
%     set(h,'FontSize',20);
    
end
end
%%
% figure;
% for i = 1 : 5
%     plot(log(curves_prox{i}), 'r');
%     axis([0, 600, 7, 12]);
%     hold on;
%     plot(log(curves_quasi{i}), 'b');
%     axis([0, 600, 7, 12]);
%     hold on;
% end
% h = legend('proximal', 'quasi-newton');
% set(h,'FontSize',20);
% ylabel('log(objective)', 'FontSize', 20);
% xlabel('# of iterations', 'FontSize', 20);
