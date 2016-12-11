%% dependencies
addpath(genpath('../cFM_proximal/'));
addpath(genpath('../yanyu_code'));
%% data - script_w_sparse_W_block & p = 100
datatype = 'script_w_sparse_W_block';
nmethods = 3;
p = 100;
% test_mse = zeros(5,nmethods);
% test_var_ratio_to_truth = zeros(5, methods);
% train_mse = zeros(5,nmethods);
% train_var_ratio_to_truth = zeros(5, methods);
numtrain = 400;
fid = fopen(['../results/', datatype, '_p', num2str(p), '.log'], 'w+');
fprintf(fid, '%s, %s, %s, %s, %s\n', 'method', 'train_mse', ...
    'train_var_ratio_to_truth', 'test_mse', 'test_var_ratio_to_truth');
for i = 1 : 5
    mymatname = ['../simulated_data/', datatype, '_p', num2str(p), '_', num2str(i)];
    load(mymatname);
    x = data{1};
    y = data{2};
    xtrain = x(1 : numtrain, :);
    ytrain = y(1 : numtrain, :);
    xtest = x(numtrain + 1 : size(x, 1), :);
    ytest = y(numtrain + 1 : size(y, 1), :);
    w0 = data{3};
    w = data{4};
    W = data{5};
    ytruetest = w0 + xtest * w + diag(xtest * W * xtest');
    vartest = sum((ytruetest - ytest) .^ 2) / size(ytest, 1);
    ytruetrain = w0 + xtrain * w + diag(xtrain * W * xtrain');
    vartrain = sum((ytruetrain - ytrain) .^ 2) / size(ytrain, 1);

    % run proximal descent
    %  L = |y - \hat{y}|^2 + lambda1 * |w|_2^2 + lambda2 * |w|_1 + lambda3 * |vec(V)|_2^2
    %  \hat{y} = w0 + w' * x + x' * V' * V * x
    opt_model1 = optimize_model1;
    lambda1 = 1;
    lambda2 = 10;
    lambda3 = 1;
    k = ceil(p * (0.2 + rand * 0.2));
    w0_init = 0;
    w_init = zeros(p, 1);
    V_init = rand(k, p); % zeros(k, p); %
    maxstep = 20 * 301;
    tol = 1e-3;
    tmax = 10;
    lina = 0.5;
    linb = 0.5;
    [w0_grad, w_grad, V_grad, W_grad, objs_grad] = ...
        opt_model1.proximal_descent(xtrain, ytrain, lambda1, ...
        lambda2, lambda3, w0_init, w_init, V_init, tol, tmax, ...
        lina, linb, maxstep);
    y_hat_grad_test = opt_model1.predict(xtest, w0_grad, w_grad, V_grad);
    y_hat_grad_train = opt_model1.predict(xtrain, w0_grad, w_grad, V_grad);
    [trainmse, trainvarratio] = give_me_measure(y_hat_grad_train, ytrain, vartrain);
    [testmse, testvarratio] = give_me_measure(y_hat_grad_test, ytest, vartest);
%     test_mse(i, 1) = testmse;
%     train_mse(i, 1) = trainmse;
%     test_var_ratio_to_truth(i, 1) = testvarratio;
%     train_var_ratio_to_truth(i, 1) = trainvarratio;
    fid = fopen(['../results/', datatype, '_p', num2str(p), '.log'], 'a');
    fprintf(fid, '%s, %d, %d, %d, %d\n', 'FM', trainmse, ...
    trainvarratio, testmse, testvarratio);
    fclose(fid);  
    % END
    
    % cFM
    epsilon = 1e-3;
    alpha = 1;
    beta = 10;
    [ff, w, Z] = cFM_prox_initial_with_zeros(xtrain, ytrain, alpha, beta, epsilon);
    y_hat_grad_test = xtest * w + diag(xtest * Z * xtest');
    y_hat_grad_train = xtrain * w + diag(xtrain * Z * xtrain');
    [trainmse, trainvarratio] = give_me_measure(y_hat_grad_train, ytrain, vartrain);
    [testmse, testvarratio] = give_me_measure(y_hat_grad_test, ytest, vartest);
    fid = fopen(['../results/', datatype, '_p', num2str(p), '.log'], 'a');
    fprintf(fid, '%s, %d, %d, %d, %d\n', 'cFM_orig', trainmse, ...
    trainvarratio, testmse, testvarratio);
    fclose(fid);  
    % END
    
    % cFM with sparsity 
    % L = |y - \hat{y}|^2 + lambda1 * |W|_tr + lambda2 * |vec(W)|_1 + lambda3 * |w|_2^2
    opt_model2 = optimize_model2_v2;
    lambda1 = 10;
    lambda2 = 10;
    lambda3 = 1;
    w0_init = 0;
    w_init = zeros(p, 1);
    W_init = zeros(p, p); 
    U_init = zeros(p, p);
    u_init = W_init;
    tol = 1e-2;
    rho = 5;
    debug = 1;
    max_step = 20; % ADMM iterations
    mode = 3; % coordinate descent with owl-qn
    [w0_admm_prox, w_admm_prox, W_admm_prox, U_admm_prox, u_admm_prox, objs_admm_prox, counter_admm_prox] ...
        = opt_model2.admm(xtrain, ytrain, lambda1, lambda2, lambda3, w0_init, w_init, W_init, U_init, u_init, rho, max_step, tol, debug, mode);
    y_hat_grad_test = opt_model2.predict(xtest, w0_admm_prox, w_admm_prox, W_admm_prox);
    y_hat_grad_train = opt_model2.predict(xtrain, w0_admm_prox, w_admm_prox, W_admm_prox);
    [trainmse, trainvarratio] = give_me_measure(y_hat_grad_train, ytrain, vartrain);
    [testmse, testvarratio] = give_me_measure(y_hat_grad_test, ytest, vartest);
    fid = fopen(['../results/', datatype, '_p', num2str(p), '.log'], 'a');
    fprintf(fid, '%s, %d, %d, %d, %d\n', 'cFM_sparse', trainmse, ...
    trainvarratio, testmse, testvarratio);
    fclose(fid);  
end
    
    