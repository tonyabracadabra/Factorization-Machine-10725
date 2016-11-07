%% test model1
data_gen = data_generator;
opt_model1 = optimize_model1;
%% data generation
n = 500;
p = 10;
x = rand(n, p);
k = 3;
maxstep = 1000;
fraction_train = 0.5;
ntrain = ceil(n * fraction_train);
noise = 0.01;
scale_w0 = 1;
scale_w = 1;
scale_V = 1;
mask_w = ones(p, 1);
mask_w(10, 1) = 0;
mask_w(1, 1) = 0;
mask_W = ones(p, p);
[w0, w, V, W, y] = data_gen.v1(x, k, noise, scale_w0, scale_w, scale_V, mask_w, mask_W);
xtrain = x(1 : ntrain, :);
ytrain = y(1 : ntrain, :);
xtest = x(ntrain + 1 : n,:);
ytest = y(ntrain + 1 : n, :);
%% parameters 
lambda1 = 0;
lambda2 = 0;
w0_init = 0;
w_init = zeros(p, 1);
V_init = rand(k, p); % zeros(k, p); %

tol = 1e-3;
tmax = 10;
lina = 0.5;
linb = 0.5;
%% optimization - gradient descent
% [w0_grad, w_grad, V_grad, W_grad, objs_grad] = opt_model1.gradient_descent(xtrain, ytrain, lambda1, lambda2, w0_init, w_init, V_init, tol, tmax, lina, linb, maxstep);
% y_hat_grad_test = opt_model1.predict(xtest, w0_grad, w_grad, V_grad);
% y_hat_grad_train = opt_model1.predict(xtrain, w0_grad, w_grad, V_grad);
% 
% figure;
% scatter(y_hat_grad_test, ytest);hold on; h = refline(1, 0);h.Color = 'r';
% figure;
% scatter(y_hat_grad_train, ytrain);hold on; h = refline(1, 0);h.Color = 'r';
%% optimization - gradient descent no backtracking
% [w0_grad, w_grad, V_grad, W_grad, objs_grad] = opt_model1.gradient_descent_noback(xtrain, ytrain, lambda1, lambda2, w0_init, w_init, V_init, tol, tmax, lina, linb, maxstep);
% y_hat_grad_test = opt_model1.predict(xtest, w0_grad, w_grad, V_grad);
% y_hat_grad_train = opt_model1.predict(xtrain, w0_grad, w_grad, V_grad);
% 
% figure;
% scatter(y_hat_grad_test, ytest);hold on; h = refline(1, 0);h.Color = 'r';
% figure;
% scatter(y_hat_grad_train, ytrain);hold on; h = refline(1, 0);h.Color = 'r';
%% optimization - gradient descent acceleration
% tmax = 1e-10;
% [w0_grad_acce, w_grad_acce, V_grad_acce, W_grad_acce, objs_grad_acce] = opt_model1.gradient_descent_acce(xtrain, ytrain, lambda1, lambda2, w0_init, w_init, V_init, tol, tmax, lina, linb, maxstep);
% y_hat_grad_acce_test = opt_model1.predict(xtest, w0_grad_acce, w_grad_acce, V_grad_acce);
% y_hat_grad_acce_train = opt_model1.predict(xtrain, w0_grad_acce, w_grad_acce, V_grad_acce);
% 
% figure;
% scatter(y_hat_grad_acce_test, ytest);hold on; h = refline(1, 0);h.Color = 'r';
% figure;
% scatter(y_hat_grad_acce_train, ytrain);hold on; h = refline(1, 0);h.Color = 'r';
%% optimization - Newton's step
%% optimization - proximal descent
% lambda2 = 10;
% [w0_grad, w_grad, V_grad, W_grad, objs_grad] = opt_model1.proximal_descent(xtrain, ytrain, lambda1, lambda2, w0_init, w_init, V_init, tol, tmax, lina, linb, maxstep);
% y_hat_grad_test = opt_model1.predict(xtest, w0_grad, w_grad, V_grad);
% y_hat_grad_train = opt_model1.predict(xtrain, w0_grad, w_grad, V_grad);
% % y_hat_test = opt_model1.predict(xtest, w0, w, V);
% 
% figure;
% scatter(y_hat_grad_test, ytest);hold on; h = refline(1, 0);h.Color = 'r';
% figure;
% scatter(y_hat_grad_train, ytrain);hold on; h = refline(1, 0);h.Color = 'r';
%% optimization - proximal descent acceleration
lambda2 = 10;
tk = 1e-5;
[w0_grad, w_grad, V_grad, W_grad, objs_grad] = opt_model1.proximal_descent_acc(xtrain, ytrain, lambda1, lambda2, w0_init, w_init, V_init, tol, tk, lina, linb, maxstep);
y_hat_grad_test = opt_model1.predict(xtest, w0_grad, w_grad, V_grad);
y_hat_grad_train = opt_model1.predict(xtrain, w0_grad, w_grad, V_grad);
% y_hat_test = opt_model1.predict(xtest, w0, w, V);

figure;
scatter(y_hat_grad_test, ytest);hold on; h = refline(1, 0);h.Color = 'r';
figure;
scatter(y_hat_grad_train, ytrain);hold on; h = refline(1, 0);h.Color = 'r';