%% test model1
data_gen = data_generator;
opt_model2 = optimize_model2;
%% data generation
n = 10;
p = 100;
x = rand(n, p);
k = 5;
max_step = 100;
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
mask_W(16:20,1:5) = 0;
mask_W(1:5,16:20) = 0;
[w0, w, V, W, y] = data_gen.v1(x, k, noise, scale_w0, scale_w, scale_V, mask_w, mask_W);
xtrain = x(1 : ntrain, :);
ytrain = y(1 : ntrain, :);
xtest = x(ntrain + 1 : n,:);
ytest = y(ntrain + 1 : n, :);
%% parameters 
lambda1 = 1;
lambda2 = 1;
lambda3 = 1;
lambda4 = 1;
w0_init = 0;
w_init = zeros(p, 1);
W_init = zeros(p, p); 
U_init = zeros(p, p);
u_init = W_init;
tol = 1e-2;
rho = 5;
%% optimization - gradient descent
debug = 1;
[w0_admm, w_admm, W_admm, U_admm, u_admm, objs_admm] = opt_model2.admm(xtrain, ytrain, lambda1, lambda2, lambda3, lambda4, w0_init, w_init, W_init, U_init, u_init, rho, max_step, tol, debug);


%% debugging
% w = rand(p, 1) * 100;
% w(2:5, 1) = 0;
% debug = 1;
% % temp_W = rand(k, p) * 2;
% % W = temp_W' * temp_W;
% xtrain = rand(n, p);
% ytrain = w0 + xtrain * w + diag(xtrain * W * xtrain') + rand(n, 1);
% xtest = rand(n, p);
% ytest = w0 + xtest * w + diag(xtest * W * xtest') + rand(n, 1);
% % W_admm = opt_model2.argmin_W(lambda2, xtrain, ytrain, W_init, w, w0, U_init, u_init, rho, tol, debug);
% [w0_admm, w_admm] = opt_model2.argmin_w0_w_coordinate(lambda3, lambda4, xtrain, ytrain, W_init, w_init, w0_init, tol, debug);
% W_admm = W_init;
y_hat_grad_test = opt_model2.predict(xtest, w0_admm, w_admm, W_admm);
y_hat_grad_train = opt_model2.predict(xtrain, w0_admm, w_admm, W_admm);
% 
figure;
scatter(y_hat_grad_test, ytest);hold on; h = refline(1, 0);h.Color = 'r';
figure;
scatter(y_hat_grad_train, ytrain);hold on; h = refline(1, 0);h.Color = 'r';
