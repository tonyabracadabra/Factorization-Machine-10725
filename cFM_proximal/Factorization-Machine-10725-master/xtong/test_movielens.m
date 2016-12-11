%% parameters 

X_train = X_train(1:10000,:);
y_train = y_train(1:10000);
[n, p] = size(X_train);

lambda1 = 0;
lambda2 = 0;
w0_init = 0;
w_init = zeros(p, 1);
V_init = rand(k, p); % zeros(k, p); %

tol = 1e-3;
tmax = 10;
lina = 0.5;
linb = 0.5;

%% optimization - proximal descent acceleration
lambda2 = 10;
tk = 1e-5;
[w0_grad, w_grad, V_grad, W_grad, objs_grad] = opt_model1.proximal_descent_acc(X_train, y_train', lambda1, lambda2, w0_init, w_init, V_init, tol, tk, lina, linb, maxstep);
y_hat_grad_test = opt_model1.predict(X_test, w0_grad, w_grad, V_grad);
y_hat_grad_train = opt_model1.predict(X_train, w0_grad, w_grad, V_grad);
% y_hat_test = opt_model1.predict(xtest, w0, w, V);

figure;
scatter(y_hat_grad_test, y_test);hold on; h = refline(1, 0);h.Color = 'r';
figure;
scatter(y_hat_grad_train, y_train);hold on; h = refline(1, 0);h.Color = 'r';