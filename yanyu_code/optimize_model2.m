%% auther: Yanyu Liang
%% contact: yanyul@andrew.cmu.edu
%% time: 12/04/2016
%% project: Factorization Machines
%% Task: Model 2 - convex FMs
%% Description: 
%  L = |y - \hat{y}|^2 + lambda1 * |W|_tr + lambda2 * |vec(W)|_1 + lambda3 * |w|_2^2 + lambda4 * |w|_1
%  \hat{y} = w0 + w' * x + x' * W * x
%% Algorithms:
%  1. ADMM: |y - \hat{y}|^2 + lambda1 * |U|_tr + lambda2 * |vec(W)|_1 + lambda3 * |w|_2^2 + lambda4 * |w|_1
%    1.1 update W:
%         arg min_W |y - \hat{y}|^2 + lambda2 * |vec(W)|_1 + lambda3 * |w|_2^2 + lambda4 * |w|_1 + rho / 2 |W - U + u|_2^2 
%    1.2 update U:
%         arg min_U lambda1 * |U|_tr + rho / 2 * |W + u - U|_2^2 (prox_{tr, lambda1 / rho}(W + u)
%    1.3 update u:
%         u + W - U
%% Limitation:
%  1. TBA
%% Memo:
%  1. gradiant w.r.t |y - \hat{y}|^2
%    grad w0 = 2(y_hat - y)
%    grad w = 2(y_hat - y) * x'
%    grad W = 2(y_hat - y) * X' * X
%% Code
function m2 = optimize_model2
    m2.admm = @m2_admm;
    m2.predict = @predict;
    m2.argmin_w0_w = @argmin_w0_w;
    m2.argmin_W = @argmin_W;
    m2.argmin_w0_w_coordinate = @argmin_w0_w_coordinate;
end 

function [w0, w, W, U, u, objs] = m2_admm(x, y, lambda1, lambda2, lambda3, lambda4, w0_init, w_init, W_init, U_init, u_init, rho, max_step, tol, debug)
    w0 = w0_init;
    w = w_init;
    W = W_init;
    U = U_init;
    u = u_init;
    objs = zeros(max_step, 1);
    for i = 1 : max_step
        [w0, w, W] = argmin_w0_w_W(lambda2, lambda3, lambda4, x, y, W, w, w0, U, u, rho, tol, debug); % coded
        disp([num2str(compute_obj(w0, w, W, U, u, rho, x, y, lambda1, lambda2, lambda3, lambda4)), ' update w0 w W']);
        U = argmin_U(lambda1, W, u, rho, debug); % coded
        disp([num2str(compute_obj(w0, w, W, U, u, rho, x, y, lambda1, lambda2, lambda3, lambda4)), ' update U']);
        u = u + W - U;
        disp([num2str(compute_obj(w0, w, W, U, u, rho, x, y, lambda1, lambda2, lambda3, lambda4)), ' update u']);
        objs(i) = compute_obj(w0, w, W, U, u, rho, x, y, lambda1, lambda2, lambda3, lambda4);
    end
end

function U = argmin_U(lambda1, W, u, rho, debug)
    if debug >= 1
        disp('argmin_U start');
    end
    U = prox_tr(W + u, lambda1 / rho);
    if debug >= 1
        disp('argmin_U end');
    end
end

function U = prox_tr(X, t)
    [u,s,v] = svd(X);
    temp = max(diag(s) - t, 0);
    U = u * diag(temp) * v';
end

function [w0, w, W] = argmin_w0_w_W(lambda2, lambda3, lambda4, x, y, W, w, w0, U, u, rho, tol, debug)
    % coordinate descent
%     inner_debug = 0;
    obj = compute_obj1(lambda2, lambda3, lambda4, x, y, W, w, w0, U, u, rho);
    delta = tol + 1;
    if debug >= 1
        disp([num2str(obj), ' argmin_w0_w_W start']);
    end
    while delta > tol
        % update w, w0
        [w0, w] = argmin_w0_w_coordinate(lambda3, lambda4, x, y, W, w, w0, tol, debug); % coded, checked
        % update W
        W = argmin_W(lambda2, x, y, W, w, w0, U, u, rho, tol, debug); % coded, checked
        obj_new = compute_obj1(lambda2, lambda3, lambda4, x, y, W, w, w0, U, u, rho); % coded
        delta = obj - obj_new;
        obj = obj_new;
        if debug >= 1
            disp([num2str(obj), ' argmin_w0_w_W inside']);
        end
    end
    if debug >= 1
        disp([num2str(obj), ' argmin_w0_w_W end']);
    end
end

function L = compute_obj1(lambda2, lambda3, lambda4, x, y, W, w, w0, U, u, rho)
    y_hat = w0 + x * w + diag(x * W * x');
    L = sum((y - y_hat) .^ 2) + lambda2 * sum(sum(abs(W))) + lambda3 * sum(w .^ 2) + lambda4 * sum(abs(w)) + rho / 2 * sum(sum((W - U + u) .^ 2));
end

function L = compute_obj(w0, w, W, U, u, rho, x, y, lambda1, lambda2, lambda3, lambda4)
    L = compute_obj1(lambda2, lambda3, lambda4, x, y, W, w, w0, U, u, rho);
    L = L + sum(lambda1 * svd(U));
end

function W = argmin_W(lambda2, x, y, W, w, w0, U, u, rho, tol, debug)
    delta = tol + 1;
    y_hat = diag(x * W * x');
    y = y - compute_y_hat_w(w0, w, x);
    g = compute_g_W(y, y_hat, W, U, u, rho);
    obj = g + sum(sum(abs(W)));
    if debug == 1
        disp(['argmin_W start ', num2str(obj)]);
    end
    while delta > tol
        grad_W = 2 * x' * diag(y_hat - y) * x + rho * (W - U + u);
        t = 1;
        Gt = (W - prox_ml1(W - t * grad_W, t * lambda2)) / t;
        while 1
            temp_W = W - t * Gt;
            temp_y_hat = diag(x * temp_W * x');
            g_temp = compute_g_W(y, temp_y_hat, temp_W, U, u, rho);
            if g_temp <= g - t * sum(sum(grad_W .* Gt)) + t / 2 * sum(sum(Gt .^ 2))
                break
            end
            t = 0.5 * t;
            Gt = (W - prox_ml1(W - t * grad_W, t * lambda2)) / t;
        end
        W = temp_W;
        y_hat = temp_y_hat;
        g = g_temp;
        obj_new = g + sum(sum(abs(W)));
        delta = obj - obj_new;
        obj = obj_new;
        if debug == 1
            disp(['argmin_W inside ', num2str(obj), ' t = ', num2str(t)]);
        end
    end
    if debug == 1
        disp(['argmin_W end ', num2str(obj)]);
    end
end  

function g = compute_g_W(y, y_hat, W, U, u, rho)
    g = sum((y - y_hat) .^ 2) + rho / 2 * sum(sum((W - U + u) .^ 2));
end

function Z = prox_ml1(X, t)
    Z = zeros(size(X));
    bigger = X > t;
    Z(bigger) = X(bigger) - t;
    smaller = X < -t;
    Z(smaller) = X(smaller) + t;
end 

function [w0, w] = argmin_w0_w(lambda3, lambda4, x, y, W, w, w0, tol, debug) % linesearch: linb = 0.5
    delta = tol + 1;
    y = y - diag(x * W * x');
    y_hat = compute_y_hat_w(w0, w, x);
    g = compute_g_w(y, y_hat, lambda3, w);% sum((temp - y) .^ 2) + lambda3 * sum(w .^ 2);
    obj = g + sum(abs(w)) * lambda4;
    if debug == 1
        disp(['argmin_w0_w start ', num2str(obj)]);
    end
    while delta > tol       
        grad_w0 = 2 * sum(y_hat - y);
        grad_w = 2 * x' * (y_hat - y) + 2 * lambda3 * w;
        t = 1;
        Gt = (w - prox_l1(w - t * grad_w, t * lambda4)) / t;      
        while 1
            temp_w0 = w0 - t * grad_w0;
            temp_w = w - t * Gt;
            temp_y_hat = compute_y_hat_w(temp_w0, temp_w, x);
            g_temp = compute_g_w(y, temp_y_hat, lambda3, temp_w);
            if g_temp <= g - t * (grad_w' * Gt + grad_w0 ^ 2) + t / 2 * (sum(Gt .^ 2) + grad_w0 ^ 2)
                break
            end
            t = 0.5 * t;
            Gt = (w - prox_l1(w - t * grad_w, t * lambda4)) / t;    
        end
        w0 = temp_w0;
        w = temp_w;
        y_hat = temp_y_hat;
        g = g_temp;
        new_obj = g + sum(abs(w)) * lambda4;
        delta = obj - new_obj;
        obj = new_obj;
        if debug == 1
            disp(['argmin_w0_w inside ', num2str(obj), ' t = ', num2str(t)]);
        end
    end
    if debug == 1
        disp(['argmin_w0_w end ', num2str(obj)]);
    end
end

function y_hat = compute_y_hat_w(w0, w, x)
    y_hat = w0 + x * w;
end

function g = compute_g_w(y, y_hat, lambda3, w)
    g = sum((y - y_hat) .^ 2) + lambda3 * sum(w .^ 2);
end

function z = prox_l1(x, t)
    z = zeros(size(x));
    bigger = x > t;
    z(bigger) = x(bigger) - t;
    smaller = x < -t;
    z(smaller) = x(smaller) + t;
end
    
function yt = predict(x, w0, w, W)
    yt = w0 + x * w + diag(x * W * x');
end

function [w0, w] = argmin_w0_w_coordinate(lambda3, lambda4, x, y, W, w, w0, tol, debug)
    delta = tol + 1;
    y = y - diag(x * W * x');
    y_hat = compute_y_hat_w(w0, w, x);
    g = compute_g_w(y, y_hat, lambda3, w);% sum((temp - y) .^ 2) + lambda3 * sum(w .^ 2);
    obj = g + sum(abs(w)) * lambda4;
    if debug == 1
        disp(['argmin_w0_w_coordinate start ', num2str(obj)]);
    end
    while delta > tol
        % update w0
        y_hat_hat = y - x * w;
        w0 = sum(y_hat_hat) / size(y_hat_hat, 1);
        for i = 1 : size(w, 1)
            y_hat_hat = y - w0 - x * w + x(:, i) * w(i);
            condition = 2 * x(:, i)' * y_hat_hat / lambda4;
            if condition >= -1 && condition <= 1
                w(i) = 0;
            elseif condition < -1
                w(i) = (condition * lambda4 + lambda4) / (2 * sum(x(:, i) .^ 2) + 2 * lambda3);
            else
                w(i) = (condition * lambda4 - lambda4) / (2 * sum(x(:, i) .^ 2) + 2 * lambda3);
            end
        end
        y_hat = compute_y_hat_w(w0, w, x);
        g = compute_g_w(y, y_hat, lambda3, w);% sum((temp - y) .^ 2) + lambda3 * sum(w .^ 2);
        new_obj = g + sum(abs(w)) * lambda4;
        delta = obj - new_obj;
        obj = new_obj;
        if debug == 1
            disp(['argmin_w0_w_coordinate inside ', num2str(obj)]);
        end
    end
    if debug == 1
        disp(['argmin_w0_w_coordinate end ', num2str(obj)]);
    end
end