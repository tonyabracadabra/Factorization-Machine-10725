%% auther: Yanyu Liang
%% contact: yanyul@andrew.cmu.edu
%% time: 12/08/2016
%% project: Factorization Machines
%% Task: Model 2 - convex FMs - v2
%% Description: 
%  L = |y - \hat{y}|^2 + lambda1 * |W|_tr + lambda2 * |vec(W)|_1 + lambda3 * |w|_2^2
%  \hat{y} = w0 + w' * x + x' * W * x
%% Algorithms:
%  1. ADMM: |y - \hat{y}|^2 + lambda1 * |U|_tr + lambda2 * |vec(W)|_1 + lambda3 * |w|_2^2 + lambda4 * |w|_1
%    1.1 update W, w, w0:
%         arg min_W |y - \hat{y}|^2 + lambda2 * |vec(W)|_1 + lambda3 * |w|_2^2 + rho / 2 |W - U + u|_2^2 
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
function m2 = optimize_model2_v2_for_argmin_w0wW
    m2.admm = @m2_admm;
    m2.predict = @predict;
    m2.argmin_W = @argmin_W;
    m2.argmin_w0_w_W_proximal = @argmin_w0_w_W_proximal;
    m2.argmin_w0_w_W_coordinate = @argmin_w0_w_W_coordinate;
    m2.argmin_W_owlqn = @argmin_W_owlqn;
    m2.argmin_w0_w_W_quasinewton = @argmin_w0_w_W_quasinewton;
    m2.compute_obj = @compute_obj;
end 

function [w0, w, W, U, u, objs, counters] = m2_admm(x, y, lambda1, lambda2, lambda3, w0_init, w_init, W_init, U_init, u_init, rho, max_step, tol, debug, mode)
    w0 = w0_init;
    w = w_init;
    W = W_init;
    U = U_init;
    u = u_init;
    objs = zeros(max_step + 1, 3);
    counters = [];
    % imagesc(W);
    steps = 0;
    objs(1, 1) = original_obj(w0, w, W, lambda1, lambda2, lambda3, x, y); 
    objs(1, 2) = steps;
    objs(1, 3) = sum(sum((W - U) .^ 2)) / size(W, 1) ^ 2;
    for i = 1 : max_step
        counter = [0, 0, 0];
        if mode == 1
            [w0, w, W, counter] = argmin_w0_w_W_proximal(lambda2, lambda3, x, y, W, w, w0, U, u, rho, tol, debug, counter); % coded
            % imagesc(W);
            steps = steps + counter(1);
        elseif mode == 2
            [w0, w, W, counter] = argmin_w0_w_W_coordinate(lambda2, lambda3, x, y, W, w, w0, U, u, rho, tol, debug, counter);
            steps = steps + counter(1) + counter(2);
        else
            [w0, w, W, counter] = argmin_w0_w_W_quasinewton(lambda2, lambda3, x, y, W, w, w0, U, u, rho, tol, debug, counter);
            steps = steps + counter(1) + counter(2);
        end
        disp([num2str(compute_obj(w0, w, W, U, u, rho, x, y, lambda1, lambda2, lambda3)), ' update w0 w W']);
        [U, counter] = argmin_U(lambda1, W, u, rho, debug, counter); % coded
        disp([num2str(compute_obj(w0, w, W, U, u, rho, x, y, lambda1, lambda2, lambda3)), ' update U']);
        u = u + W - U;
        steps = steps + 2;
        disp([num2str(compute_obj(w0, w, W, U, u, rho, x, y, lambda1, lambda2, lambda3)), ' update u']);
        objs(i + 1, 1) = original_obj(w0, w, W, lambda1, lambda2, lambda3, x, y); 
        objs(i + 1, 2) = steps;
        objs(i + 1, 3) = sum(sum((W - U) .^ 2)) / size(W, 1) ^ 2;
        counters = [counters; counter];
    end
end

function L = original_obj(w0, w, W, lambda1, lambda2, lambda3, x, y)
    y_hat = w0 + x * w + diag(x * W * x');
    L = sum((y - y_hat) .^ 2) + lambda2 * sum(sum(abs(W))) + ...
        lambda3 * sum(w .^ 2) + sum(lambda1 * svd(W));
end

function [w0, w, W, counter, objs] = argmin_w0_w_W_proximal(lambda2, lambda3, x, y, W, w, w0, U, u, rho, tol, debug, counter)
    obj = compute_obj1(lambda2, lambda3, x, y, W, w, w0, U, u, rho);
    objs = [obj, 0];
    delta = tol + 1;
    if debug >= 1
        disp([num2str(obj), ' argmin_w0_w_W start']);
    end
    g = sum((y - w0 - x * w - diag(x * W * x')) .^ 2) + lambda3 * sum(w .^ 2) + rho / 2 * sum(sum((W - U + u) .^ 2));
    while delta > tol
        yhat = diag(x * W * x') + x * w + w0;
        gradW = 2 * x' * diag(yhat - y) * x + rho * (W - U + u);
        gradw = 2 * x' * (yhat - y) + 2 * lambda3 * w;
        gradw0 = 2 * sum(yhat - y);
        
        t = 1;
        Gt = (W - prox_ml1(W - t * gradW, t * lambda2)) / t;
        while 1
            temp_W = W - t * Gt;
            temp_w = w - t * gradw;
            temp_w0 = w0 - t * gradw0;
%             temp_y_hat_W = diag(x * temp_W * x');
%             temp_yw = y - temp_y_hat_W;
%             temp_y_hat_w = compute_y_hat_w(temp_w0, temp_w, x);
%             temp_yW = y - temp_y_hat_w;
            g_temp = sum((y - temp_w0 - x * temp_w - diag(x * temp_W * x')) .^ 2) + lambda3 * sum(temp_w .^ 2) + rho / 2 * sum(sum((temp_W - U + u) .^ 2));
            if g_temp <= g - t * sum(sum(gradW .* Gt)) + t / 2 * sum(sum(Gt .^ 2)) - t / 2 * (sum(gradw .^ 2) + gradw0 ^ 2)
                break
            end
            t = 0.5 * t;
            Gt = (W - prox_ml1(W - t * gradW, t * lambda2)) / t;
        end
        W = temp_W;
        w = temp_w;
        w0 = temp_w0;
        g = g_temp;
        obj_new = compute_obj1(lambda2, lambda3, x, y, W, w, w0, U, u, rho); % coded
        delta = obj - obj_new;
        obj = obj_new;
        if debug >= 1
            disp([num2str(obj), ' argmin_w0_w_W_proximal inside', ' t = ', num2str(t)]);
        end
        counter(1) = counter(1) + 1;
        objs = [objs; [obj, 0]];
    end
    if debug >= 1
        disp([num2str(obj), ' argmin_w0_w_W end']);
    end
end

function [U, counter] = argmin_U(lambda1, W, u, rho, debug, counter)
    if debug >= 1
        disp('argmin_U start');
    end
    U = prox_tr(W + u, lambda1 / rho);
    counter(3) = counter(3) + 1;
    if debug >= 1
        disp('argmin_U end');
    end
end

function U = prox_tr(X, t)
    [u,s,v] = svd(X);
    temp = max(diag(s) - t, 0);
    U = u * diag(temp) * v';
end

function [w0, w, W, counter, objs] = argmin_w0_w_W_coordinate(lambda2, lambda3, x, y, W, w, w0, U, u, rho, tol, debug, counter)
    % coordinate descent
%     inner_debug = 0;
    obj = compute_obj1(lambda2, lambda3, x, y, W, w, w0, U, u, rho);
    objs = [obj, 0];
    delta = tol + 1;
    if debug >= 1
        disp([num2str(obj), ' argmin_w0_w_W start']);
    end
    while delta > tol
        % update w, w0
        [w0, w, counter, objs] = argmin_w0_w_operator(lambda2, lambda3, x, y, W, w, w0, U, u, rho, tol, debug, counter, objs); % coded, checked
        % update W
        [W, counter, ~, objs] = argmin_W(lambda2, lambda3, x, y, W, w, w0, U, u, rho, tol, debug, counter, objs); % coded, checked
        obj_new = compute_obj1(lambda2, lambda3, x, y, W, w, w0, U, u, rho); % coded
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

function L = compute_obj1(lambda2, lambda3, x, y, W, w, w0, U, u, rho)
    y_hat = w0 + x * w + diag(x * W * x');
    L = sum((y - y_hat) .^ 2) + lambda2 * sum(sum(abs(W))) + lambda3 * sum(w .^ 2) + rho / 2 * sum(sum((W - U + u) .^ 2));
end

function L = compute_obj(w0, w, W, U, u, rho, x, y, lambda1, lambda2, lambda3)
    L = compute_obj1(lambda2, lambda3, x, y, W, w, w0, U, u, rho);
    L = L + sum(lambda1 * svd(U)) - rho / 2 * sum(sum((W - U + u) .^ 2));
end

function [W, counter, objs, objss] = argmin_W(lambda2, lambda3, x, y, W, w, w0, U, u, rho, tol, debug, counter, objss)
    tol = tol * 0.1;
    delta = tol + 1;
    y_hat = diag(x * W * x');
    y_ori = y;
    y = y - compute_y_hat_w(w0, w, x);
    g = compute_g_W(y, y_hat, W, U, u, rho);
    obj = g + lambda2 * sum(sum(abs(W)));
    if debug == 1
        disp(['argmin_W start ', num2str(obj)]);
    end
    
    objs = [];
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
        obj_new = g + lambda2 * sum(sum(abs(W)));
        delta = obj - obj_new;
        obj = obj_new;
        if debug == 1
            disp(['argmin_W inside ', num2str(obj), ' t = ', num2str(t)]);
        end
        counter(1) = counter(1) + 1;
        objs = [objs; obj];
        objss = [objss; [compute_obj1(lambda2, lambda3, x, y_ori, W, w, w0, U, u, rho), 1]];
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

function [w0, w, counter, objs] = argmin_w0_w_operator(lambda2, lambda3, x, y, W, w, w0, U, u, rho, tol, debug, counter, objs) % linesearch: linb = 0.5
    tol = tol * 0.01;
    delta = tol + 1;
    y_tilde = y - diag(x * W * x') - w0;
    obj = sum((y_tilde - x * w) .^ 2) + lambda3 * sum(w .^ 2);
    if debug == 1
        disp(['argmin_w0_w start ', num2str(obj)]);
    end
    while delta > tol
        
        right = x' * y_tilde;
        left = x' * x + lambda3 * eye(size(x, 2));
        w = linsolve(left, right);
        w0 = mean(y_tilde - x * w);
        y_tilde = y - diag(x * W * x') - w0;
        new_obj = sum((y_tilde - x * w) .^ 2) + lambda3 * sum(w .^ 2);
        delta = obj - new_obj;
        obj = new_obj;
        
        if debug == 1
            disp(['argmin_w0_w inside ', num2str(obj)]);
        end
        counter(2) = counter(2) + 1;
        objs = [objs; [compute_obj1(lambda2, lambda3, x, y, W, w, w0, U, u, rho), 2]];
    end
    if debug == 1
        disp(['argmin_w0_w end ', num2str(obj)]);
    end
end

function y_hat = compute_y_hat_w(w0, w, x)
    y_hat = w0 + x * w;
end
    
function yt = predict(x, w0, w, W)
    yt = w0 + x * w + diag(x * W * x');
end 

function [W, counter, objs] = argmin_W_owlqn(lambda2, lambda3, x, y, W, w, w0, U, u, rho, tol, debug, counter, objs) 
    y_hatt = diag(x * W * x');
    yt = y - compute_y_hat_w(w0, w, x);
    g = compute_g_W(yt, y_hatt, W, U, u, rho);
    obj = g + lambda2 * sum(sum(abs(W)));
    if debug == 1
        disp(['argmin_W start ', num2str(obj)]);
    end
    
    yhat = y - w0 - x * w;
    tempUu = U - u;
    [W, counter, objs, ~] = owlbfgs_for_argmin_w0wW(@compute_objW_grad, W, w0, w, U, u, x, y, yhat, lambda2, lambda3, tempUu, rho, counter, objs, 'display',1,'maxiter',0,'tol',tol*0.01, 'max_linesearch', 100);
    
    y_hatt = diag(x * W * x');
    yt = y - compute_y_hat_w(w0, w, x);
    g = compute_g_W(yt, y_hatt, W, U, u, rho);
    obj = g + lambda2 * sum(sum(abs(W)));
    if debug == 1
        disp(['argmin_W end ', num2str(obj)]);
    end
end

function [obj, grad] = compute_objW_grad(W, x, yhat, lambda2, tempUu, rho)
    diff_y = - yhat + diag(x * W * x');
    obj = sum(diff_y .^ 2) + rho / 2 * sum(sum((W - tempUu) .^ 2)) + lambda2 * sum(sum(abs(W)));
    grad = 2 * x' * diag(diff_y) * x + 2 * (W - tempUu) + lambda2 * sign(W);
    I1 = find(W == 0 & grad > 0);
    I2 = find(W == 0 & grad < 0);

    grad(I1) = grad(I1) + max(-lambda2, -grad(I1));
    grad(I2) = grad(I2) + min(lambda2, -grad(I2));
end

function [w0, w, W, counter, objs] = argmin_w0_w_W_quasinewton(lambda2, lambda3, x, y, W, w, w0, U, u, rho, tol, debug, counter)
        % coordinate descent
%     inner_debug = 0;
    obj = compute_obj1(lambda2, lambda3, x, y, W, w, w0, U, u, rho);
    objs = [obj, 0];
    delta = tol + 1;
    if debug >= 1
        disp([num2str(obj), ' argmin_w0_w_W start']);
    end
    inner_count = 1;
    while delta > tol % && inner_count <= 10
        % update w, w0
        [w0, w, counter, objs] = argmin_w0_w_operator(lambda2, lambda3, x, y, W, w, w0, U, u, rho, tol, debug, counter, objs); % coded, checked
        % update W
        [W, counter, objs] = argmin_W_owlqn(lambda2, lambda3, x, y, W, w, w0, U, u, rho, tol, debug, counter, objs); % coded, checked
        obj_new = compute_obj1(lambda2, lambda3, x, y, W, w, w0, U, u, rho); % coded
        delta = obj - obj_new;
        obj = obj_new;
        if debug >= 1
            disp([num2str(obj), ' argmin_w0_w_W inside']);
        end
        inner_count = inner_count + 1;
        
    end
    if debug >= 1
        disp([num2str(obj), ' argmin_w0_w_W end']);
    end
end