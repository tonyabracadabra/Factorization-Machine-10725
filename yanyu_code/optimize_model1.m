%% auther: Yanyu Liang
%% contact: yanyul@andrew.cmu.edu
%% time: 11/05/2016
%% project: Factorization Machines
%% Task: Model 1 - classic FMs
%% Description: 
%  L = |y - \hat{y}|^2 + lambda1 * |w|_2^2 + lambda2 * |w|_1
%  \hat{y} = w0 + w' * x + x * W * x'
%% Algorithms:
%  1. gradient descent
%  2. proximal descent
%  3. Newton's method
%  4. primal-dual method
%  5. Quasi-Newton method
%% Limitation:
%  1. TBA
%% Memo:
%  1. gradiant w.r.t |y - \hat{y}|^2
%    grad w0 = 2(y_hat - y)
%    grad w = 2(y_hat - y) * x'
%    grad V_j = 2(y_hat - y) * (2 * V * x' * x)
%  2. gradiant w.r.t lambda1 * |w|_2^2
%    grad w = 2 * lambda1 * w
%% Code
function m1 = optimize_model1
    m1.gradient_descent = @m1_gradient_descent;
    m1.predict = @c_y_hat;
    m1.gradient_descent_acce = @m1_gradient_descent_acc;
    m1.gradient_descent_noback = @m1_gradient_descent_noback;
    m1.proximal_descent = @m1_proximal_descent;
    m1.proximal_descent_acc = @m1_proximal_descent_acc;
end

function [w0, w, V, W, objs] = m1_gradient_descent(x, y, lambda1, lambda2, w0_init, w_init, V_init, tol, tmax, lina, linb, maxstep)
    w0 = w0_init;
    w = w_init;
    V = V_init;
    
    y_hat = c_y_hat(x, w0, w, V);
    
    obj_pre = squared_error_loss(y, y_hat) + l2(w, lambda1);
    objs = [obj_pre, 1];
    delta = tol + 1;
    
    step_counter = 0;
    inner_counter = 1;
    
    while delta > tol && step_counter < maxstep
        grad_w0 = -c_grad_w0(y, y_hat);
        grad_w = -c_grad_w(y, y_hat, x);
        grad_V = -c_grad_V(y, y_hat, x, V);
        v_length = vlength(grad_w0, grad_w, grad_V);
        inner_counter = inner_counter + 1;
        
        t = tmax;
        while 1
            inner_counter = inner_counter + 1;
            w0_temp = w0 + t * grad_w0;
            w_temp = w + t * grad_w;
            V_temp = V + t * grad_V;
            y_hat_temp = c_y_hat(x, w0_temp, w_temp, V_temp);
            f_temp = squared_error_loss(y, y_hat_temp) + l2(w_temp, lambda1);
            if f_temp <= obj_pre - lina * t * v_length;
                break
            end
            t = t * linb;
        end
        w0 = w0_temp;
        w = w_temp;
        V = V_temp;
        y_hat = y_hat_temp;
        objs = [objs; f_temp, inner_counter];
        delta = obj_pre - f_temp;
        obj_pre = f_temp; 
        disp(obj_pre);
        step_counter = step_counter + 1;
    end
    W = V' * V;
end

function [w0, w, V, W, objs] = m1_gradient_descent_acc(x, y, lambda1, lambda2, w0_init, w_init, V_init, tol, tk, lina, linb, maxstep)
    w0 = w0_init;
    w = w_init;
    V = V_init;
    
    w0_km2 = w0;
    w_km2 = w;
    V_km2 = V;
    
    y_hat = c_y_hat(x, w0, w, V);
    objs = squared_error_loss(y, y_hat) + l2(w, lambda1);
    step_counter = 0;
    
    while step_counter < maxstep
        k = step_counter + 1;
        inter_v_w0 = w0 + (k - 2) / (k + 1) * (w0 - w0_km2);
        inter_v_w = w + (k - 2) / (k + 1) * (w - w_km2);
        inter_v_V = V + (k - 2) / (k + 1) * (V - V_km2);
        
        grad_w0 = -c_grad_w0(y, y_hat);
        grad_w = -c_grad_w(y, y_hat, x);
        grad_V = -c_grad_V(y, y_hat, x, inter_v_V);
        
        w0_km2 = w0;
        w_km2 = w;
        V_km2 = V;
        
        w0 = inter_v_w0 + tk * grad_w0;
        w = inter_v_w + tk * grad_w;
        V = inter_v_V + tk * grad_V;
        y_hat = c_y_hat(x, w0, w, V);
        f_temp = squared_error_loss(y, y_hat) + l2(w, lambda1);
        objs = [objs; f_temp];
        disp(f_temp);
        step_counter = step_counter + 1;
    end
    W = V' * V;
end

function [w0, w, V, W, objs] = m1_gradient_descent_noback(x, y, lambda1, lambda2, w0_init, w_init, V_init, tol, t, lina, linb, maxstep)
    w0 = w0_init;
    w = w_init;
    V = V_init;
    
    y_hat = c_y_hat(x, w0, w, V);
    
    objs = squared_error_loss(y, y_hat) + l2(w, lambda1);
    obj_pre = objs;
    delta = tol + 1;
    
    step_counter = 0;
    
    while delta > tol && step_counter < maxstep
        grad_w0 = -c_grad_w0(y, y_hat);
        grad_w = -c_grad_w(y, y_hat, x);
        grad_V = -c_grad_V(y, y_hat, x, V);
        
        w0_temp = w0 + t * grad_w0;
        w_temp = w + t * grad_w;
        V_temp = V + t * grad_V;
        y_hat_temp = c_y_hat(x, w0_temp, w_temp, V_temp);
        f_temp = squared_error_loss(y, y_hat_temp) + l2(w_temp, lambda1);
        
        y_hat = y_hat_temp;
        objs = [objs; f_temp];
        delta = obj_pre - f_temp;
        obj_pre = f_temp; 
        disp(t);
        step_counter = step_counter + 1;
    end
    W = V' * V;
end

function [w0, w, V, W, objs] = m1_proximal_descent(x, y, lambda1, lambda2, w0_init, w_init, V_init, tol, tmax, lina, linb, maxstep)
    w0 = w0_init;
    w = w_init;
    V = V_init;
    
    y_hat = c_y_hat(x, w0, w, V);
    
    obj_pre = squared_error_loss(y, y_hat) + l2(w, lambda1) + l1(w, lambda2);
    objs = [obj_pre, 1];
    delta = tol + 1;
    
    step_counter = 0;
    inner_counter = 1;
    
    while delta > tol && step_counter < maxstep
        t = tmax;
        
        grad_w0 = -c_grad_w0(y, y_hat);
        grad_w = -c_grad_w(y, y_hat, x);
        grad_V = -c_grad_V(y, y_hat, x, V);
        
        inner_counter = inner_counter + 1;
        g = squared_error_loss(y, y_hat) + l2(w, lambda1);
        
        while 1
            inner_counter = inner_counter + 1;
            Gt_w = (w - prox_l1(t * lambda2, w + t * grad_w)) / t;
            v_length = vlength(grad_w0, Gt_w, grad_V);
            w0_temp = w0 + t * grad_w0;
            w_temp = w - t * Gt_w;
            V_temp = V + t * grad_V;
            y_hat_temp = c_y_hat(x, w0_temp, w_temp, V_temp);
            g_temp = squared_error_loss(y, y_hat_temp) + l2(w_temp, lambda1);
            if g_temp <= g - lina * t * v_length;
                break
            end
            t = t * linb;
        end
        w0 = w0_temp;
        w = w_temp;
        V = V_temp;
        y_hat = y_hat_temp;
        f_temp = squared_error_loss(y, y_hat_temp) + l2(w_temp, lambda1) + l1(w_temp, lambda2);
        objs = [objs; f_temp, inner_counter];
        delta = obj_pre - f_temp;
        obj_pre = f_temp; 
        disp(obj_pre);
        step_counter = step_counter + 1;
%         disp(t);
    end
    W = V' * V;
end

function [w0, w, V, W, objs] = m1_proximal_descent_acc(x, y, lambda1, lambda2, w0_init, w_init, V_init, tol, tk, lina, linb, maxstep)
    w0 = w0_init;
    w = w_init;
    V = V_init;
    
    w0_km2 = w0;
    w_km2 = w;
    V_km2 = V;
    
    y_hat = c_y_hat(x, w0, w, V);
    
    obj_pre = squared_error_loss(y, y_hat) + l2(w, lambda1) + l1(w, lambda2);
    objs = [obj_pre, 1];
%     delta = tol + 1;
    
    step_counter = 0;
    inner_counter = 1;
    
    while step_counter < maxstep
        
        k = step_counter + 1;
        inter_v_w0 = w0 + (k - 2) / (k + 1) * (w0 - w0_km2);
        inter_v_w = w + (k - 2) / (k + 1) * (w - w_km2);
        inter_v_V = V + (k - 2) / (k + 1) * (V - V_km2);
        
        grad_w0 = -c_grad_w0(y, y_hat);
        grad_w = -c_grad_w(y, y_hat, x);
        grad_V = -c_grad_V(y, y_hat, x, inter_v_V);
        
        w0_km2 = w0;
        w_km2 = w;
        V_km2 = V;
        
        w0 = inter_v_w0 + tk * grad_w0;
        w = prox_l1(tk * lambda2, inter_v_w + tk * grad_w);
        V = inter_v_V + tk * grad_V;
        y_hat = c_y_hat(x, w0, w, V);
        f_temp = squared_error_loss(y, y_hat) + l2(w, lambda1) + l1(w, lambda2);
        objs = [objs; f_temp, inner_counter];
%         delta = obj_pre - f_temp;
%         obj_pre = f_temp; 
        disp(f_temp);
        step_counter = step_counter + 1;
    end
    W = V' * V;
end

function prox = prox_l1(t, x)
    temp1 = x > t;
    temp2 = x < -t;
    prox = zeros(size(x));
    prox(temp1) = x(temp1) - t;
    prox(temp2) = t - x(temp2);
end
    

function l = squared_error_loss(y, y_hat) 
    l = sum((y - y_hat) .^ 2);
end

function y_hat = c_y_hat(x, w0, w, V)
    temp = x * V';
    y_hat = w0 + x * w + diag(temp * temp');
end

function l = l2(w, lambda)
    l = w' * w * lambda;
end

function l = l1(w, lambda)
    l = sum(abs(w)) * lambda;
end

function grad_w0 = c_grad_w0(y, y_hat)
    grad_w0 = 2 * sum(y_hat - y);
end

function grad_w = c_grad_w(y, y_hat, x)
    temp = 2 * x' * (y_hat - y);
    grad_w = sum(temp, 2);
end

function grad_V = c_grad_V(y, y_hat, x, V)
    temp1 = diag(2 * (y_hat - y));
    grad_V = 2 * (V * (x')) * (temp1 * x);
end

function l_v = vlength(grad_w0, grad_w, grad_V)
    l_v = grad_w0 ^ 2 + sum(grad_w .^ 2) + sum(sum(grad_V .^ 2));
end

% function y_hat = predict(

 
