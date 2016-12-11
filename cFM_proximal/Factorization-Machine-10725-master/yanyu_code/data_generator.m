%% auther: Yanyu Liang
%% contact: yanyul@andrew.cmu.edu
%% time: 11/05/2016
%% project: Factorization Machines
%% Task: data generator
%% Description: 
%  regression problem: input x, output y
%  underlying model: y = w0 + w' * x + x' * (V' * V) * x + noise
%  simulated data generation:
%    step 1: randomly generate w0, w, V
%    step 2: randomly generate x
%    step 3: generate y with white noise
%% I/O
%  input:
%    x - input data (for factorization, it should be sparse and its 
%        ditribution is specified by user, continuous or binary)
%    k - the length of vector in V (for factorization, column vector in V 
%        encode the map: feature_value -> vector in R^k space)
%    noise - the scale of noise (here we use N(0, noise))
%    scale_w0, scale_w, scale_V - the scale of w0, w, V
%    mask_w, mask_W - sparsity on w and W = V'V
%  output:
%    w0
%    w
%    V
%    y
%% Limitation:
%  1. have not included structure information of V, namely some features
%     should have higher inner product that others namely encoding they
%     behave similar to each other
%  2. TBA
%% Code
function gen = data_generator
    gen.v1 = @v1;
end

function [w0, w, V, W, y] = v1(x, k, noise, scale_w0, scale_w, scale_V, mask_w, mask_W)
    p = size(x, 2);
    n = size(x, 1);
    w0 = rand * scale_w0;
    w = rand(p, 1) * scale_w;
    V = rand(k, p) * scale_V;
    w = w .* mask_w;
    W = (V' * V) .* mask_W;
    y = w0 + x * w + diag(x * W * x');
    y = y + normrnd(0, noise, n, 1);
end
    
    