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
    gen.w_sparse_W_lowrank_sym = @w_sparse_W_lowrank_sym;
    gen.w_sparse_W_lowrank_asym = @w_sparse_W_lowrank_asym;
    gen.w_sparse_W_sparse = @w_sparse_W_sparse;
    gen.w_sparse_W_block = @w_sparse_W_block;
end

function [w0, w, V, W, y] = v1(x, k, noise, scale_w0, scale_w, scale_V, mask_w, mask_W)
    p = size(x, 2);
    n = size(x, 1);
    w0 = ((rand - 0.5) * 2) * scale_w0;
    w = ((rand(p, 1) - 0.5) * 2) * scale_w;
    V = (rand(k, p) - 0.5) * 2 * scale_V;
    w = w .* mask_w;
    mask_W = mask_W .* mask_W';
    W = (V' * V) .* mask_W;
    y = w0 + x * w + diag(x * W * x');
    y = y + normrnd(0, noise, n, 1);
    report(w0, w, W);
end

function [w0, w, V, W, y] = w_sparse_W_lowrank_sym(x, k, noise, scale_w0, scale_w, scale_W, mask_w)
    p = size(x, 2);
    n = size(x, 1);
    w0 = ((rand - 0.5) * 2) * scale_w0;
    w = ((rand(p, 1) - 0.5) * 2) * scale_w;
    V = sqrt(rand(k, p));
    negative = rand(k, p) < 0.5;
    V(negative) = -V(negative);
   
    w = w .* mask_w;
    W = (V' * V) * scale_W;
    y = w0 + x * w + diag(x * W * x');
    y = y + normrnd(0, noise, n, 1);
    report(w0, w, W, 'w_sparse_W_lowrank_sym');
end

function [w0, w, V, W, y] = w_sparse_W_lowrank_asym(x, k, noise, scale_w0, scale_w, scale_W, mask_w)
    p = size(x, 2);
    n = size(x, 1);
    k = floor(k / 2);
    w0 = ((rand - 0.5) * 2) * scale_w0;
    w = ((rand(p, 1) - 0.5) * 2) * scale_w;
    A = rand(p, p);
    [U, S, V]= svd(A);
    W = U(:, 1: k)* S(1: k, 1: k)* V(:, 1: k)';
    W = (W + W') / 2;
    w = w .* mask_w;
    W = W * scale_W;
    y = w0 + x * w + diag(x * W * x');
    y = y + normrnd(0, noise, n, 1);
    report(w0, w, W, 'w_sparse_W_lowrank_asym');
end

function [w0, w, W, y] = w_sparse_W_sparse(x, noise, scale_w0, scale_w, scale_W, mask_w, mask_W)
    p = size(x, 2);
    n = size(x, 1);
    w0 = ((rand - 0.5) * 2) * scale_w0;
    w = ((rand(p, 1) - 0.5) * 2) * scale_w;
    W = (rand(p, p) - 0.5) * 2;
    W = (W + W') / 2;
    w = w .* mask_w;
    mask_W = mask_W .* mask_W';
    W = W .* mask_W * scale_W;
    y = w0 + x * w + diag(x * W * x');
    y = y + normrnd(0, noise, n, 1);
    report(w0, w, W, 'w_sparse_W_sparse');
end

function [w0, w, W, y, low_rank_list] = w_sparse_W_block(x, noise, scale_w0, scale_w, scale_W, mask_w, nblocks)
    p = size(x, 2);
    n = size(x, 1);
    w0 = ((rand - 0.5) * 2) * scale_w0;
    w = ((rand(p, 1) - 0.5) * 2) * scale_w;
    w = w .* mask_w;
    
    low_rank_list = [];
    W = zeros(p, p);
    block_size = floor(p / nblocks);
    if block_size == 0
        disp('nblocks is too big');
        return
    end
    
    for i = 1 : nblocks
        k = randi(nblocks);
        A = rand(block_size, block_size);
        [U, S, V]= svd(A);
        W_i = U(:, 1: k)* S(1: k, 1: k)* V(:, 1: k)' * scale_W;
        from = (i - 1) * block_size + 1;
        to = i * block_size;
        W(from : to, from : to) = W_i;
        low_rank_list = [low_rank_list; k];
    end
    
    rest = p - block_size * nblocks;
    if rest > 0
        k = randi(rest);
        low_rank_list = [low_rank_list; k];
        V = sqrt(rand(rest, k));
        W_i = (V' * V) * scale_W;
        W(p - rest : p, p - rest : p) = W_i;
    end
    y = w0 + x * w + diag(x * W * x');
    y = y + normrnd(0, noise, n, 1);
    report(w0, w, W, 'w_sparse_W_block');
end

function report(w0, w, W, name) % report the scale of non zero entries
    disp(['--- ', name, ' ---']);
    disp(['scale of w0 = ', num2str(abs(w0))]);
    scale_w = mean(abs(w(w ~= 0)));
    disp(['scale of w = ', num2str(scale_w)]);
    scale_W = mean(abs(W(W ~= 0)));
    disp(['scale of W = ', num2str(scale_W)]);
end
    