%% test data generator
data_gen = data_generator;
p = 100;
n = 200;
x = rand(n, p);
noise = 0.5;
scale_w0 = 5;
scale_w = 1;
scale_W = 0.5;
%% test w_sparse_W_lowrank_sym
k = 5;
mask_w = rand(p, 1) > 0.5;
[w0, w, V, W1, y] = data_gen.w_sparse_W_lowrank_sym(x, k, ...
    noise, scale_w0, scale_w, scale_W, mask_w);
%% test w_sparse_W_lowrank_asym
[w0, w, V, W2, y] = data_gen.w_sparse_W_lowrank_asym(x, k, ...
    noise, scale_w0, scale_w, scale_W, mask_w);
%% test w_sparse_W_sparse
% mask_W = rand(p, p) > 0.8;
% [w0, w, W, y] = data_gen.w_sparse_W_sparse(x, noise, ...
%     scale_w0, scale_w, scale_W, mask_w, mask_W);
%% test w_sparse_W_block
% nblocks = 4
% [w0, w, W, y, low_rank_list] = data_gen.w_sparse_W_block(x, noise, scale_w0, scale_w, scale_W, mask_w, nblocks);