datatype = {'w_sparse_W_sparse', 'w_sparse_W_block', ...
    'w_sparse_W_lowrank_sym', 'w_sparse_W_lowrank_asym'};
p = 30;
i = 1;
[objs_coor_container_i1, objs_prox_container_i1, objs_owlqn_container_i1] = test_ADMM(datatype{i}, p);
save('objs_coor_container_i1.mat', 'objs_coor_container_i1')
save('objs_prox_container_i1.mat', 'objs_prox_container_i1')
save('objs_owlqn_container_i1.mat', 'objs_owlqn_container_i1')
