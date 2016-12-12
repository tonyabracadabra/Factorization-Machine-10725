datatype = {'w_sparse_W_sparse', 'w_sparse_W_block', ...
    'w_sparse_W_lowrank_sym', 'w_sparse_W_lowrank_asym'};
p = 30;
i = 4;
[objs_coor_container_i4, objs_prox_container_i4, objs_owlqn_container_i4] = test_ADMM(datatype{i}, p);
save('objs_coor_container_i4.mat', 'objs_coor_container_i4')
save('objs_prox_container_i4.mat', 'objs_prox_container_i4')
save('objs_owlqn_container_i4.mat', 'objs_owlqn_container_i4')
