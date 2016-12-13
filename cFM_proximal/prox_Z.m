function [ff,Z]=prox_Z(Z,w,y,X,alpha,beta) 
    beta0=0.1;
    f_pre=f(y,X,w,Z,alpha,beta);
    epsilon=1e-3;
    ff=[f_pre];
    while true
        %proximal gradient descent in Z
        %%backtrack
        t=1;
        Gt_=Gt(Z,w,y,X,t,beta);
        while g(y,X,w,Z-t*Gt_)>g(y,X,w,Z)-t*trace(grad_g(Z,w,y,X)*Gt_')+t/2*sum(sum(Gt_.^2))  %%% check here
            t=t*beta0;
            Gt_=Gt(Z,w,y,X,t,beta);
        end
        %%prox update
        Z=Z-t*grad_g(Z,w,y,X); %%%check grad !!!!
        Z=prox(Z,t,beta);

        %check
        if abs(f_pre-f(y,X,w,Z,alpha,beta))<epsilon
            break
        end
        f_pre=f(y,X,w,Z,alpha,beta);
        ff=[ff,f_pre];
    end
    f_pre;
end
