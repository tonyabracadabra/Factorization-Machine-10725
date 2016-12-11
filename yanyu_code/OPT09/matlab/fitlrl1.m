function ww=fitlrl1(ww, A, yy, lambda)

[fval0,info]=obj_lrl1(ww,[],A,yy,lambda);

J=find(ww);

[fval,gg,H,info]=obj_lrl1(ww(J),[],A(:,J),yy,lambda);

R = chol(H);

dd = -R\(R'\gg);

step=1;
ww0  =ww;
while fval>=fval0 && step>1e-10
  ww(J)=ww0(J)+step*dd;
  [fval,info]=obj_lrl1(ww,[],A,yy,lambda);
  step=step/2;
end

fprintf('step=%g\n',step);




