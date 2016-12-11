function varargout=obj_lrl1(xx, info, A, yy, lambda)

[fval,gg]=loss_lrp(A*xx, yy);
pp=-gg.*yy;
hh=pp.*(1-pp);

fval=fval+lambda*sum(abs(xx));
gg  =A'*gg+lambda*sign(xx);

I1=find(xx==0 & gg>0);
I2=find(xx==0 & gg<0);

gg(I1)=max(gg(I1)-lambda, 0);
gg(I2)=min(gg(I2)+lambda, 0);

info.ginfo=norm(gg);

varargout{1}=fval;
if nargout>2
  varargout{2}=gg;
  if nargout>3
    H=A'*diag(hh)*A;
    varargout{3}=H;
    varargout{4}=info;
  else
    varargout{3}=info;
  end
else
  varargout{2}=info;
end
