function ff=f(y,X,w,Z,alpha,beta)
	[U,S,V] = svd(Z); %%% right or wrong ????
	ff=g(y,X,w,Z)+alpha/2*sum(w.^2)+beta*sum(diag(S));
end