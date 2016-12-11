function ff2=f2(y,X,w,Z,beta)
	[U,S,V] = svd(Z); %%% right or wrong ????
	ff2=g(y,X,w,Z)+beta*sum(diag(S));
end