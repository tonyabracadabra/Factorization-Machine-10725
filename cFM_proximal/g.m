function gg=g(y,X,w,Z)
	gg=0.5*sum((y-X*w-diag(X*Z*X')).^2);
end