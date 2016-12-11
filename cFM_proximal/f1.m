function ff1=f1(y,X,w,Z,alpha)
	ff1=g(y,X,w,Z)+alpha/2*sum(w.^2);
end