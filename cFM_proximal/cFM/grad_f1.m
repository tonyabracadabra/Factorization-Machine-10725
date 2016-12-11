function grad_f1_=grad_f1(y,X,w,Z,alpha)
	grad_f1_=-X'*(y-diag(X*Z*X')-X*w)+alpha*w;
end