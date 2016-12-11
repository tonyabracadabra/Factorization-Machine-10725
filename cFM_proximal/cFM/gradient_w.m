function w=gradient_w(y,X,w,Z,alpha)
	alpha0=0.5;
	beta0=0.1;
	epsilon=1e-5;
	f_pre=f1(y,X,w,Z,alpha);
	while true
		t=1;
		grad_f1_=grad_f1(y,X,w,Z,alpha);
		while f1(y,X,w-t*grad_f1_,Z,alpha)>f1(y,X,w,Z,alpha)-alpha0*t*sum(grad_f1_.^2)
			t=t*beta0;
		end
	    w=w-t*grad_f1(y,X,w,Z,alpha);

	    %check
	    if abs(f_pre-f1(y,X,w,Z,alpha))<epsilon
	        break
	    end
	    f_pre=f1(y,X,w,Z,alpha);
   	end
end