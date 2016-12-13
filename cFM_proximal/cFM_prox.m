function [ff,w,Z]=cFM_prox(X,y,alpha,beta,epsilon)
	[n,p]=size(X);
	w=ones(p,1);
	Z=ones(p,p);

	f_pre=f(y,X,w,Z,alpha,beta)

	ff=[f_pre];
	while true
	    %gradient descent on w, 
	    w=gradient_w(y,X,w,Z,alpha);

	    %proximal gradient on Z
	    Z=prox_Z(Z,w,y,X,beta);

	    %check
	    if abs(f_pre-f(y,X,w,Z,alpha,beta))<epsilon
	        break
	    end
	    f_pre=f(y,X,w,Z,alpha,beta)
	    ff=[ff,f_pre];
	end
	plot(ff);
end