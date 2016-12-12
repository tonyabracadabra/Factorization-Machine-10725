function [ff,w,Z]=cFM_prox(X,y,alpha,beta,epsilon, maxstep)
	[n,p]=size(X);
	w=zeros(p,1);
	Z=zeros(p,p);

	f_pre=f(y,X,w,Z,alpha,beta)

	ff=[f_pre];
	counter = 1;
	while true && maxstep >= counter
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
	    counter = counter + 1;
	end
	% plot(ff);
end