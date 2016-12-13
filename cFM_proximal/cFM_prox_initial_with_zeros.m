function [ff,w,Z]=cFM_prox_initial_with_zeros(X,y,alpha,beta,epsilon, maxstep)
	[n,p]=size(X);
	w=zeros(p,1);
	Z=zeros(p,p);
	f_pre=f(y,X,w,Z,alpha,beta)

	ff=[f_pre];
	counter = 1;
	while true && maxstep >= counter
	    %gradient descent on w, 
	    [ff1,w]=gradient_w(y,X,w,Z,alpha,beta);

	    %proximal gradient on Z
	    [ff2,Z]=prox_Z(Z,w,y,X,alpha,beta);

	    %check
	    if abs(f_pre-f(y,X,w,Z,alpha,beta))<epsilon
	        break
	    end
	    f_pre=f(y,X,w,Z,alpha,beta)
	    ff=[ff1,ff2,f_pre];
	    counter = counter + 1;
	end
	% plot(ff);
end