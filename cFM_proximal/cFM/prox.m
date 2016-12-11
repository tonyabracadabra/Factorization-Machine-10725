function ZZ=prox(Z,t,beta)
	[n,~]=size(Z);
    [U,S,V] = svd(Z);
    S_=max(S-beta*t*eye(n),0);
    ZZ=U*S_*V';
end