function ZZ=Gt(Z,w,y,X,t,beta)
	ZZ=(Z-prox(Z-t*grad_g(Z,w,y,X),t,beta))/t;
end