function x=fillnan(x,n)

if length(x)<n
  x(length(x)+1:n)=nan;
end
