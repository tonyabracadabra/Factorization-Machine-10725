%code from: http://blog.csdn.net/rk2900/article/details/9061263
function w = ridge(X, y, lambda, pi0)
%RIDGE Ridge Regression.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%           lambda: regularization parameter.
%
%   OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE
X=X';
[P, N] = size(X);
X = [pi0';X];
w = inv(X*X'+lambda*eye(P+1))*X*y';


end
