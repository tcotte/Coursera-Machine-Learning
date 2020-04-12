function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

##Columns number of X
col = size(X,2);

% Regularized linear regression cost function

h = X*theta;
J_sin_reg = (1/(2*m))*sum((h-y).^2);

reg = (lambda / m) * (sum(theta .^ 2) - theta(1) ^ 2);


J = J_sin_reg + reg*0.5;

% Regularized linear regression gradient
##Init grad
grad = [1, size(X,2)];

der_J = (1/m)*sum(X(:,1)'*(h-y));

grad(1,1) = der_J;
for i=2:col
##  der_J_reg = (1/m)*sum(X(:,2)'*(h-y))+reg
  grad(1, i) = (1/m)*sum(X(:,i)'*(h-y)) + (lambda / m) *theta(i);
endfor





% =========================================================================

grad = grad(:);

end
