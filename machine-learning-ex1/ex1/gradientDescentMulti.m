function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta
X
h = zeros(m,1)
num_iters
deltaJ = zeros(size(X))
[nl, nc] = size(X)
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

% Code for 3 variables
##predictions = theta(1) + (theta(2)*X(:,2)) + (theta(3)*X(:,3));
##h = X*theta;
##if(predictions == h)
##  printf('true \n')
##endif
##  theta0 = theta(1) - alpha*(1/m)*sum(h-y);
##  theta1 = theta(2) - alpha*(1/m)*sum((h-y).*X(:,2));
##  theta2 = theta(3)-alpha*(1/m)*sum((h-y).*X(:,3));
##
##
##  theta = [theta0; theta1; theta2];

% Code for n variables
theta = theta -(alpha * (1/m) * ((X') * ((X * theta)- y)))

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
