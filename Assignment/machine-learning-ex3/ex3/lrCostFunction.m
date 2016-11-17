function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% m: number of examples
% n: number of features
% theta: [n x 1]
% X: [m x n]
% X*theta: [m x 1]
% y: [m x 1]

% 1. computing cost and gradient for logistic regression
% computing hypothesis
Hypothesis_H = sigmoid(X * theta); % [m x 1]

% computing cost function
J = 1 / m * sum( -1 * y' *log(Hypothesis_H) - (1 - y') * log(1-Hypothesis_H) ); % [1 x 1]

% computing gradient
grad = 1 / m * (X' * (Hypothesis_H - y)); % [n x 1]


% 2. computing cost and gradient for logistic regression with regularization
% making 1st element of theta 0
theta_first_element_zeroed = [0; theta(2:length(theta));];

% computing cost function again
J = J + lambda / (2 * m) * sum(theta_first_element_zeroed.^2);

% computing gradient again
grad = grad .+ lambda / m * theta_first_element_zeroed;

% =============================================================

grad = grad(:);

end
