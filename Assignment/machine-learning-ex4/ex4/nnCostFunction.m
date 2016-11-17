function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

K = num_labels;
X = [ones(m,1) X];

for i = 1 : m
    % getting data of i-th training example
    X_i = X(i,:);
    
    % computing hypothesis of X_i
    H_Xi = sigmoid( [1 sigmoid(X_i * Theta1')] * Theta2'); % dim: 1xK
    
    % getting labels as vectors. E.g. [0 1 0 ... 0]
    Y_i = zeros(1,K); % dim: 1xK
    Y_i(y(i)) = 1;
    
    % computing cost function
    J = J + sum(-1 * Y_i .* log(H_Xi) - (1 - Y_i) .* log(1 - H_Xi));
end

% computing cost function without regularization
J = 1/m * J;

% adding regularization
Regular_1st = sum( sumsq( Theta1(:,2:input_layer_size+1) ) );
Regular_2nd = sum( sumsq( Theta2(:,2:hidden_layer_size+1) ) );
Regular = lambda/(2*m) * (Regular_1st + Regular_2nd);
J = J + Regular;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

delta_acc_1 = zeros(size(Theta1)); % dim: 25x401
delta_acc_2 = zeros(size(Theta2)); % dim: 10x26

for i = 1 : m
    % dim of Theta1: 25x401
    % dim of Theta2: 10x26
    
    % setting activation at 1st layer
    a_1 = X(i,:); % dim: 1x401
    
    % performing forward propogration to comput a_2 (hidden layer) and a_3 (output layer)
    z_2 = a_1 * Theta1'; % dim: 1x25
    a_2 = [ 1 sigmoid(z_2)]; % dim: 1x26
    
    z_3 = a_2 * Theta2'; % dim: 1x10
    a_3 = [sigmoid(z_3)]; % dim: 1x10
    
    % getting labels as vectors. E.g. [0 1 0 ... 0]
    Y_i = zeros(1,K); % dim: 1x10
    Y_i(y(i)) = 1; % dim: 1x10
    
    % computing delta: from output layer to 2nd layer
    delta_3 = a_3 - Y_i; % dim: 1x10
    delta_2 = delta_3 * Theta2 .* sigmoidGradient([1 z_2]); % dim: 1x10 * 10x26 .* [1 1x25] = 1x26
    
    % accumulating delta
    delta_acc_1 = delta_acc_1 + delta_2(2:end)' * a_1; % dim: 25x401 + (1x25)' * (1x401) = 25x401
    delta_acc_2 = delta_acc_2 + delta_3' * a_2; % dim: 10x26 + (1x10) * 1x26 = 10x26
end

% computing the unregularized gradient
Theta1_grad = delta_acc_1 / m; % dim: 25x401
Theta2_grad = delta_acc_2 / m; % dim: 10x26


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:, 2:input_layer_size+1) = Theta1_grad(:, 2:input_layer_size+1) + lambda / m * Theta1(:, 2:input_layer_size+1);
Theta2_grad(:, 2:hidden_layer_size+1) = Theta2_grad(:, 2:hidden_layer_size+1) + lambda / m * Theta2(:, 2:hidden_layer_size+1);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
