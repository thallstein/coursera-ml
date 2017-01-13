function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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



% We want the regularization to exclude the bias feature, so we can set theta(1) to zero. 
% Since we already calculated h, and theta is a local variable, we can modify theta(1) without causing any problems.

    
    % cost: J, this time with the penalty for the magnitude of theta
    J = -1/m * sum(...
        y      .* log(sigmoid(X * theta)) + ...
        (1 - y) .* log(1 - sigmoid(X * theta)) ...
    ) + lambda / (2 * m) * sum(theta(2:end) .* theta(2:end));

    % gradient: compute as the derivative of the cost function
    grad = 1/m * X' * ((sigmoid(X * theta)) - y) + theta * lambda / m;
   
    % we do not regularize the constant offset term, 
    % undo the gradient penalization for the constant x_0 term
    grad(1) = grad(1) - lambda / m * theta(1);

% =============================================================

end
