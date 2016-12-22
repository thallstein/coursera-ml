function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

    % Initialize some useful values
    n = length(y); % number of training examples
    
    % cost: J
    J = -1/n * sum(...
        y      .* log(sigmoid(X * theta)) + ...
        (1 - y) .* log(1 - sigmoid(X * theta)) ...
    );

    % gradient: compute as the derivative of the cost function
    grad = 1/n * X' * ((sigmoid(X * theta)) - y);


% htheta = sigmoid(X * theta);
% J = 1 / m * sum(-y .* log(htheta) - (1 - y) .* log(1 - htheta));
% for i = 1:size(theta, 1)
%    grad(i) = 1 / m * sum((htheta - y) .* X(:, i));
% end


% unregularized terms

%prob1 = sigmoid(theta'*X');
%sumProb1 = y'*log(prob1)';
%sumProb0 = (1-y)'*log(1-prob1)';
%unregCost = 1/m*(sumProb1 - sumProb0);

% regularization term

% set theta 0 to 0
%theta(1) = 0; 

%lambda = 1;
%regularizedTerm = (lambda / (2 * m)) *theta'*theta;

%J = unregCost + regularizedTerm;

% =============================================================

end
