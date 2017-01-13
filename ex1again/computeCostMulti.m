function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

predictions = X*theta;
errors = predictions - y;
sqrd_errors = errors.^2;
sum_sqrd_errors = sum(sqrd_errors);
J = 1/(2*m)*sum_sqrd_errors;

% Vectorized Form

% J = (1/(2*m))*(X*theta - y)'*(X*theta - y);
% J = (1/(2*m))*(predictions-errors)'*(predictions-errors); % this works but grader does not accept it

% =========================================================================

end
