function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m1 = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


A1 = [ones(m1, 1) X]; % add 1s to create design matrix, col of 1s is for theta(0)
Z2 = A1 * Theta1'; % compute layer 2
X2 = sigmoid(Z2); % take the sigmoid
m2 = size(X2,1); % num rows in X2
A2 = [ones(m2,1) X2]; % add 1s to create design matrix, col of 1s is for theta(0)
Z3 = A2 * Theta2'; % compute layer 3
A3 = sigmoid(Z3); % take the sigmoid
[maxval p] = max(A3,[],2); % pull out the index of the highest value, the highest probability guess for the number


% =========================================================================


end
