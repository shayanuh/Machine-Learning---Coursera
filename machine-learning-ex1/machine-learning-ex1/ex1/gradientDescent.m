function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    r1=0,r2=0;
    h = (theta')*(X');
    for i = 1:m
        r1= r1+(h(i)-y(i))*X(i,1);
        r2 = r2+(h(i)-y(i))*X(i,2)
    end
    
    temp1 = theta(1)- (alpha/m)*(r1);
    temp2 = theta(2)- (alpha/m)*(r2);
    theta(1)= temp1;
    theta(2)=temp2;





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
