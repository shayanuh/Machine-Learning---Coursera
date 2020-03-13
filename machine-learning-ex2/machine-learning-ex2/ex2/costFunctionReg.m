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
hx= sigmoid(theta'*X');
for i= 1 : m
    J=J+(-y(i)*(log(hx(i)))-(1-y(i))*(log(1-hx(i))));
   for j= 1:size(theta)
    grad(j)=grad(j)+(hx(i)-y(i))*X(i,j);
    end
end
J=J/m;
for i= 2 : size(theta)
    J=J+(lambda)*(theta(i)^2)/(2*m);
   
end
grad=grad./m;
for i= 2:size(theta)
    grad(i)=grad(i)+((lambda*theta(i))/m);
    
end




% =============================================================

end
