function [J, grad] = regression_cost_func(theta, X, y, lambda)
  % funcao de custo
	m = length(y); % numero de exemplos de treinamento

	sum_theta = 0;
	grad = zeros(size(theta));
	  
	  h_theta = sigmoid(X * theta);

	  J = ((-y' * log(h_theta)) - (1 - y)' * log(1 - h_theta));
  
	  theta(1) = 0;
	  
	  sum_theta = sumsq( theta(2:end) );
	  
	  J = J/m + sum_theta/(2*m);
	  
	  dJ = h_theta - y;

	  grad = ((dJ' * X)/m) + ((lambda/m).*theta)';
end