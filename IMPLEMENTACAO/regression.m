function y_pred = regression(train, train_labels, test, lambda)
	[m, n] = size(train);
  % thetas for all the n labels. each row means a classifier for a label.
  all_thetas = [];
   
  train = [ones(m, 1) train];
  new_train = poly_features(train);
	% Inicializa os parametros que serao ajustados
	init_theta = zeros(n+1, 1);
  
  % options for improve descendent gradient
	options = optimset('GradObj', 'on', 'MaxIter', 400);

  % one_vs_all: k iterates over all 6 labels 
  for k = 1: 6
    class_labels = (train_labels == k);
    
	  [theta, J, exit_flag] = ...
		fminunc(@(t)(cost_func(t, train, class_labels, lambda)), init_theta, options);

     all_thetas = [all_thetas, theta];
  end

  %y_train = predict(all_thetas, train);  
  %printf('\nAcurácia na base de treinamento: %f\n', mean(double(y_train == train_labels) * 100));
  
  test = [ones(size(test),1) test];
  y_pred = predict(all_thetas, test);
end


function g = sigmoid(z)
	g = zeros(size(z));
	g = 1 ./(1 + exp(-z));
end

function p = predict(all_thetas, train)
	p = zeros(size(all_thetas,1), 1);
	p = sigmoid(train * all_thetas);

  [maxVal, maxIx] = max(p, [], 2);
  p = maxIx;
end

function [J, grad] = cost_func(theta, X, y, lambda)
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

% generate new features with different polynomial degree
function out = poly_features(X)
	grau = 3;
	out = ones(size(X(:,1)));  
  
	for i = 1:grau
		for j = 0:i
		    out(:, end+1) = (X(:,1).^(i-j)).*(X(:,4).^j);
		end
	end
end


