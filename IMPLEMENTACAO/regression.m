function ypred = regression(train, train_labels)
	[m, n] = size(train);

	train = [ones(m, 1) train];

  % thetas for all the n labels. each row means a classifier for a label.
  all_thetas = [];
   
	% Adiciona atributos polinomiais calculados a partir dos atributos
	% originais
	%X = atributosPolinomiais(X(:,1), X(:,2));

	% Inicializa os parametros que serao ajustados
	theta_inicial = zeros(n+1, 1);
  
	% set lambda. 
  % TODO: change lambda and check best results
	lambda = 1;
  
  % options for improve descendent gradient
	options = optimset('GradObj', 'on', 'MaxIter', 400);

  % one_vs_all: k iterates over all 6 labels 
  for k = 1: 6
    class_labels = (train_labels == k);
    
	  [theta, J, exit_flag] = ...
		fminunc(@(t)(cost_func(t, train, class_labels, lambda)), theta_inicial, options);

     all_thetas = [all_thetas, theta];
  end
  
  printf("testando...");
  y_pred = predict(all_thetas, train);
  fprintf('\nAcurácia: %f\n', mean(double(y_pred == train_labels)) * 100);
end


function g = sigmoid(z)
	g = zeros(size(z));
	g = 1 ./(1 + exp(-z));
end

function p = predict(all_thetas, train)
  size(all_thetas)
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

	J = ((-y' * log(h_theta)) - (1 - y)' * log(1 - h_theta))/m;
  
	  theta(1) = 0;
	  
    % TODO: tirar isso aqui desse for tbm
	  for j = 2: size(theta)
		  sum_theta = sum_theta + theta(j) ^ 2;
	  endfor
	  J = J/m + sum_theta/(2*m);
	  
	  dJ = h_theta - y;

	  grad = ((dJ' * X)/m) + ((lambda/m).*theta)';
end


% tô pensando em tacar fogo nessa funcao
function out = atributosPolinomiais(X1, X2)
	grau = 6;
	out = ones(size(X1(:,1)));
	for i = 1:grau
		for j = 0:i
		    out(:, end+1) = (X1.^(i-j)).*(X2.^j);
		end
	end
end


