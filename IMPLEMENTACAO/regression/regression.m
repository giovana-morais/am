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
		fminunc(@(t)(regression_cost_func(t, train, class_labels, lambda)), init_theta, options);

     all_thetas = [all_thetas, theta];
  end

  %y_train = predict(all_thetas, train);  
  %printf('\nAcurácia na base de treinamento: %f\n', mean(double(y_train == train_labels) * 100));
  
  test = [ones(size(test),1) test];
  y_pred = regression_predict(all_thetas, test);
end











