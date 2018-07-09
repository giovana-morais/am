function p = regression_predict(all_thetas, train)
  % predicao da regressao dada amostra por train
  
	p = zeros(size(all_thetas,1), 1);
	p = sigmoid(train * all_thetas);

  [maxVal, maxIx] = max(p, [], 2);
  p = maxIx;
end