function p = prediction_1l(Theta1, Theta2, X)
  % Prediz o rotulo de uma amostra apresentada a rede neural de 1 camada oculta
  m = size(X, 1);
      
  h1 = sigmoid([ones(m, 1) X] * Theta1');
  h2 = sigmoid([ones(m, 1) h1] * Theta2');
    
  [dummy, p] = max(h2, [], 2);

end
