function p = prediction_2l(Theta1, Theta2, Theta3, X)
  % Prediz o rotulo de uma amostra apresentada a rede neural de 2 camadas ocultas

  m = size(X, 1);
      
  h1 = sigmoid([ones(m, 1) X] * Theta1');
  h2 = sigmoid([ones(m, 1) h1] * Theta2');
  h3 = sigmoid([ones(m, 1) h2] * Theta3');
  
  [dummy, p] = max(h3, [], 2);

end
