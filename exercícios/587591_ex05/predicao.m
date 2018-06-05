function p = predicao(Theta1, Theta2, X)
%PREDICAO Prediz o rotulo de uma amostra apresentada a rede neural
%   p = PREDICAO(Theta1, Theta2, X) prediz o rotulo de X ao utilizar
%   os pesos treinados na rede neural (Theta1, Theta2)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

h1 = sigmoide([ones(m, 1) X] * Theta1');
h2 = sigmoide([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);

% =========================================================================


end
