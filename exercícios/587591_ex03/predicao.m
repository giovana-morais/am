function p = predicao(theta, X)
%PREDICAO Prediz se a entrada pertence a classe 0 ou 1 usando o parametro
%theta obtido pela regressao logistica
%   p = PREDICAO(theta, X) calcula a predicao de X usando um 
%   limiar igual a 0.5 (ex. se sigmoid(theta'*x) >= 0.5, classe = 1)

m = size(X, 1); % Numero de examplos de treinamento

% Voce precisa retornar a seguinte variavel corretamente
p = zeros(m, 1);

% ====================== ESCREVA O SEU CODIGO AQUI ======================
% Instrucoes: Complete o codigo a seguir para fazer predicoes usando
%               os paramentros ajustados pela regressao logistica. 
%               p devera ser um vetor composto por 0's e 1's
%

p = sigmoid(X * theta);

p(p < 0.5) = 0;
p(p >= 0.5) = 1;

% =========================================================================


end

