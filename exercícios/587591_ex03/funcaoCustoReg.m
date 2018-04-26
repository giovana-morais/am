function [J, grad] = funcaoCustoReg(theta, X, y, lambda)
%FUNCAOCUSTOREG Calcula o custo da regressao logistica com regularizacao
%   J = FUNCAOCUSTOREG(theta, X, y, lambda) calcula o custo de usar theta 
%   como parametros da regressao logistica para ajustar os dados de X e y 

% Initializa algumas variaveis uteis
m = length(y); % numero de exemplos de treinamento

% Voce precisa retornar as seguintes variaveis corretamente
J = 0;
sum_theta = 0;
grad = zeros(size(theta));

% ====================== ESCREVA O SEU CODIGO AQUI ======================
% Instrucoes: Calcule o custo de uma escolha particular de theta.
%             Voce precisa armazenar o valor do custo em J.
%             Calcule as derivadas parciais e encontre o valor do gradiente
%             para o custo com relacao ao parametro theta
%
% Obs: grad deve ter a mesma dimensao de theta
%
  
  h_theta = sigmoid(X * theta);

  % size(grad) = [28,1]  
  % TODO: se der tempo, pensar na versao sem usar o for e se compensa por aqui
  for i = 1: m
    J = J + (-y(i) * log(h_theta(i)) - (1-y(i)) * log(1 - h_theta(i)));
  endfor
  
  for j = 2: size(theta)
    sum_theta = sum_theta + theta(j) ^ 2;
  endfor
  J = J/m + sum_theta/(2*m);
  
  dJ = h_theta - y;

  theta(1) = 0;
  grad = ((dJ' * X)/m) + ((lambda/m).*theta)';
  
% =============================================================

end
