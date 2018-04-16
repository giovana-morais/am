function [J, grad] = funcaoCusto(theta, X, y)
%FUNCAOCUSTO Calcula o custo da regressao logistica
%   J = FUNCAOCUSTO(X, y, theta) calcula o custo de usar theta como 
%   parametro da regressao logistica para ajustar os dados de X e y

% Initializa algumas variaveis uteis
m = length(y); % numero de exemplos de treinamento

% Voce precisa retornar as seguintes variaveis corretamente
J = 0;
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
  
  % seguindo a minha implementaçao da regressao linear:
  for i = 1: m
    J = J + (-y(i) * log(h_theta(i)) - (1-y(i)) * log(1 - h_theta(i)));
  endfor
  J = J/m;
  % MAS da pra fazer diretao se transpor y. talvez seja mais eficiente??? 
  %J = ((-y' * log(h_theta)) - (1 - y)' * log(1 - h_theta))/m;
  
  dJ = h_theta - y;
  grad = (dJ' * X)/m;
  
% =============================================================

end
