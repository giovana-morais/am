function g = sigmoid(z)
	g = zeros(size(z));
	g = 1 ./(1 + exp(-z));
end

function p = predicao(theta, X)
%PREDICAO Prediz se a entrada pertence a classe 0 ou 1 usando o parametro
%theta obtido pela regressao logistica
%   p = PREDICAO(theta, X) calcula a predicao de X usando um 
%   limiar igual a 0.5 (ex. se sigmoid(theta'*x) >= 0.5, classe = 1)
	m = size(X, 1); % Numero de examplos de treinamento

	p = zeros(m, 1);

	p = sigmoid(X * theta);

	p(p < 0.5) = 0;
	p(p >= 0.5) = 1;

end

function [J, grad] = funcaoCustoReg(theta, X, y, lambda)
%FUNCAOCUSTOREG Calcula o custo da regressao logistica com regularizacao
%   J = FUNCAOCUSTOREG(theta, X, y, lambda) calcula o custo de usar theta 
%   como parametros da regressao logistica para ajustar os dados de X e y 

	m = length(y); % numero de exemplos de treinamento

	J = 0;
	sum_theta = 0;
	grad = zeros(size(theta));
	  
	  h_theta = sigmoid(X * theta);

	  % size(grad) = [28,1]  
	J = ((-y' * log(h_theta)) - (1 - y)' * log(1 - h_theta))/m;
	  %for i = 1: m
	%	J = J + (-y(i) * log(h_theta(i)) - (1-y(i)) * log(1 - h_theta(i)));
	%  endfor
	  
	  for j = 2: size(theta)
		sum_theta = sum_theta + theta(j) ^ 2;
	  endfor
	  J = J/m + sum_theta/(2*m);
	  
	  dJ = h_theta - y;

	  theta(1) = 0;
	  grad = ((dJ' * X)/m) + ((lambda/m).*theta)';
end

function [J, grad] = funcaoCusto(theta, X, y)
m = length(y); % numero de exemplos de treinamento

% Voce precisa retornar as seguintes variaveis corretamente
J = 0;
grad = zeros(size(theta));
  
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
end

function out = atributosPolinomiais(X1, X2)
% ATRIBUTOS POLINOMIAIS Gera atributos polinomiais a partir dos atriburos
% originais da base
%
%   ATRIBUTOSPOLINOMIAIS(X1, X2) mapeia os dois atributos de entrada
%   para atributos quadraticos
%
%   Retorna um novo vetor de mais atributos:
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   As entradas X1, X2 devem ser do mesmo tamanho
%

grau = 6;
out = ones(size(X1(:,1)));
for i = 1:grau
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end


%% Carrega os dados
load('ex03Dados2.mat');

%  Configura a matriz apropriadamente e adiciona uns na primeira coluna
[m, n] = size(X);

X = [ones(m, 1) X];

% Adiciona atributos polinomiais calculados a partir dos atributos
% originais

% atributosPolinomiais adiciona novas colunas que correspondem a atributos
% polinomiais
X = atributosPolinomiais(X(:,1), X(:,2));

% Inicializa os parametros que serao ajustados
theta_inicial = zeros(size(X, 2), 1);

% Configura parametro de regularizacao lambda igual a 1
lambda = 1;

% Calcula e exibe custo inicial e gradiente para regressao logistica com
% regularizacao
[custo, grad] = funcaoCustoReg(theta_inicial, X, y, lambda);

%% ============= Parte 8: Regularizacao e desempenho =============
%  Nesta etapa, voce pode testar diferente valores de lambda e verificar
%  como a regularizacao afeta o limite de decisao
%

% Inicializa os parametros que serao ajustados
theta_inicial = zeros(size(X, 2), 1);

% Configura o parametro regularizacao lambda igual a 1
lambda = 1;

% Configura opticoes
opcoes = optimset('GradObj', 'on', 'MaxIter', 400);

% Otimiza o gradiente
[theta, J, exit_flag] = ...
	fminunc(@(t)(funcaoCustoReg(t, X, y, lambda)), theta_inicial, opcoes);


%% ============= Parte 9: Predizendo a classe de novos dados =============

fprintf('\n\nPredizendo a classe de novos dados...\n\n')

x1_novo = input('Informe o comprimento da p�tala (normalizado) ou -1 para SAIR: ');

while (x1_novo ~= -1)
    x2_novo = input('Informe a largura da p�tala (normalizado): ');
    x_novo = atributosPolinomiais(x1_novo, x2_novo);

    classe = predicao(theta, x_novo); % Faz a predicao usando theta encontrado
    
    if (classe)
        fprintf('Classe = Iris Virginica (y = 1)\n\n');
    else
        fprintf('Classe = Iris Versicolour (y = 0)\n\n');
    end
    
    x1_novo = input('Informe o comprimento da p�tala (em cm) ou -1 para SAIR: ');
end

%% Finalizacao
clear; close all;
