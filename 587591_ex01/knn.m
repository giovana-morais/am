function [y, ind_viz] = knn(x, X, Y, K)
%KNN m�todo do K-vizinhos mais proximos para predizer a classe de um novo
%   dado.
%   [y, ind_viz] = KNN (x, X, Y, K) retorna o rotulo de x em y e os indices
%       ind_viz dos K-vizinhos mais proximos em X.
%
%       Parametros de entrada:
%       -> x (1xn): amostra a ser classificada
%       -> X (mxn): base de dados de treinamento
%       -> Y (mx1): rotulo de cada amostra de X
%       -> K (1x1): quantidade de vizinhos mais proximos
%
%       Parametros de saida:
%       -> y (1x1): predicao (0 ou 1) do rotulo de x
%       -> ind_viz (Kx1): indice das K amostras mais proximas de x
%                         encontradas em X (da mais proxima a menos
%                         proxima)
%

%  Inicializa a variavel de retorno e algumas variaveis uteis
y = 0;                % Inicializa rotulo como classe negativa
ind_viz = ones(K,1);  % Inicializa indices (linhas) em X das K amostras mais 
                      % proximas de x.


% ====================== ESCREVA O SEU CODIGO AQUI ========================
% Instrucoes: Implemente o m�todo dos K-vizinhos mais proximos. Primeiro, 
%             eh preciso calcular a distancia entre x e cada amostra de X. 
%             Depois, encontre os K-vizinhos mais proximos e use voto
%             majoritario para definir o rotulo de x. 
%
% Obs: primeiro eh necessario implementar a funcao de similaridade
%      (distancia).
%

%  Calcula a distancia entre a amostra de teste x e cada amostra de X. Voce
%  devera completar essa funcao.
  D = distancia(x, X);  
  
  % pega os K menores objetos do vetor de distancias
  ind_viz = find(D == min(D), K);
  
  if K > 1
    y = Y(mode(ind_viz));
  else
    y = Y(ind_viz);
  endif
% =========================================================================

end