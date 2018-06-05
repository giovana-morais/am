function [J grad] = rnaCusto(nn_params, ...
                             input_layer_size, ...
                             hidden_layer_size, ...
                             num_labels, ...
                             X, y, lambda)
%RNACUSTO Implementa a funcao de custo para a rede neural com duas camadas
%voltada para tarefa de classificacao
%   [J grad] = RNACUSTO(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) calcula o custo e gradiente da rede neural. The
%   Os parametros da rede neural sao colocados no vetor nn_params
%   e precisam ser transformados de volta nas matrizes de peso.
%
%   input_layer_size - tamanho da camada de entrada
%   hidden_layer_size - tamanho da camada oculta
%   num_labels - numero de classes possiveis
%   lambda - parametro de regularizacao
%
%   O vetor grad de retorno contem todas as derivadas parciais
%   da rede neural.
%

% Extrai os parametros de nn_params e alimenta as variaveis Theta1 e Theta2.
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Definindo variaveis uteis
m = size(X, 1);
         
% As variaveis a seguir precisam ser retornadas corretamente
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== INSIRA SEU CODIGO AQUI ======================
% Instrucoes: Voce deve completar o codigo a partir daqui 
%               acompanhando os seguintes passos.
%
% (1): Lembre-se de transformar os rotulos Y em vetores com 10 posicoes,
%      onde tera zero em todas posicoes exceto na posicao do rotulo
%
% (2): Execute a etapa de feedforward e coloque o custo na variavel J.
%      Apos terminar, verifique se sua funcao de custo esta correta,
%      comparando com o custo calculado em ex05.m.
%
% (3): Implemente o algoritmo de backpropagation para calcular 
%      os gradientes e alimentar as variaveis Theta1_grad e Theta2_grad.
%      Ao terminar essa etapa, voce pode verificar se sua implementacao 
%      esta correta atraves usando a funcao verificaGradiente.
%
% (4): Implemente a regularização na função de custo e gradiente.
%

% cria uma 'matriz de permutacao' que é na vdd uma matriz que é 1 só na classe
% da amostra avaliada
% ref: https://octave.org/doc/v4.0.1/Special-Utility-Matrices.html 
rotulos = eye(num_labels)(y,:);

% Theta1 = (25,401)
% s1 = size(Theta1)
% Theta 2 = (10,26) 
% s2 = size(Theta2)

% adiciona a coluna de 1 pros tamanhos darem certo (e pro bias)
% a1 = (5000,401)
a1 = [ones(m,1) X];
% z2 = 5000 * 25
z2 = a1 * Theta1';
% a2 = 5000 * 26
a2 = [ones(m,1) sigmoide(z2)]; 
% z3 = 5000 * 10
z3 = a2 * Theta2';  
% a3, h_theta = 5000 * 10
a3 = sigmoide(z3);
h_theta = a3;

% J regularizado 
J = 1/m * sum(sum(-rotulos.*log(h_theta) - (1-rotulos).*log(1-h_theta)),2);
J_reg = J + lambda*(sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2))) /(2*m);
J = J_reg;

% calcula os sigmas e os deltas (acumulo)
sigma_3 = a3 .- rotulos;
sigma_2 = (sigma_3 * Theta2) .* gradienteSigmoide([ones(size(z2, 1), 1) z2]);
size(sigma_2);
sigma_2 = sigma_2(:, 2:end);
 
delta_1 = sigma_2' * a1;
delta_2 = sigma_3' * a2;

Theta1_grad = (delta_1 ./ m) + (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_grad = (delta_2 ./ m) + (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

% -------------------------------------------------------------



% =========================================================================

% Junta os gradientes
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
