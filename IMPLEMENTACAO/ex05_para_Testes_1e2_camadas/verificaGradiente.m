function verificaGradiente(lambda)
%VERIFICAGRADIENTE Cria uma pequena rede neural para verificar
%os gradientes de backpropagation
%   VERIFICAGRADIENTE(lambda) Cria uma pequena rede neural para verificar
%   os gradientes de backpropagation. Serao exibidos o gradiente produzido
%   pelo seu codigo de backpropagation e o gradiente numerico obtido 
%   na funcao gradienteNumerico. Ambos gradientes devem ser bem proximos.
%

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

input_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;
m = 5;

% Dados de teste gerados aleatoriamente
Theta1 = inicializaPesos(hidden_layer_size, input_layer_size);
Theta2 = inicializaPesos(num_labels, hidden_layer_size);

% Utiliza a funcao inicializaPesos para criar X
X  = inicializaPesos(m, input_layer_size - 1);
y  = 1 + mod(1:m, num_labels)';

% Junta os parametros
nn_params = [Theta1(:) ; Theta2(:)];

funcaoCusto = @(p) rnaCusto(p, input_layer_size, hidden_layer_size, ...
                               num_labels, X, y, lambda);

[cost, grad] = funcaoCusto(nn_params);
numgrad = gradienteNumerico(funcaoCusto, nn_params);

% Examina visualmente os dois gradientes.
% As duas colunas devem ser bem proximas.
disp([numgrad grad]);
fprintf(['As duas colunas acima deve ser bem semelhantes..\n' ...
         '(Esquerda - Gradiente numerico, Direita - Seu gradiente)\n\n']);

% Calcula a diferenca entre as duas solucoes
% Se a implementacao estiver correta, assumindo epsilon = 0.0001 
% na funcao gradienteNumerico.m, a diferenca devera ser menor que 1e-9
diff = norm(numgrad-grad)/norm(numgrad+grad);

fprintf(['Se sua implementacao de backpropagation esta correta, \n' ...
         'a diferenca relativa devera ser pequena (menor que 1e-9). \n' ...
         '\nDiferenca relativa: %g\n'], diff);

end
