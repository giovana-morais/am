function W = inicializaPesos(fan_out, fan_in)
%INICIALIZAPESOS Inicializa os pesos para uma camada com fan_in
%(numero de conexoes de entrada) e fan_out (numero de conexoes de saida)
%usando uma estrategia fixa, ajudando a testar seu codigo
%sem aleatoriedade embutida.
%   W = INICIALIZAPESOS(fan_in, fan_out) Inicializa os pesos para 
%        uma camada com fan_in (numero de conexoes de entrada) 
%        e fan_out (numero de conexoes de saida) de forma fixa
%
%   Observe que W sera definido como uma matriz do tamanho (1 + fan_in, fan_out)
%   por conta que a primeira linha eh separada para o "bias"
%

% Define W como matriz de zeros
W = zeros(fan_out, 1 + fan_in);

% Inicializa W usando a funcao "sin", para garantir sempre os mesmos valores
W = reshape(sin(1:numel(W)), size(W)) / 10;

% =========================================================================

end
