function W = inicializaPesosAleatorios(L_in, L_out)
%INICIALIZAPESOSALEATORIOS Inicializa aleatoriamente os pesos de uma camada usando L_in
%como conexoes de entrada e L_out como conexoes de saida.
%   W = INICIALIZAPESOSALEATORIOS(L_in, L_out) nicializa aleatoriamente os pesos 
%       de uma camada usando L_in (conexoes de entrada) e L_out (conexoes de saida).
%
%   W sera definido como uma matriz de dimensoes [L_out, 1 + L_in]
%   visto que devera armazenar os termos para "bias".
%

epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

end
