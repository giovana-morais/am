function [h, display_array] = visualizaDados(X, example_width)
%VISUALIZADADOS Plota os dados em 2D de forma agradavel
%   [h, display_array] = VISUALIZADADOS(X, example_width) plota os dados em 2D
%   armazenados em X de forma agradavel. Retorna a figura e o conjunto selecionado, se necessario.

% Se nao for passado o tamanho do exemplo, define um fixo
if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = round(sqrt(size(X, 2)));
end

% Imagem cinza
colormap(gray);

% Enontra numero de linhas e colunas
[m n] = size(X);
example_height = (n / example_width);

% Calcula numeros de itens a ser exibido
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);

% Espacamento entre imagens
pad = 1;

% Define exibicao em branco
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));

% Copia cada exemplo para um slot na exibicao
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m, 
			break; 
		end
		max_val = max(abs(X(curr_ex, :)));
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

% Exibe a imagem
h = imagesc(display_array, [-1 1]);
axis image off
drawnow;

end
