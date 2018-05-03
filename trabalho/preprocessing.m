printf("Carregando dados...\n\n");
fflush(stdout);

% leitura dos dados de treinamento
train_data = load("-ascii", "dataset_uci/final_X_train.txt");

% leitura dos dados de teste
test_data = load("-ascii", "dataset_uci/final_X_test.txt");

% mantem todos os dados de X em apenas uma matriz
X_data = [train_data; test_data];

% leitura das classes de treinamento
train_classes = load("-ascii", "dataset_uci/final_Y_train.txt");

% leitura das classes de teste
test_classes = load("-ascii", "dataset_uci/final_Y_test.txt");

% mantem todos os dados de Y em apenas uma matriz
Y_data = [train_classes; test_classes];

printf("\nDados carregados!\n\nRemovendo dados redundantes da base de dados...\n");
fflush(stdout);
% seleciona apenas as amostras nao repetidas sem os elementos Y (sem duplicacao de classes)
[elem, ind] = unique(X_data, 'rows');

printf("\nTivemos uma reducao de %d amostras\nO que equivale a %f %%\n", length(X_data) - length(elem), 100 - (length(elem)/length(X_data)*100));

% concatena matriz de amostra X com suas classes
all_data = [elem, Y_data(ind)];

printf("\nCalculando matriz de correlacao dos atributos...\n");
fflush(stdout);

R = corr(all_data(:,1:end-1));

printf("\nCorrelacoes calculadas\n");

% retira a diagonal principal e o triangulo superior
R = tril(R, -1);

printf("\nDeletando colunas com indice de correlacao muito alto...\n");
fflush(stdout);
delete = [];
% percorre apenas triangulo inferior da matriz de correlacoes
for i = 2:rows(R)
	for j = 1:i
    % verifica se a correlação é maior que 0.9 ou menor que -0.9 e verifica se coluna ja não foi deletada
		if((R(i,j) >= .9 || R(i,j) <= -0.9) && ~ismember(j, delete))
      % insere na lista de colunas a serem deletadas se satisfaz a condição
      delete = [delete; j];
		endif
	end
end

% deleta as colunas de all_data com base nas correlacoes encontradas
all_data(:, delete) = [];
printf("\nDe %d colunas antes do calculo de correlacoes, agora restaram %d colunas de atributos\n", (length(R)+1), columns(all_data));