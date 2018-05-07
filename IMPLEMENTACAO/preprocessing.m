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

% codigo para retirada de outliers (nao esta sendo usado, pois ha muitos outliers)
%delete = [];
%for i = 1:(columns(all_data)-1)
%  column = all_data(:,i);
%  q1 = prctile(column, 25);
%  q3 = prctile(column, 75);
%  range_limit = (q3 - q1)*3;
  
  % maior ou menor que barreiras externas interquartis + percentil (25 ou 75)
%  gorl = union(find(column >= q3+range_limit), find(column <= q1-range_limit));
  
  % reune os que podem ser deletados
%  delete = [delete; setdiff(gorl, delete)];
%end

% por fim, remove todos as amostras que possuem colunas com outliers
%all_data(delete, :) = [];
    
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

% pega colunas que não serão deletadas para checagem
notdeleted = setdiff([1:columns(all_data)], delete);

% deleta as colunas de all_data com base nas correlacoes encontradas
all_data(:, delete) = [];
printf("\nDe %d colunas antes do calculo de correlacoes, agora restaram %d colunas de atributos:\n", (length(R)+1), columns(all_data));
printf("\n%d %d %d %d %d %d %d %d %d", notdeleted);
printf("\n");

% para checagem de balanceamento de dados
printf("\nPara checarmos o balanceamento dos dados atuais, temos que:\n");
printf("Para Y = 1, existem %d amostras\n", length(find(all_data(:,end) == 1)));
printf("Para Y = 2, existem %d amostras\n", length(find(all_data(:,end) == 2)));
printf("Para Y = 3, existem %d amostras\n", length(find(all_data(:,end) == 3)));
printf("Para Y = 4, existem %d amostras\n", length(find(all_data(:,end) == 4)));
printf("Para Y = 5, existem %d amostras\n", length(find(all_data(:,end) == 5)));
printf("Para Y = 6, existem %d amostras\n", length(find(all_data(:,end) == 6)));