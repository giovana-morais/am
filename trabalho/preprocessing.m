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

printf("Dados carregados! \nRemovendo dados redundantes da base de dados...");
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