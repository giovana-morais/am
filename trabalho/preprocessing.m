% leitura dos dados de treinamento
train_data = load("-ascii", "dataset_uci/final_X_train.txt");

% leitura das classes de treinamento
train_classes = load("-ascii", "dataset_uci/final_Y_train.txt");

% une dados de treinamento com classes de treinamento
train_data = [train_data, train_classes];

% leitura dos dados de teste
test_data = load("-ascii", "dataset_uci/final_X_test.txt");

% leitura das classes de teste
test_classes = load("-ascii", "dataset_uci/final_Y_test.txt");

% une dados de teste com classes de teste
test_data = [test_data, test_classes];

% mantem todos os dados em apenas uma matriz
all_data = [train_data; test_data];

% seleciona apenas as amostras nao repetidas da matriz
A = unique(all_data, 'rows');