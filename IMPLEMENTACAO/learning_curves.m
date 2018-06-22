% Grupo 7 - Reconhecimento de Atividades Humanas
pkg load statistics
if(strcmp(computer(), "x86_64-pc-linux-gnu") )
  addpath("./libsvm/libsvm_x86_64-pc-linux-gnu");
elseif(ispc())
  addpath(".\libsvm\libsvm_windows");
endif

addpath("./ann");
addpath("./regression");

printf("Iniciando execucao.\n");
clear all, clc, close all;

% inicialmente ja fizemos pre-processamento dos dados para tentar diminuir a dimens�o de atributos e amostras
% retirando inconsistencias, redundancias, c�lulas nulas e fazendo a normalizacao de valores
% agora recuperamos os dados pre_processados que foram salvos no arquivo pre_processed

printf("\nCarregando dados pre-processados...\n");
fflush(stdout);


if(exist ("./data/pre_processed.mat.zip", "file") )
  printf("\nCarregando dados pre-processados...\n");
  fflush(stdout);
  load("./data/pre_processed.mat.zip");
else
  printf("Detectado que o pre-processamento ainda nao fora executado, aguarde, pre-processando....\n\n");
  preprocessing;
endif

if(exist ("./data/data_pca.mat.zip", "file") )
  load("./data/data_pca.mat.zip");
  printf("Dados do PCA Carregados !!!\n\n");
else
  # passando os dados pelo PCA para reduzir os atributos e utilizar em algoritmos que demandam mais processamento
  data_pca = pca(all_data);
  save("-zip", "./data/data_pca.mat.zip", "data_pca");
endif


% proveniente do 10-fold cross validation utilizado para a escolha de melhor fold
ksize = floor(rows(all_data)/10);

ktest = [];
ktrain = [];
ktest_pca = [];
ktrain_pca = [];

ktam = floor((rows(all_data)-rows(ktest))/20);

k = 1;
lambda_rl = 1;
hidden_neurons_rn1 = 75;
max_iter_rn1 = 200;
lambda_rn1 = 0.1;
hidden_neurons_rn2 = 75;
max_iter_rn2 = 500;
lambda_rn2 = 0.1;
c = 1;
gamma = 1;

for tam = 1:20
    cur_ktrain = ktrain(1:ktam*tam, :);
    cur_ktrain_pca = ktrain_pca(1:ktam*tam, :);

    % knn com melhor k encontrado com grid search ----------------------------------------------------------
    for i = 1:rows(ktest)
      ypred_knn_test(i) = knn(cur_ktrain(:,1:end-1), cur_ktrain(:,end), ktest(i,1:end-1), k);
    endfor

    for i = 1:rows(cur_ktrain)
      ypred_knn_train(i) = knn(cur_ktrain(:,1:end-1), cur_ktrain(:,end), cur_ktrain(i,1:end-1), k);
    endfor
    
    knncost_test(tam) = immse(ypred_knn_test, ktest(:, end));
    knncost_train(tam) = immse(ypred_knn_train, cur_ktrain(:,end));

    % regressao logistica com melhor lambda encontrado com grid search ---------------------------------------------------
 
    ypred_rl_test = regression(cur_ktrain(:,1:end-1), cur_ktrain(:,end), ktest(:,1:end-1), lambda_rl);

    ypred_rl_train = regression(cur_ktrain(:,1:end-1), cur_ktrain(:,end), cur_ktrain(:,1:end-1), lambda_rl);
    
    rlcost_test(tam) = immse(ypred_rl_test, ktest(:, end));
    rlcost_train(tam) = immse(ypred_rl_train, cur_ktrain(:,end));

    % redes neurais com o melhor lambda, max_iter e hidden_neurons encontrados com grid search -------------------------------------

    ypred_rn1_test = neural_network_1l(hidden_neurons_rn1, max_iter_rn1, cur_ktrain_pca, ktest_pca, lambda_rn1);
    ypred_rn1_train = neural_network_1l(hidden_neurons_rn1, max_iter_rn1, cur_ktrain_pca, cur_ktrain_pca, lambda_rn1);
    
    rn1cost_test(tam) = immse(ypred_rn1_test, ktest(:, end));
    rn1cost_train(tam) = immse(ypred_rn1_train, cur_ktrain(:,end));
    
    % redes neurais com o melhor lambda, max_iter e hidden_neurons encontrados com grid search -------------------------------------

    ypred_rn2_test = neural_network_2l(hidden_neurons_rn2, max_iter_rn2, cur_ktrain_pca, ktest_pca, lambda_rn2);
    ypred_rn1_train = neural_network_1l(hidden_neurons_rn2, max_iter_rn2, cur_ktrain_pca, cur_ktrain_pca, lambda_rn2);
    
    rn2cost_test(tam) = immse(ypred_rn2_test, ktest(:, end));
    rn2cost_train(tam) = immse(ypred_rn2_train, cur_ktrain(:,end));

    % svm com melhor c e gamma encontrados com grid search -----------------------------------------
    
    [ypred_svm_test, ~, ~] = svm(cur_ktrain_pca(:,1:end-1), cur_ktrain_pca(:,end), ktest_pca, c, g);

    [ypred_svm_train, ~, ~] = svm(cur_ktrain_pca(:,1:end-1), cur_ktrain_pca(:,end), cur_ktrain_pca, c, g);
    
    svmcost_test(tam) = immse(ypred_svm_test, ktest(:, end));
    svmcost_train(tam) = immse(ypred_svm_train, cur_ktrain(:,end));
endfor

plotCost(knncost_test, knncost_train, 'Curva de aprendizado para o KNN');
plotCost(rlcost_test, rlcost_train, 'Curva de aprendizado para a Regressao Linear');
plotCost(rn1cost_test, rn1cost_train, 'Curva de aprendizado para Redes Neurais Artificiais com 1 camada oculta');
plotCost(rn2cost_test, rn2cost_train, 'Curva de aprendizado para Redes Neurais Artificiais com 2 camadas ocultas');
plotCost(svmcost_test, svmcost_train, 'Curva de aprendizado para o SVM');